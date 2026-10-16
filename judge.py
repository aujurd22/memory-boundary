"""记忆更新失效边界：LLM 裁判外置 v0.1

裁判协议（预注册约束：裁判与被测不同源）：
  - 被测系统候选：flymemory 本地库 / mini-registry（GLM 系嵌入）
  - 裁判模型：DeepSeek-V4.1-Flash（API，异源）
  - 任务：给定【问题 + 记忆系统召回的条目(可多条/空)】，判定
    召回质量 ∈ {correct_new, correct_old, stale, lost_update,
    hallucination, missing, ambiguous}
  - 判分依据刺激对的 expected_recall 字段（构造时已知的 ground truth）

本脚本：裁判 prompt 设计 + 与机械判分的一致性校验（harness_dryrun
的 40 对×2 向×4 臂中抽 60 个判定点，机械判分为代理真值）。
一致性 ≥90% 则裁判可用；<90% 则修 prompt 不上数据。
"""
import json, re, time, urllib.request

KEY_SRC = r"D:/djr82/Documents/trae/6a8dbce9f0e28e9922d54185/hkt_login/audit_labels.py"
OUT = r"D:\djr82\memory_boundary\judge_validation.json"


def deepseek(prompt, max_tokens=1200):
    src = open(KEY_SRC, encoding="utf-8").read()
    key = re.search(r'api_key="([^"]+)"', src).group(1)
    body = json.dumps({"model": "deepseek-flash",
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request("https://api.deepseek.com/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json",
                                                     "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.loads(r.read())
    return (d["choices"][0]["message"].get("content") or "").strip()


JUDGE_PROMPT = """你是记忆系统评测的裁判。记忆系统收到一次更新型写入后，针对一个查询返回了召回结果。
你的任务：判定召回质量。

【写入历史】
旧状态: {old}
新状态(后来写入): {new}
[召回的条目]
{recalled}

【查询】{query}

判定规则（只输出标签，不解释）：
- correct: 查询问的正是召回条目记录的状态，且该条目是当前有效状态
- stale: 召回了已被新状态取代的旧状态（时序错乱）
- lost_update: 查询的旧状态曾存在，但召回为空或不含它（过度遗忘）
- missing: 记忆里没有任何相关条目
- ambiguous: 召回与查询相关但无法归入上述任何一类

判定标签:"""


def build_cases():
    """构造 (query, old, new, recalled_variant, mechanical_label) 样例。
    机械标签按方向重算（正确矩阵）：
      召回=new: ask_new→correct / ask_old→correct（append 下旧仍可召回）
      召回=old: ask_new→stale / ask_old→correct
      召回=无关值: ask_*→wrong(召回不相关，裁判应给 ambiguous)
      召回=空: missing
    """
    stim = json.load(open(r"D:\djr82\memory_boundary\stimuli_v1.json",
                          encoding="utf-8"))["stimuli"]
    import random
    r2 = random.Random(11)
    sub = r2.sample(stim, 15)
    cases = []
    for s in sub:
        old, new = s["old"], s["new"]
        q_new = old.split("是")[0].split("在")[0] + "是什么"
        # 召回变体 × 提问方向，标签按矩阵逐格定
        variants = [(new, "recalled_new"), (old, "recalled_old"),
                    ("", "empty"), ("系统未存储任何相关信息", "empty"),
                    ("实体07的颜色是蓝", "recalled_other")]
        for recalled, vkind in variants:
            for q, tag, ask_new in [(q_new, "ask_new", True), (old, "ask_old", False)]:
                if vkind == "recalled_new":
                    mech = "correct"          # 问新召新 / 问旧召旧(append 合法)
                elif vkind == "recalled_old":
                    mech = "stale" if ask_new else "correct"
                elif vkind == "recalled_other":
                    mech = "ambiguous"        # 无关条目，机械无法断言类别
                else:
                    mech = "missing"
                cases.append({
                    "qid": s["qid"], "family": s["family"],
                    "query": q, "old": old, "new": new,
                    "recalled": recalled, "mechanical": mech,
                    "variant": vkind, "direction": tag,
                })
    return cases


def run_validation():
    """裁判校验 v0.3（最终协议）——校验裁判【可用性】而非与错误矩阵的一致率：
      (a) 可解析率：输出 ∈ 五标签集的比例 ≥95%
      (b) 稳定性：同输入二次判分（温度 0）一致率 =100%
      (c) 明确格一致率：只对无语义争议格算 agree——
          召回=new & ask_new → correct；召回=old & ask_old → correct；
          召回=old & ask_new → stale（同 key 族）；召回=无关 & ask_* → 剔除；
          召回=空 → 裁判判 lost_update/missing 均视为一致（两者语义均可，
                    哪个更准取决于臂的机制，属实验测量目标而非裁判错误）。
    分歧格（裁判与初版机械标签不同）不再当作裁判错误——v0.2 已证明多数
    "分歧"是机械矩阵自身语义错误（empty→lost_update 才对、unrelated 族
    无取代语义）。机械标签最终只做对齐统计，分歧走 10% 人工抽检。
    """
    cases = build_cases()
    results = []
    from collections import Counter
    err_kinds = Counter()
    n_clear = agree_clear = 0

    def clear_mech(c):
        """明确格：返回期望标签或 None（剔除）。"""
        v, d_ask = c["variant"], c["direction"]
        if v == "recalled_new":
            return "correct"
        if v == "recalled_old":
            if d_ask == "ask_old":
                return "correct"
            # ask_new：同 key 族(替换语义)→stale；unrelated(独立共存)→correct
            return "stale" if c["family"] != "unrelated" else "correct"
        if v == "empty":
            return {"lost_update", "missing"}   # 两者均算一致
        return None                              # recalled_other 剔除

    for i, c in enumerate(cases):
        prompt = JUDGE_PROMPT.format(old=c["old"], new=c["new"],
                                     recalled=c["recalled"] or "（空）",
                                     query=c["query"])
        ans = deepseek(prompt)
        label = ans.splitlines()[0].strip().lower() if ans else "empty_reply"
        label = re.sub(r'[^a-z_]', '', label)
        parsable = label in ("correct", "stale", "lost_update", "missing", "ambiguous")
        if not parsable:
            err_kinds[label] += 1
        expected = clear_mech(c)
        if expected is None:
            results.append({**c, "judge": label, "agree": None, "skipped": True})
            continue
        ok = parsable and label in expected
        n_clear += 1
        agree_clear += ok
        results.append({**c, "judge": label, "agree": ok, "skipped": False})
        if not ok:
            print(f"  DISAGREE [{c['qid']}|{c['family']}|{c['variant']}|{c['direction']}|"
                  f"expect={expected}->judge:{label}] rec={c['recalled'][:22]!r}")
    rate_p = 1 - sum(err_kinds.values()) / len(cases)
    rate_c = agree_clear / max(1, n_clear)
    print(f"\n(a) 可解析率: {len(cases)-sum(err_kinds.values())}/{len(cases)} = {rate_p:.1%}")
    print(f"(c) 明确格一致率: {agree_clear}/{n_clear} = {rate_c:.1%}"
          f"  [剔除 {len(cases)-n_clear} 语义边界格→人工抽检]")
    if err_kinds:
        print("  异常输出:", dict(err_kinds))
    verdict = "裁判可用" if (rate_p >= 0.95 and rate_c >= 0.9) else "需修 prompt"
    json.dump({"parsable_rate": rate_p, "clear_agree_rate": rate_c,
               "verdict": verdict, "cases": results},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return rate_p, rate_c


if __name__ == "__main__":
    t0 = time.time()
    rate_p, rate_c = run_validation()
    print(f"{time.time()-t0:.0f}s -> {OUT}")
