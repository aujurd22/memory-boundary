"""H7：把我们的刺激集喂给 Hy-Memory 的 reconciler prompt——跨系统对照实测

问题：TriLink 的 fact_reconciler（纯 LLM 规则决策，保守 INSERT 默认）
在我们相图的三轴上表现如何？与我们验证过的机制（簇内时间层级 89%）
对照。

方法：从 stimuli_v1.json 抽 60 对（四族×Δ箱分层），每对构造为
"existing session fact + new session fact"，用他们的 FACT_RECONCILE_PROMPT
原文（DeepSeek-V4.1-Flash 代跑 LLM 决策），判定输出 SUPERSEDE 还是 keep both。
ground truth 来自 expected_recall：
  attribute/flip → 应 SUPERSEDE（同一事实的状态更新）
  paraphrase     → 应 SUPERSEDE（同义改述=同一事实，他们的"目标实质相同"规则该判同）
  unrelated      → 应 keep（不同属性，主体相同但目标不同——他们的规则该判 parallel）

预测（基于他们 prompt 的保守默认）：
  paraphrase 对（同 key 同值不同措辞）：他们 prompt 无"同义等价"规则，
    LLM 大概率判 keep both → 他们的设计在我们相图的高 Δ 区失效
    （这正是我们 L4 失败定律的 LLM 侧形态）。
  attribute/flip（同 key 不同值）：主体匹配+目标延续 → 应正确 SUPERSEDE。
  unrelated：应正确 keep。
"""
import json, re, time, urllib.request
from collections import Counter, defaultdict

KEY_SRC = r"D:/djr82/Documents/trae/6a8dbce9f0e28e9922d54185/hkt_login/audit_labels.py"
OUT = r"D:\djr82\memory_boundary\h7_cross.json"

# 他们的 prompt 原文（从 fact_reconciler.py 抓的）
FACT_RECONCILE_PROMPT = """\
You maintain session-level WORKING FACTS. Each statement summarizes one whole
trace/session: the user's task, the rough approach, and the result. It is not an
atomic claim.

Every new session fact below WILL be stored; that is not your decision. Your only
job is to detect task continuation: is the new session continuing to advance the
SAME user task as an existing session fact? If so, SUPERSEDE the older fact so
the links form a cross-session progress chain for that task.

Mark an existing fact as superseded ONLY when there is strong evidence that:
  - The subjects match: both records are about the SAME entity — the same
    person, the same file/module, the same host/service. A shared subject is
    necessary but not sufficient; AND
  - The user goal is materially the SAME, not merely in the same topic/project;
    AND
  - The new trace continues, retries, extends, verifies, or finishes the work
    recorded by the existing trace.

Do NOT supersede when:
  - The two records name different subjects. Two people in one group, two files
    in one module, two hosts in one cluster are PARALLEL, not successive: one
    person's state is not a newer version of another's. Where a subject is shown
    as (unspecified), fall back to judging by the statement alone.
  - The sessions merely mention the same file, symbol, component, repository,
    error, or technology but pursue different user goals.
  - One task only supplies context or a technique for another task.
  - The relationship is uncertain. Be conservative: INSERT is the safe default;
    merging unrelated sessions destroys their separate task history.

Treat each statement as evidence-bounded text. Do not reinterpret an inference
inside it as an established cause or use a shared conjecture/error as proof that
the user goals are the same. Your decision is only about task continuation; do
not "correct", complete, or embellish either session record.

═══════════════════════════════════════════════════════════════════
NEW SESSION FACTS
═══════════════════════════════════════════════════════════════════
Draft 0: {new}

═══════════════════════════════════════════════════════════════════
EXISTING SESSION FACTS (per new fact, retrieved by statement similarity)
═══════════════════════════════════════════════════════════════════
fact_a: {old}

═══════════════════════════════════════════════════════════════════
OUTPUT (strict JSON only, no markdown)
═══════════════════════════════════════════════════════════════════
{{
  "ops": [
    {{"draft_idx": 0, "supersedes": ["fact_a"], "reason": "..."}},
  ]
}}
If nothing is superseded, output {{"ops": []}}.
"""


def deepseek(prompt, max_tokens=1200):
    src = open(KEY_SRC, encoding="utf-8").read()
    key = re.search(r'api_key="([^"]+)"', src).group(1)
    body = json.dumps({"model": "deepseek-flash",
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request("https://api.deepseek.com/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json",
                                                     "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return (d["choices"][0]["message"].get("content") or "").strip()


def expected(s):
    """我们的 ground truth：该 SUPERSEDE 还是 keep both。"""
    return {"attribute": "SUPERSEDE", "flip": "SUPERSEDE",
            "paraphrase": "SUPERSEDE", "unrelated": "KEEP"}[s["family"]]


def judge_theirs(s, reply):
    """解析他们的输出，对照 ground truth。"""
    m = re.search(r'"supersedes"\s*:\s*\[([^\]]*)\]', reply)
    if not m:
        return "PARSE_FAIL"
    return "SUPERSEDE" if m.group(1).strip() else "KEEP"


def main():
    stim = json.load(open(r"D:\djr82\memory_boundary\stimuli_v1.json",
                          encoding="utf-8"))["stimuli"]
    # 分层抽样：每族 15 对 = 60 对
    import random
    rng = random.Random(7)
    by_fam = {}
    for s in stim:
        by_fam.setdefault(s["family"], []).append(s)
    sample = []
    for fam, lst in by_fam.items():
        sample += rng.sample(lst, 15)
    rng.shuffle(sample)

    results = []
    stats = defaultdict(Counter)
    for i, s in enumerate(sample):
        prompt = FACT_RECONCILE_PROMPT.format(old=s["old"], new=s["new"])
        for attempt in range(3):
            try:
                reply = deepseek(prompt)
                break
            except Exception as e:
                time.sleep(3)
        else:
            reply = ""
        verdict = judge_theirs(s, reply)
        exp = expected(s)
        ok = verdict == exp
        stats[s["family"]][("✓" if ok else "✗") + f"→{verdict}"] += 1
        results.append({"qid": s["qid"], "family": s["family"],
                        "cos": s["cos"], "expected": exp,
                        "theirs": verdict, "ok": ok,
                        "raw": reply[:150]})
        if not ok and len(results) <= 30:
            print(f"  MISS [{s['qid']}|{s['family']}|Δ={s['cos']:.2f}] "
                  f"exp={exp} theirs={verdict}")
    total = len(results)
    agree = sum(1 for r in results if r["ok"])
    print(f"\n他们的 reconciler vs 我们的 ground truth: {agree}/{total} = {agree/total:.0%}")
    for fam in ("attribute", "flip", "paraphrase", "unrelated"):
        print(f"  {fam:11s}: {dict(stats[fam])}")
    json.dump({"agree_rate": agree / total, "results": results},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
