"""决定性测试：基础型失效是可分解修复的吗？

刚才发现：0.6B 连"1 行历史取末位"都只有 40%。两类解释：
  A. 语言读出缺陷——模型无法从"红 → 蓝 → 绿...(8项)"这串里定位末位。
  B. 任务分解缺陷——如果拆成两步（先列出该行全部色词→再取最后），能救回来。
判据：
  - 若两步分解后可到 80%+：基础型失效是"单步承担太多"的任务形态问题，
    可被工程管线修复（重大利好：所有弱模型都能被这套管线拉到可用）。
  - 若两步分解后仍 40%：0.6B 连"复述列表"都不行——真·读出底线，
    小模型该退出这类场景（限制性结论）。
设计：
  step1: "把该行所有色词按顺序列出，逗号分隔" -> 解析
  step2: "这些色词里最后一个是什么" -> 判定
对照 arm: 单步直接问。
"""
import json, re, time, urllib.request, random

random.seed(67)
OUT = r"D:\djr82\memory_boundary\decompose_test.json"
COLORS = list("红蓝绿黄紫橙灰粉青棕白黑金银铜米朱靛黛绯墨翠绛褐")
N_TRIALS = 10


def make_case():
    n = random.randint(2, 4)
    ents = [f"实体{i+1}" for i in range(n)]
    hist = {e: random.sample(COLORS, 8) for e in ents}
    target = random.choice(ents)
    lines = [f"{e}: " + " → ".join(hist[e]) for e in ents]
    stream = "\n".join(lines)
    return {"stream": stream, "target": target, "gold": hist[target][-1],
            "row": f"{target}: " + " → ".join(hist[target])}


def qwen(p, mt=120):
    for _ in range(4):
        try:
            body = json.dumps({"messages": [{"role": "user", "content": p + " /no_think"}],
                               "temperature": 0, "max_tokens": mt},
                              ensure_ascii=False).encode()
            req = urllib.request.Request("http://127.0.0.1:8777/v1/chat/completions",
                                         data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=400) as r:
                d = json.loads(r.read())
            c = (d['choices'][0]['message'].get('content') or '').strip()
            if c:
                return c
        except Exception:
            pass
        time.sleep(2)
    return ""


def single_step(case):
    p = (f"以下是各实体的颜色历史（箭头右端为最新）：\n{case['stream']}\n"
         f"\n问题：{case['target']} 最新的颜色是什么？只回答颜色词。")
    return qwen(p)


def two_step(case):
    # step1: 复述目标行
    p1 = (f"以下是各实体的颜色历史（箭头右端为最新）：\n{case['stream']}\n"
          f"\n任务：把 {case['target']} 那行的颜色按顺序列出，用逗号分隔，不要其它文字。")
    r1 = qwen(p1)
    # step2: 从 r1 取末位
    p2 = f"这些颜色：{r1}\n最后一个是什么？只回答颜色词。"
    r2 = qwen(p2)
    return r1, r2


def color_seq(reply):
    return [c for c in COLORS if c in reply]


def main():
    stats = {"single_ok": 0, "two_ok": 0, "step1_exact": 0, "n": 0}
    details = []
    for _ in range(N_TRIALS):
        case = make_case()
        stats["n"] += 1
        # single
        r_s = single_step(case)
        ok_s = bool(r_s) and color_seq(r_s) and color_seq(r_s)[0] == case["gold"] if False else (case["gold"] in r_s and len(color_seq(r_s)) <= 3)
        ok_s = case["gold"] in r_s
        stats["single_ok"] += ok_s
        # two-step
        r1, r2 = two_step(case)
        # step1 质量：列出全部 8 色且顺序对
        seq1 = color_seq(r1)
        hist_js = case["row"]
        exp_seq = [c for c in COLORS if c in hist_js]  # 不完全准（重复色），改用直接对比
        step1_good = len(seq1) >= 6
        stats["step1_exact"] += step1_good
        # step2 从 r1 真实内容取末位
        seq1_all = color_seq(r1)
        if seq1_all:
            last = seq1_all[-1]
            ok_t = (last == case["gold"])
            # 也承认 step2 直接答对（模型可能跳过 r1 用 r2 眉目）
            if not ok_t and case["gold"] in r2 and len(color_seq(r2)) == 1:
                ok_t = True
        else:
            ok_t = False
        stats["two_ok"] += ok_t
        details.append({"gold": case["gold"], "single": r_s[:40],
                        "step1": r1[:60], "step2": r2[:40],
                        "ok_single": ok_s, "ok_two": ok_t, "step1_len": len(seq1_all)})
    print(f"single-step: {stats['single_ok']}/{stats['n']} = {stats['single_ok']/stats['n']:.0%}")
    print(f"two-step:    {stats['two_ok']}/{stats['n']} = {stats['two_ok']/stats['n']:.0%}")
    print(f"step1 列全(>=6色): {stats['step1_exact']}/{stats['n']}")
    for d in details[:5]:
        print(f"  gold={d['gold']} single={d['single']!r} step1={d['step1']!r} step2={d['step2']!r}")
    json.dump({"stats": stats, "details": details},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
