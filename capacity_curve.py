"""读出容量曲线：提取区竞争行数 N 的扫描

发现演变：
  - pilot v2: 只给目标实体时间线（1 行）→ 0.6B 90%
  - Lstar_families 结构版: 给全部 16 实体时间线 → 0.6B 20%
差异变量 = 提取区的竞争行数 N。本实验扫 N = 1/2/4/8/16/32。
控制：总长度随 N 线性增长（每行 8 色历史）；任务恒为 track。
预测：存在容量拐点 N*（0.6B 小，DeepSeek 无或很大）。
"""
import json, re, time, urllib.request, random

random.seed(67)
OUT = r"D:\djr82\memory_boundary\capacity_curve.json"
KEY_SRC = r"D:/djr82/Documents/trae/6a8dbce9f0e28e9922d54185/hkt_login/audit_labels.py"
COLORS = list("红蓝绿黄紫橙灰粉青棕白黑金银铜米朱靛黛绯墨翠绛褐")
NS = [1, 2, 4, 8, 16, 32]
N_TRIALS = 5


def make_case(n_lines):
    """n_lines 条实体行（含目标），每行 8 色历史。"""
    ents = [f"实体{i+1:03d}" for i in range(n_lines)]
    hist = {e: random.sample(COLORS, 8) for e in ents}
    target = random.choice(ents)
    return {"ents": ents, "hist": hist, "target": target,
            "gold": hist[target][-1]}


def render(case, order="target_middle"):
    """行序：目标位置随机（防位置偏置），默认放中间。"""
    ents = list(case["ents"])
    tgt = case["target"]
    if order == "target_middle":
        ents.remove(tgt)
        mid = len(ents) // 2
        ents.insert(mid, tgt)
    lines = [f"{e}: " + " → ".join(case["hist"][e]) for e in ents]
    return ("以下是各实体的颜色历史（箭头右端为最新）：\n" + "\n".join(lines) +
            f"\n\n问题：{tgt} 最新的颜色是什么？只回答颜色词。")


def qwen(p, mt=40):
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


def deepseek(p, mt=600):
    src = open(KEY_SRC, encoding='utf-8').read()
    key = re.search(r'api_key="([^"]+)"', src).group(1)
    for _ in range(4):
        try:
            body = json.dumps({"model": "deepseek-flash",
                               "messages": [{"role": "user", "content": p}],
                               "temperature": 0, "max_tokens": mt}).encode()
            req = urllib.request.Request("https://api.deepseek.com/v1/chat/completions",
                                         data=body, headers={"Content-Type": "application/json",
                                                             "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            c = (d['choices'][0]['message'].get('content') or '').strip()
            if c:
                return c
        except Exception:
            pass
        time.sleep(3)
    return ""


def score(reply, gold):
    if not reply:
        return False, ""
    hits = [c for c in COLORS if c in reply]
    return (gold in hits and (len(hits) == 1 or min(hits, key=lambda c: reply.find(c)) == gold)), reply[:50]


def main():
    out = {"ns": NS, "points": []}
    for model_name, fn in [("qwen", qwen), ("deepseek", deepseek)]:
        print(f"{model_name}:")
        for n in NS:
            ok, misses = 0, []
            for _ in range(N_TRIALS):
                case = make_case(n)
                r = fn(render(case))
                good, raw = score(r, case["gold"])
                ok += good
                if not good:
                    misses.append({"gold": case["gold"], "got": raw})
            print(f"  N={n:2d}: {ok}/{N_TRIALS} = {ok/N_TRIALS:.0%}" +
                  (f"  misses={misses[:2]}" if misses else ""), flush=True)
            out["points"].append({"model": model_name, "n_lines": n,
                                  "ok": ok, "n": N_TRIALS, "misses": misses})
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
