"""L* 曲线实验：读出失效前沿 L*(load) 扫描

问题：读出失效在什么负载点出现？结构修复（遮蔽）把 L* 推多远？
规模（0.6B vs 前沿 API）与结构，谁支配 L*？

设计：
  - 负载轴: 64 / 128 / 256 / 512 记录（实体数×版本数组合逼近）
  - 模型: qwen-0.6B（本地） / deepseek-flash（API）
  - 臂:   plain（原样流） / mask（竞争项遮蔽）
  - 每负载点 8 trials；记录 token 量
判据：各 (model, arm) 的 correct 率 vs 负载 → 画 L* 曲线
"""
import json, re, time, urllib.request, random

random.seed(23)
OUT = r"D:\djr82\memory_boundary\Lstar_curve.json"
KEY_SRC = r"D:/djr82/Documents/trae/6a8dbce9f0e28e9922d54185/hkt_login/audit_labels.py"
COLORS = ["红","蓝","绿","黄","紫","橙","灰","粉","青","棕","白","黑",
          "金","银","铜","米","天蓝","草绿","玫红","藏青","杏","绛","黛","雪"]
LOADS = [
    # (n_ent, n_upd, 总记录)
    (8, 8, 64),
    (16, 8, 128),
    (16, 16, 256),
    (32, 16, 512),
]
N_TRIALS = 8


def make_trials(n_ent, n_upd, n=N_TRIALS):
    trials = []
    for _ in range(n):
        ents = [f"节点{i+1:02d}" for i in range(n_ent)]
        upd = {e: random.sample(COLORS, n_upd) for e in ents}
        seq = []
        for j in range(n_upd):
            layer = [(e, j) for e in ents]
            random.shuffle(layer)
            seq += layer
        target = random.choice(ents)
        trials.append({"seq": seq, "updates": upd, "target": target,
                       "latest": upd[target][-1], "history": upd[target][:-1]})
    return trials


def plain_prompt(t):
    lines = [f"[{i+1:03d}] {e} 的颜色更新为{t['updates'][e][j]}。"
             for i, (e, j) in enumerate(t['seq'])]
    return ("以下是系统记录流（时间顺序）：\n" + "\n".join(lines) +
            f"\n\n问题：{t['target']} 现在最新的颜色是什么？\n只回答颜色词。")


def mask_prompt(t):
    lines = []
    for i, (e, j) in enumerate(t['seq']):
        if e == t['target'] and j == len(t['updates'][e]) - 1:
            lines.append(f"[{i+1:03d}] {e} 的颜色更新为{t['updates'][e][j]}。")
        else:
            lines.append(f"[{i+1:03d}] （记录已遮蔽）")
    return ("以下是系统记录流（时间顺序）：\n" + "\n".join(lines) +
            f"\n\n问题：{t['target']} 现在最新的颜色是什么？\n只回答颜色词。")


def qwen(p, mt=60):
    for _ in range(5):
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


def deepseek(p, mt=1500):
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


def score(reply, latest, history):
    if not reply:
        return "empty"
    hn = latest in reply
    ho = any(h in reply and h != latest for h in history)
    if hn and not ho:
        return "correct"
    if hn and ho:
        m = re.search(r'([红蓝绿黄紫橙灰粉青棕白黑])', reply)
        return "correct" if m and m.group(1) == latest else "ambig"
    if ho:
        return "stale"
    return "other"


def run_point(model_name, fn, n_ent, n_upd, load, arm):
    trials = make_trials(n_ent, n_upd)
    c = {"correct": 0, "stale": 0, "ambig": 0, "other": 0, "empty": 0}
    t0 = time.time()
    for t in trials:
        p = plain_prompt(t) if arm == "plain" else mask_prompt(t)
        r = fn(p)
        c[score(r, t['latest'], t['history'])] += 1
    n = sum(c.values())
    dt = time.time() - t0
    print(f"  [{model_name:8s}|{arm:5s}|load={load:3d}] "
          f"correct {c['correct']}/{n} = {c['correct']/n:.0%} "
          f"stale={c['stale']} empty={c['empty']} ({dt:.0f}s)", flush=True)
    return {"n_ent": n_ent, "n_upd": n_upd, "load": load, "arm": arm,
            "correct": c["correct"], "n": n, "detail": c,
            "seconds": round(dt)}


def main():
    out = {"loads": LOADS, "points": []}
    # 先跑 deepseek（主力），0.6B 只跑 64/128/256（512 记录超 8K ctx 会截断）
    for n_ent, n_upd, load in LOADS:
        for arm in ("plain", "mask"):
            pt = run_point("deepseek", deepseek, n_ent, n_upd, load, arm)
            out["points"].append({"model": "deepseek", **pt})
    for n_ent, n_upd, load in LOADS[:3]:
        for arm in ("plain", "mask"):
            pt = run_point("qwen", qwen, n_ent, n_upd, load, arm)
            out["points"].append({"model": "qwen", **pt})
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved ->", OUT)
    # 画 ASCII 曲线
    print("\n=== L* 曲线 (correct%) ===")
    for model in ("deepseek", "qwen"):
        for arm in ("plain", "mask"):
            pts = [(p["load"], p["correct"] / p["n"]) for p in out["points"]
                   if p["model"] == model and p["arm"] == arm]
            line = "  ".join(f"{l}:{r:.0%}" for l, r in sorted(pts))
            print(f"  {model:8s}|{arm:5s}: {line}")


if __name__ == "__main__":
    main()
