"""L* 曲线实验 v2：修复评分器子串碰撞（天蓝/蓝、藏青/青）后重扫。

修复：
  1. 色池换 24 个无子串关系的单字色（红蓝绿...绛褐）
  2. 评分器：精确提取回复中的全部色词（单字），多重命中时按首次出现位置判定
  3. 任何非 correct 结果记录原始回复，防再次被假象骗
"""
import json, re, time, urllib.request, random

random.seed(23)
OUT = r"D:\djr82\memory_boundary\Lstar_curve_v2.json"
KEY_SRC = r"D:/djr82/Documents/trae/6a8dbce9f0e28e9922d54185/hkt_login/audit_labels.py"

# 24 个互不为子串的单字颜色
COLORS = list("红蓝绿黄紫橙灰粉青棕白黑金银铜米朱靛黛绯墨翠绛褐")
assert len(COLORS) == 24

LOADS = [(8, 8, 64), (16, 8, 128), (16, 16, 256), (32, 16, 512)]
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
    """精确评分：提取回复中出现的全部色词（单字色，无子串碰撞）。
    - latest 出现且无其它色词 → correct
    - 多色词 → 按首次出现位置判 correct/ambig
    - 只有历史色 → stale；无色词 → other"""
    if not reply:
        return "empty", ""
    hits = [c for c in COLORS if c in reply]
    if latest in hits:
        if len(hits) == 1:
            return "correct", reply[:60]
        first = min(hits, key=lambda c: reply.find(c))
        if first == latest:
            return "correct", reply[:60]
        return "ambig", reply[:60]
    if hits:
        if any(h in history for h in hits):
            return "stale", reply[:60]
        return "other", reply[:60]
    return "other", reply[:60]


def run_point(model_name, fn, n_ent, n_upd, load, arm):
    trials = make_trials(n_ent, n_upd)
    c = {"correct": 0, "stale": 0, "ambig": 0, "other": 0, "empty": 0}
    samples = []
    t0 = time.time()
    for t in trials:
        p = plain_prompt(t) if arm == "plain" else mask_prompt(t)
        r = fn(p)
        s, raw = score(r, t['latest'], t['history'])
        c[s] += 1
        if s != "correct" and len(samples) < 3:
            samples.append({"latest": t['latest'], "got": raw, "kind": s})
    n = sum(c.values())
    dt = time.time() - t0
    print(f"  [{model_name:8s}|{arm:5s}|load={load:3d}] "
          f"correct {c['correct']}/{n} = {c['correct']/n:.0%} "
          f"stale={c['stale']} ambig={c['ambig']} other={c['other']} empty={c['empty']} "
          f"({dt:.0f}s)", flush=True)
    for s in samples:
        print(f"      miss: latest={s['latest']} kind={s['kind']} reply={s['got']!r}")
    return {"n_ent": n_ent, "n_upd": n_upd, "load": load, "arm": arm,
            "correct": c["correct"], "n": n, "detail": c,
            "samples": samples, "seconds": round(dt)}


def main():
    out = {"loads": LOADS, "points": []}
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
    print("\n=== L* 曲线 (correct%) ===")
    for model in ("deepseek", "qwen"):
        for arm in ("plain", "mask"):
            pts = [(p["load"], p["correct"] / p["n"]) for p in out["points"]
                   if p["model"] == model and p["arm"] == arm]
            line = "  ".join(f"{l}:{r:.0%}" for l, r in sorted(pts))
            print(f"  {model:8s}|{arm:5s}: {line}")


if __name__ == "__main__":
    main()
