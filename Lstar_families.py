"""L* 多任务族扫描 v2：四种任务形态找失效区（干净重写版）

任务族：
  F1 track    —— 追踪：某实体最新颜色（基准，上轮已测）
  F2 multi    —— 两跳：上级组的最新颜色（先查关系再查状态）
  F3 temporal —— 时序回溯：第 k 次更新的颜色
  F4 count    —— 聚合：最终颜色为 X 的实体数量

每族：确定性生成器 + 真值；两负载 × 两模型。
"""
import json, re, time, urllib.request, random

random.seed(53)
OUT = r"D:\djr82\memory_boundary\Lstar_families.json"
KEY_SRC = r"D:/djr82/Documents/trae/6a8dbce9f0e28e9922d54185/hkt_login/audit_labels.py"
COLORS = list("红蓝绿黄紫橙灰粉青棕白黑金银铜米朱靛黛绯墨翠绛褐")


def gen_track(n_ent, n_upd, n=5):
    trials = []
    for _ in range(n):
        ents = [f"节点{i+1:03d}" for i in range(n_ent)]
        upd = {e: random.sample(COLORS, n_upd) for e in ents}
        seq = []
        for j in range(n_upd):
            layer = [(e, j) for e in ents]
            random.shuffle(layer)
            seq += layer
        target = random.choice(ents)
        trials.append({"upd": upd, "seq": seq,
                       "question": f"{target} 现在最新的颜色是什么？",
                       "gold": upd[target][-1]})
    return trials


def gen_multi(n_ent, n_upd, n=5):
    trials = []
    for _ in range(n):
        groups = [f"组{i+1:03d}" for i in range(n_ent)]
        children = [f"节点{i+1:03d}" for i in range(4)]
        parent = {c: random.choice(groups) for c in children}
        upd = {g: random.sample(COLORS, n_upd) for g in groups}
        seq = [("REL", c, parent[c]) for c in children]
        for j in range(n_upd):
            layer = [(g, j) for g in groups]
            random.shuffle(layer)
            seq += [("COL", g, j) for g, j in layer]
        child = random.choice(children)
        g = parent[child]
        trials.append({"upd": upd, "seq": seq, "child": child, "group": g,
                       "question": f"{child} 属于哪个组？该组现在最新的颜色是什么？回答颜色。",
                       "gold": upd[g][-1]})
    return trials


def gen_temporal(n_ent, n_upd, n=5):
    trials = []
    for _ in range(n):
        ents = [f"节点{i+1:03d}" for i in range(n_ent)]
        upd = {e: random.sample(COLORS, n_upd) for e in ents}
        seq = []
        for j in range(n_upd):
            layer = [(e, j) for e in ents]
            random.shuffle(layer)
            seq += layer
        target = random.choice(ents)
        k = random.randint(1, n_upd)
        trials.append({"upd": upd, "seq": seq,
                       "question": f"{target} 第 {k} 次颜色更新是什么？回答颜色。",
                       "gold": upd[target][k - 1]})
    return trials


def gen_count(n_ent, n_upd, n=5):
    from collections import Counter
    trials = []
    attempts = 0
    while len(trials) < n and attempts < 50:
        attempts += 1
        ents = [f"节点{i+1:03d}" for i in range(n_ent)]
        upd = {e: random.sample(COLORS, n_upd) for e in ents}
        seq = []
        for j in range(n_upd):
            layer = [(e, j) for e in ents]
            random.shuffle(layer)
            seq += layer
        finals = [upd[e][-1] for e in ents]
        cnt = Counter(finals)
        tc, gold = cnt.most_common(1)[0]
        if gold < 2:
            continue
        trials.append({"upd": upd, "seq": seq,
                       "question": f"最终颜色是{tc}的实体一共有几个？回答数字。",
                       "gold": str(gold)})
    return trials


def render(t, family):
    parts = []
    if family == "multi":
        for item in t["seq"]:
            if item[0] == "REL":
                parts.append(f"[关系] {item[1]} 属于 {item[2]}。")
            else:
                _, g, j = item
                parts.append(f"[更新] {g} 的颜色更新为 {t['upd'][g][j]}。")
    else:
        for i, (e, j) in enumerate(t["seq"]):
            parts.append(f"[{i+1:04d}] {e} 的颜色更新为 {t['upd'][e][j]}。")
    return ("以下是记录流（时间顺序）：\n" + "\n".join(parts) +
            f"\n\n问题：{t['question']}")


def qwen(p, mt=60):
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
            with urllib.request.urlopen(req, timeout=240) as r:
                d = json.loads(r.read())
            c = (d['choices'][0]['message'].get('content') or '').strip()
            if c:
                return c
        except Exception:
            pass
        time.sleep(3)
    return ""


def correct(reply, gold, family):
    if not reply:
        return False
    if family == "count":
        m = re.search(r'\d+', reply)
        return bool(m) and m.group(0) == gold
    return gold in reply


def run_family(family, gen, n_ent, n_upd, model_name, fn, sample=5):
    trials = gen(n_ent, n_upd, sample)
    ok = 0
    misses = []
    t0 = time.time()
    for t in trials:
        r = fn(render(t, family))
        good = correct(r, t["gold"], family)
        ok += good
        if not good:
            misses.append({"q": t["question"][:70], "gold": t["gold"],
                           "got": r[:60]})
    dt = time.time() - t0
    rate = ok / max(1, len(trials))
    print(f"  [{model_name:8s}|{family:8s}|ent={n_ent:02d}|upd={n_upd:02d}|"
          f"n={len(trials)}] {ok}/{len(trials)} = {rate:.0%} ({dt:.0f}s)", flush=True)
    for m in misses[:2]:
        print(f"      MISS gold={m['gold']!r} got={m['got']!r} | {m['q']}")
    return {"family": family, "n_ent": n_ent, "n_upd": n_upd,
            "model": model_name, "ok": ok, "n": len(trials),
            "seconds": round(dt), "misses": misses}


CONFIGS = [
    ("track", gen_track, 16, 8),
    ("multi", gen_multi, 16, 8),
    ("temporal", gen_temporal, 16, 8),
    ("count", gen_count, 16, 8),
    ("track", gen_track, 32, 16),
    ("temporal", gen_temporal, 32, 16),
    ("multi", gen_multi, 32, 16),
    ("count", gen_count, 32, 16),
]


def main():
    out = {"points": []}
    for family, gen, n_ent, n_upd in CONFIGS:
        out["points"].append(run_family(family, gen, n_ent, n_upd, "deepseek", deepseek))
    for family, gen, n_ent, n_upd in CONFIGS[:4]:
        out["points"].append(run_family(family, gen, n_ent, n_upd, "qwen", qwen))
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved ->", OUT)
    print("\n=== 任务族 × 模型 汇总 ===")
    for p in out["points"]:
        print(f"  {p['model']:8s} {p['family']:8s} {p['n_ent']}x{p['n_upd']}: "
              f"{p['ok']}/{p['n']}")


if __name__ == "__main__":
    main()
