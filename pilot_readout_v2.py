"""Pilot v2: 高强度读出干扰——40+ 记录、多实体交错、长距离版本。

v1 教训：5 记录场景两模型全对（任务太简单，干扰没咬）。
v2 设计（贴近真实 agent 会话规模）：
  - 12 个实体交错写入，共 48 条记录
  - 每个实体 4 次更新（颜色不断变化），查询某实体的"最新颜色"
  - 旧版本与新版本可能相隔 20+ 条记录
三臂：
  plain = 原样流
  label = 每个记录前贴【时间t】+ 实体的所有历史并列（TEM 式版本化：直接给出
          该实体版本列表，按序编号）——这是更强力结构：不是遮蔽，是重组
  mask  = 只保留该实体最新记录（理想结构，上界参考）
"""
import json, re, time, urllib.request, random

random.seed(7)
OUT = r"D:\djr82\memory_boundary\pilot_readout_v2.json"
KEY_SRC = r"D:/djr82/Documents/trae/6a8dbce9f0e28e9922d54185/hkt_login/audit_labels.py"

COLORS = ["红", "蓝", "绿", "黄", "紫", "橙", "灰", "粉", "青", "棕", "白", "黑"]


def make_trials(n_trials=10, n_entities=12, n_updates=4):
    """生成 trial：12 实体交错 48 条记录，查询目标实体的最新颜色。"""
    trials = []
    for t in range(n_trials):
        entities = [f"节点{i + 1:02d}" for i in range(n_entities)]
        # 每实体 4 次更新
        updates = {e: random.sample(COLORS, n_updates) for e in entities}
        # 交错写入
        order = [(e, j) for j in range(n_updates) for e in entities]
        random.shuffle_safe = None
        # 不彻底打乱：真实场景是同实体更新在时间上散布，用两轮交错
        seq = []
        for j in range(n_updates):
            layer = [(e, j) for e in entities]
            random.shuffle(layer)
            seq += layer
        # 目标
        target = random.choice(entities)
        latest = updates[target][-1]
        trials.append({"entities": entities, "updates": updates,
                       "seq": seq, "target": target, "latest": latest,
                       "history": updates[target][:-1]})
    return trials


def render(trial, arm):
    ent, seq, updates, target = (trial["target"], trial["seq"],
                                 trial["updates"], trial["target"])
    if arm == "plain":
        lines = [f"[{i + 1:02d}] {e} 的颜色更新为{updates[e][j]}。" for i, (e, j) in enumerate(seq)]
    elif arm == "mask":
        lines = []
        for i, (e, j) in enumerate(seq):
            if e == target and j == len(updates[e]) - 1:
                lines.append(f"[{i + 1:02d}] {e} 的颜色更新为{updates[e][j]}。")
        if not lines:
            lines = ["（无记录）"]
    elif arm == "timeline":
        # TEM 式：列出目标实体的版本时间线（标注个版本的时间戳）
        stamps = [i + 1 for i, (e, j) in enumerate(seq)
                  if e == target]
        lines = [f"{target} 的版本时间线（按时间序）："]
        for k, (col, st) in enumerate(zip(updates[target], stamps)):
            tag = "（最新）" if k == len(updates[target]) - 1 else ""
            lines.append(f"  时间{st:02d}: 颜色={col}{tag}")
    prompt = ("以下是系统的记录流（按时间顺序）：\n" + "\n".join(render_lines(seq, updates)
              if arm != "timeline" else "\n".join(lines)) +
              f"\n\n问题：{target} 现在（最新）的颜色是什么？\n只回答颜色词。").replace(
        "render_lines", "")
    if arm == "timeline":
        prompt = ("以下是一个实体的版本时间线：\n" + "\n".join(lines) +
                  f"\n\n问题：{target} 现在（最新）的颜色是什么？\n只回答颜色词。")
    return prompt


def render_lines(seq, updates):
    return [f"[{i + 1:02d}] {e} 的颜色更新为{updates[e][j]}。"
            for i, (e, j) in enumerate(seq)]


def qwen(prompt, mt=60):
    body = json.dumps({"messages": [{"role": "user", "content": prompt + " /no_think"}],
                       "temperature": 0, "max_tokens": mt},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8777/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        d = json.loads(r.read())
    return (d["choices"][0]["message"].get("content") or "").strip()


def deepseek(prompt, mt=800):
    src = open(KEY_SRC, encoding="utf-8").read()
    key = re.search(r'api_key="([^"]+)"', src).group(1)
    body = json.dumps({"model": "deepseek-flash",
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": mt}).encode()
    req = urllib.request.Request("https://api.deepseek.com/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json",
                                                     "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return (d["choices"][0]["message"].get("content") or "").strip()


def score(reply, latest, history):
    if not reply:
        return "empty"
    hit_new = latest in reply
    hit_old = any(h in reply and h != latest for h in history)
    if hit_new and not hit_old:
        return "correct"
    if hit_new and hit_old:
        m = re.search(r'([红蓝绿黄紫橙灰粉青棕白黑])', reply)
        return "correct" if m and m.group(1) == latest else "ambiguous"
    if hit_old and not hit_new:
        return "stale"
    return "other"


def run(name, fn, trials):
    stats = {}
    for arm in ("plain", "mask", "timeline"):
        c = {"correct": 0, "stale": 0, "ambiguous": 0, "empty": 0, "other": 0}
        for trial in trials:
            prompt = render(trial, arm)
            reply = ""
            for _ in range(3):
                try:
                    reply = fn(prompt)
                    if reply:
                        break
                except Exception:
                    time.sleep(2)
            c[score(reply, trial["latest"], trial["history"])] += 1
        stats[arm] = c
        n = sum(c.values())
        print(f"  {name:8s} {arm:8s}: {c['correct']}/{n} = {c['correct']/n:.0%}  {c}")
    return stats


def main():
    trials = make_trials()
    out = {}
    print("qwen3-0.6b (local):")
    out["qwen"] = run("qwen", qwen, trials)
    print("deepseek-flash (api):")
    out["deepseek"] = run("deepseek", deepseek, trials)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
