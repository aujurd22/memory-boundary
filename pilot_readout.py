"""Pilot: 读出干扰定律的跨层验证——四项独立证据的第一次直接对照实验

今天四份独立证据指向同一机制：
  1. Readout Bottleneck (arXiv 2610.21176, today): LLM 工作记忆的失效在
     "干扰下的读出"，抑制竞争项身份 > 锐化注意力。
  2. 我们 H4-H6: 外部记忆库的失效同样是版本干扰下的读出，结构抑制（时间层级）
     把 stale 46%→0。
  3. TEM (arXiv 2610.21900, today): 给历史记录贴标签 +17-55pp。
  4. flymemory L6: 锚定使判断机械化 70.8→97.9%。

假设：记忆访问（内部/外部）由两个供给侧因素支配——竞争项抑制 + 参照锚定，
其效应超过模型规模。

本 pilot：in-context 嵌入更新场景，三臂 × 两模型：
  plain  = 记录流原样（新旧值都在，无结构）
  mask   = 旧记录遮蔽（竞争项抑制——Readout 式修复）
  label  = 旧记录贴【已过期】/【当前】标签（TEM 式修复）
模型：Qwen3-0.6B（本地）vs DeepSeek-V4.1-Flash（API）
预测：mask/label >> plain；0.6B+修复 ≥ 大模型+plain（素材供应 > 规模）
"""
import json, re, time, urllib.request, random

random.seed(7)
OUT = r"D:\djr82\memory_boundary\pilot_readout.json"
KEY_SRC = r"D:/djr82/Documents/trae/6a8dbce9f0e28e9922d54185/hkt_login/audit_labels.py"

COLORS = ["红", "蓝", "绿", "黄", "紫", "橙", "灰", "粉"]
NOISE = ["系统日志例行检查通过。", "备份任务已完成。", "网络延迟正常。",
         "例行巡检无异常。", "缓存已刷新。", "配置同步完成。"]


def make_trial(i):
    ent = f"实体{i + 1:02d}"
    old_c, new_c = random.sample(COLORS, 2)
    lines = [f"{ent}的颜色是{old_c}。"]
    for _ in range(3):
        lines.append(random.choice(NOISE))
    lines.append(f"{ent}的颜色是{new_c}。")
    return ent, old_c, new_c, lines


def build_prompt(ent, lines, arm, think_tag=False):
    ls = list(lines)
    if arm == "mask":
        ls[0] = "（早期记录已失效，内容不可读）"
    elif arm == "label":
        ls[0] = "【已过期】" + ls[0]
        ls[-1] = "【当前】" + ls[-1]
    stream = "\n".join(ls)
    p = (f"以下是一个实体的记录流（按时间顺序）：\n{stream}\n\n"
         f"问题：现在{ent}的颜色是什么？\n只回答颜色词。")
    if think_tag:
        p += " /no_think"
    return p


def qwen(prompt, mt=40):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": mt},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8777/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    c = d["choices"][0]["message"].get("content", "")
    if not c and d["choices"][0]["message"].get("reasoning_content"):
        c = d["choices"][0]["message"]["reasoning_content"]
    return c.strip()


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


def score(reply, new_c, old_c):
    """correct = 出现 new_c 且不把 old_c 当答案。宽容解析。"""
    if not reply:
        return "empty"
    # 找回答里的颜色词
    hit_new = new_c in reply
    hit_old = old_c in reply
    if hit_new and not hit_old:
        return "correct"
    if hit_new and hit_old:
        # 两者都出现：看哪个位置靠后/更接近"答："
        m = re.search(r'(?:答[:：]|答案是)?\s*([红蓝绿黄紫橙灰粉])', reply)
        if m and m.group(1) == new_c:
            return "correct"
        return "ambiguous"
    if hit_old:
        return "stale"
    return "other"


def run_model(name, fn, trials, think_tag=False, retries=1):
    stats = {}
    for arm in ("plain", "mask", "label"):
        c = {"correct": 0, "stale": 0, "ambiguous": 0, "empty": 0, "other": 0}
        for ent, old_c, new_c, lines in trials:
            prompt = build_prompt(ent, lines, arm, think_tag)
            reply = ""
            for _ in range(retries + 1):
                try:
                    reply = fn(prompt)
                    if reply:
                        break
                except Exception:
                    time.sleep(2)
            c[score(reply, new_c, old_c)] += 1
        stats[arm] = c
        n = sum(c.values())
        print(f"  {name:8s} {arm:5s}: correct {c['correct']}/{n} "
              f"= {c['correct']/n:.0%}  {c}")
    return stats


def main():
    trials = [make_trial(i) for i in range(20)]
    out = {}
    print("qwen3-0.6b (local):")
    out["qwen"] = run_model("qwen", qwen, trials, think_tag=True)
    print("deepseek-flash (api):")
    out["deepseek"] = run_model("deepseek", deepseek, trials)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
