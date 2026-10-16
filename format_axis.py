"""格式对齐实验：读出侧"框效应"（Same Answer New Box 的等价物）

Same Answer 发现：RLVR 的域外增益 82% 是答案格式效应——模型把答案挪进
提取器看的框里，能力没变，测量变了。

本实验把该机制搬到问答读出侧，检验它对【真实知识】的增益/损耗轨迹：
场景：模型知道一条事实（预训知识），但提问格式在三条轴变化：
  A. 提问格式: 直问 / 前缀补全式 / JSON 槽位式
  B. 答案侧约束: 自由回答 / 强制格式（"只回答 X 词"）
  C. 干扰: 有/无 竞争上下文
测：什么格式下"知识读出"成功率最高；修复增益到底是格式增益还是知识增益。

设计（真实世界事实，模型应已知）：
  20 条常识（中国首都/水的沸点/一年月数...变体），三格式。
判据：格式间成功率差 > 干扰修复效应 = 格式主导读出
"""
import json, re, time, urllib.request, random

random.seed(5)
OUT = r"D:\djr82\memory_boundary\format_axis.json"
KEY_SRC = r"D:/djr82/Documents/trae/6a8dbce9f0e28e9922d54185/hkt_login/audit_labels.py"

# 常识事实（模型应已掌握）——用不常见的表述防语境套话
FACTS = [
    ("法国的首都", "巴黎"),
    ("水的化学式", "H2O"),
    ("一年有多少个月", "12"),
    ("太阳从哪个方向升起", "东"),
    ("一周有几天", "7"),
    ("光速大约多少（万公里/秒）", "30"),
    ("地球有几个天然卫星", "1"),
    ("中国最长的河流", "长江"),
    ("人类有多少对染色体", "23"),
    ("冰的熔点（摄氏度）", "0"),
]


def make_items():
    items = []
    for q, a in FACTS:
        # 干扰上下文：三条貌似相关但不同主题的记录
        distract = [f"[记录] 关于{q}的旧讨论已归档（内容缺失）。",
                    f"[记录] 某系统管理员提到过{q}相关配置。",
                    f"[记录] 一份待办清单包含{q}的检查项。"]
        items.append({"q": q, "a": a, "distract": distract})
    return items


def prompts(item, fmt):
    q, a, dis = item["q"], item["a"], item["distract"]
    if fmt == "direct":
        return f"问题：{q}是什么？\n回答："
    if fmt == "prefix":
        return f"补全下面句子：{q}是"
    if fmt == "json":
        return (f"请以 JSON 输出：{{\"问题\": \"{q}\", \"答案\": \"?\"}}\n"
                "只输出 JSON。")
    if fmt == "direct_noise":
        return ("以下是一些上下文记录：\n" + "\n".join(dis) +
                f"\n\n问题：{q}是什么？\n回答：")
    if fmt == "json_noise":
        return ("以下是一些上下文记录：\n" + "\n".join(dis) +
                f"\n\n请以 JSON 输出：{{\"问题\": \"{q}\", \"答案\": \"?\"}}\n只输出 JSON。")


def qwen(p, mt=80):
    for _ in range(5):
        try:
            body = json.dumps({"messages": [{"role": "user", "content": p + " /no_think"}],
                               "temperature": 0, "max_tokens": mt},
                              ensure_ascii=False).encode()
            req = urllib.request.Request("http://127.0.0.1:8777/v1/chat/completions",
                                         data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            c = (d['choices'][0]['message'].get('content') or '').strip()
            if c:
                return c
        except Exception:
            pass
        time.sleep(2)
    return ""


def deepseek(p, mt=1200):
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
            with urllib.request.urlopen(req, timeout=150) as r:
                d = json.loads(r.read())
            c = (d['choices'][0]['message'].get('content') or '').strip()
            if c:
                return c
        except Exception:
            pass
        time.sleep(3)
    return ""


def extractor_free(reply, gold):
    """自由提取器：看整段回复。"""
    return gold in reply


def extractor_boxed(reply, gold):
    """格式提取器：只看"答案"框（JSON 值 / 冒号后首段）——Same Answer 的提取方式。"""
    m = re.search(r'"答案"\s*:\s*"([^"]+)"', reply)
    if m:
        return gold in m.group(1)
    m = re.search(r'答[:：]\s*\S*', reply)
    if m:
        return gold in m.group(0)
    return False


def main():
    items = make_items()
    out = {}
    fmts = ["direct", "prefix", "json", "direct_noise", "json_noise"]
    for model_name, fn in [("qwen", qwen), ("deepseek", deepseek)]:
        stats = {f: {"free": 0, "boxed": 0, "n": 0} for f in fmts}
        for item in items:
            for f in fmts:
                r = fn(prompts(item, f))
                stats[f]["n"] += 1
                stats[f]["free"] += extractor_free(r, item["a"])
                stats[f]["boxed"] += extractor_boxed(r, item["a"])
                if f in ("json", "json_noise") and not extractor_free(r, item["a"]):
                    print(f"  MISS {model_name}/{f}: {item['q']} -> {r[:60]!r}")
        out[model_name] = stats
        print(f"\n{model_name}:")
        for f in fmts:
            s = stats[f]
            print(f"  {f:14s}: free {s['free']}/{s['n']}, boxed {s['boxed']}/{s['n']}")
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
