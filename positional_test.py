"""位置操作隔离测试：纯列表上的基础操作能力（无干扰、无上下文）

背景：decompose_test 发现 0.6B 能完整抄出一行 8 色序列（90%），
但"取最后一个"仅 10-30%，而 regex 程序取末位 80%。
本测试剥离一切：单行纯列表输入，测五个原子操作：
  1. last   —— 最后一个
  2. first  —— 第一个
  3. nth    —— 第 k 个（k 随机）
  4. count  —— 有多少个
  5. copy   —— 完整复述
对两模型。这给"基础读出失效"一张精确的能力剖面。
"""
import json, re, time, urllib.request, random

random.seed(89)
OUT = r"D:\djr82\memory_boundary\positional_test.json"
KEY_SRC = r"D:/djr82/Documents/trae/6a8dbce9f0e28e9922d54185/hkt_login/audit_labels.py"
COLORS = list("红蓝绿黄紫橙灰粉青棕白黑金银铜米朱靛黛绯墨翠绛褐")
N_TRIALS = 10


def make_list():
    n = 8
    return random.sample(COLORS, n)


def prompts(lst, op):
    js = "，".join(lst)
    if op == "last":
        return f"列表：{js}\n列表的最后一个元素是什么？只回答一个色词。", lst[-1]
    if op == "first":
        return f"列表：{js}\n列表的第一个元素是什么？只回答一个色词。", lst[0]
    if op == "nth":
        k = random.randint(3, 7)
        return f"列表：{js}\n列表的第{k}个元素是什么？只回答一个色词。", lst[k-1]
    if op == "count":
        return f"列表：{js}\n列表里一共有几个元素？只回答数字。", str(len(lst))
    if op == "copy":
        return f"列表：{js}\n把列表完整复述一遍，逗号分隔，不要其它内容。", None
    raise ValueError(op)


def qwen(p, mt=80):
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


def judge(reply, gold, op, lst):
    if not reply:
        return False
    if op == "copy":
        hits = [c for c in lst if c in reply]
        return len(hits) >= 7   # 至少 7/8 个且大体顺序对
    if op == "count":
        m = re.search(r'\d+', reply)
        return bool(m) and m.group(0) == gold
    # 色词位置判定：正确色出现且非首位出现的是别的色
    hits = [c for c in COLORS if c in reply]
    if not hits:
        return False
    return gold in hits and min(hits, key=lambda c: reply.find(c)) == gold


def main():
    ops = ["copy", "count", "first", "nth", "last"]
    out = {}
    for model_name, fn in [("qwen", qwen), ("deepseek", deepseek)]:
        stats = {}
        for op in ops:
            ok = 0
            misses = []
            for _ in range(N_TRIALS):
                lst = make_list()
                p, gold = prompts(lst, op)
                r = fn(p)
                good = judge(r, gold, op, lst)
                ok += good
                if not good:
                    misses.append({"list": "".join(lst), "gold": gold if op != 'copy' else '7/8+',
                                   "got": r[:50]})
            stats[op] = {"ok": ok, "n": N_TRIALS, "misses": misses[:3]}
            print(f"  [{model_name:8s}|{op:5s}] {ok}/{N_TRIALS} = {ok/N_TRIALS:.0%}", flush=True)
            for m in misses[:2]:
                print(f"      MISS list={m['list']} gold={m['gold']} got={m['got']!r}")
        out[model_name] = stats
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved ->", OUT)
    print("\n=== 原子操作剖面 ===")
    print(f"  {'op':6s} | qwen-0.6B | deepseek")
    for op in ops:
        q = out["qwen"][op]; d = out["deepseek"][op]
        print(f"  {op:6s} | {q['ok']:2d}/{q['n']}     | {d['ok']:2d}/{d['n']}")


if __name__ == "__main__":
    main()
