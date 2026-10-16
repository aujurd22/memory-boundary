"""H8b：合并管线在大候选池下的实测——用 flymemory 真库当候选池

H8 首跑的教训：4 条同族干扰的小池子里，灰区兜底让 zero-call 归零。
真实场景候选池是 1.6 万条。H8b 用真库构造：
  - 每对刺激配 K 条真实库条目（K=20/100），其中正确候选（对照 old）埋在池中
  - 结构层（双阈值）：max_cos < 0.40 → zero-call KEEP；否则交 LLM
  - 对比 full（全部 K 条都交 LLM，模拟他们无结构层的全量候选）
测：zero-call 率（省调用）+ 准确率保持 + 灰区候选数压缩比。
"""
import json, re, time, urllib.request, random
import numpy as np
from collections import Counter, defaultdict

KEY_SRC = r"D:/djr82/Documents/trae/6a8dbce9f0e28e9922d54185/hkt_login/audit_labels.py"
OUT = r"D:\djr82\memory_boundary\h8b_bigpool.json"
random.seed(7)

import pickle
with open(r"D:/djr82/flymemory/flymemory/flymemory_v3.pkl", "rb") as f:
    mem = pickle.load(f)
REAL = [m for m in mem["memories"] if m.get("text")][:8000]

STIM = json.load(open(r"D:\djr82\memory_boundary\stimuli_v1.json",
                      encoding="utf-8"))["stimuli"]
STIM_BY_FAM = defaultdict(list)
for s in STIM:
    STIM_BY_FAM[s["family"]].append(s)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 预编码真库子集（缓存）
print("encoding real pool ...", flush=True)
REAL_TEXTS = [m["text"][:120] for m in REAL[:2000]]
REAL_E = model.encode(REAL_TEXTS, normalize_embeddings=True,
                      show_progress_bar=False)


def build_prompt_multi(new_text, candidates):
    from h7_cross import FACT_RECONCILE_PROMPT
    cand_block = "\n".join(f"fact_{i}: {t}" for i, t in enumerate(candidates))
    p = FACT_RECONCILE_PROMPT.replace("{new}", new_text)
    p = p.replace("fact_a: {old}", cand_block)
    return p


def deepseek_call(prompt):
    import re as _re
    src = open(KEY_SRC, encoding="utf-8").read()
    key = _re.search(r'api_key="([^"]+)"', src).group(1)
    body = json.dumps({"model": "deepseek-flash",
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": 1200}).encode()
    req = urllib.request.Request("https://api.deepseek.com/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json",
                                                     "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return (d["choices"][0]["message"].get("content") or "").strip()


def parse_verdict(reply, candidates, old_text):
    if not reply.strip():
        return "EMPTY"
    if re.search(r'"ops"\s*:\s*\[\s*\]', reply):
        return "KEEP"
    for m in re.finditer(r'"supersedes"\s*:\s*\[(.*?)\]', reply, re.S):
        for idx_s in re.findall(r'fact_(\d+)', m.group(1)):
            if candidates[int(idx_s)] == old_text:
                return "SUPERSEDE"
    return "KEEP"


def expected(s):
    return {"attribute": "SUPERSEDE", "flip": "SUPERSEDE",
            "paraphrase": "SUPERSEDE", "unrelated": "KEEP"}[s["family"]]


def run(K, n_per_fam=10):
    stats = {"zero_call": 0, "llm_calls": 0, "correct": 0, "total": 0,
             "cands_llm": 0}
    misses = []
    rng = random.Random(7)
    sample = []
    for fam, lst in STIM_BY_FAM.items():
        sample += rng.sample(lst, n_per_fam)
    rng.shuffle(sample)
    for s in sample:
        old, new = s["old"], s["new"]
        # 候选池：正确候选 + K-1 条真实库干扰
        distractors = rng.sample(REAL_TEXTS, K - 1)
        candidates = [old] + distractors
        e_new = model.encode(new, normalize_embeddings=True)
        # 结构层：算 new 与全部候选的 cos
        cand_embs = [model.encode(t, normalize_embeddings=True) for t in candidates[:1]]
        # old 单独编码；真实干扰用预编码池（同文本对齐）
        cos_old = float(np.dot(e_new, cand_embs[0]))
        # 干扰的真实 cos（用预编码池里同样的文本找回来太麻烦，统一编码 K-1 条）
        d_embs = model.encode(distractors, normalize_embeddings=True,
                              show_progress_bar=False)
        cos_dis = (d_embs @ e_new)
        max_all = max(cos_old, float(cos_dis.max()))
        stats["total"] += 1
        if max_all < 0.40:
            verdict = "KEEP"
            stats["zero_call"] += 1
            ok = verdict == expected(s)
            stats["correct"] += ok
            if not ok:
                misses.append({"qid": s["qid"], "family": s["family"],
                               "mode": "zero-call", "exp": expected(s),
                               "got": verdict, "max_cos": round(max_all, 3)})
            continue
        # 灰区/同簇：只把 cos>=0.40 的候选交 LLM
        cands = [candidates[0]] + [t for t, c in zip(distractors, cos_dis)
                                   if c >= 0.40]
        stats["llm_calls"] += 1
        stats["cands_llm"] += len(cands)
        prompt = build_prompt_multi(new, cands)
        reply = ""
        for _ in range(4):
            try:
                reply = deepseek_call(prompt)
                if reply.strip():
                    break
            except Exception:
                pass
            time.sleep(2)
        verdict = parse_verdict(reply, cands, old)
        ok = verdict == expected(s)
        stats["correct"] += ok
        if not ok:
            misses.append({"qid": s["qid"], "family": s["family"],
                           "mode": "llm", "exp": expected(s),
                           "got": verdict, "n_cands": len(cands)})
    return stats, misses


def main():
    out = {"K_20": {}, "K_100": {}}
    for K in (20, 100):
        stats, misses = run(K)
        n = stats["total"]
        out[f"K_{K}"] = {
            "accuracy": round(stats["correct"] / n, 3),
            "zero_call_rate": round(stats["zero_call"] / n, 3),
            "llm_calls_per_item": round(stats["llm_calls"] / n, 2),
            "avg_cands_to_llm": round(stats["cands_llm"] / max(1, stats["llm_calls"]), 1),
            "misses": misses,
        }
        print(f"K={K}: acc={stats['correct']}/{n}={stats['correct']/n:.0%}, "
              f"zero-call {stats['zero_call']}/{n}={stats['zero_call']/n:.0%}, "
              f"交LLM均候选 {stats['cands_llm']/max(1,stats['llm_calls']):.1f} (原池 {K})")
        if misses[:3]:
            print("  misses:", misses[:3])
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
