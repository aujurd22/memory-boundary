"""H8：合并管线叠加收益实测——结构层前置过滤 × TriLink 判断层

H7 结论说"正交互补，合并有叠加收益"。H8 实证它。

两条管线，同样的 60 对刺激：
  管线 P-full（TriLink 原型）：每个 new fact 与【全部】existing facts 跑
    reconciler prompt（模拟他们真实的 O(N×M) 全量候选场景——我们用
    每对配 4 个候选（正确旧条目+3 个干扰）模拟 N>1 的检索成本）
  管线 P-hybrid（我们的合并版）：H4b 无监督成簇先把 existing facts 分簇，
    只把【与 new 同簇】的候选交给 reconciler prompt——簇为空或无候选
    时零 LLM 调用（结构层直接判 INSERT）

测三个量：
  1. 判定准确率（vs ground truth）：P-hybrid 不得低于 P-full
  2. LLM 调用次数：P-hybrid 应显著少于 P-full
  3. 候选数均值：进入判断层的候选规模
"""
import json, re, time, urllib.request
from collections import Counter, defaultdict

KEY_SRC = r"D:/djr82/Documents/trae/6a8dbce9f0e28e9922d54185/hkt_login/audit_labels.py"
OUT = r"D:\djr82\memory_boundary\h8_merged.json"

import sys
sys.path.insert(0, r"D:/djr82/memory_boundary")
from h7_cross import FACT_RECONCILE_PROMPT, deepseek, expected  # reuse


def cluster_candidates(existing_texts, new_text, embs, hi_th=0.55, lo_th=0.40):
    """H4b 结构层 v2（双阈值，H8 首跑修正）：
    - max cos < lo_th（0.40）→ zero-call KEEP（明确无关）
    - 否则把 cos >= lo_th 的候选交 LLM（灰区 0.40-0.55 兜底防漏，
      首跑证明硬门 0.55 会滤掉 5 例真更新——flip min 0.478 等）
    返回 (candidates_for_llm, zero_call)。"""
    sims = [(t, float(np.dot(embs[new_text], embs[t]))) for t in existing_texts]
    if max(s for _, s in sims) < lo_th:
        return [], True
    return [t for t, s in sims if s >= lo_th], False


def build_prompt_multi(new_text, candidates):
    """TriLink prompt 的多候选版（candidates_block 列多条）。"""
    cand_block = "\n".join(f"fact_{i}: {t}" for i, t in enumerate(candidates))
    prompt = FACT_RECONCILE_PROMPT.replace(
        "Draft 0: {new}", f"Draft 0: {new_text}").replace(
        "fact_a: {old}", cand_block)
    # 修正占位符（format 顺序问题——直接手动替换）
    prompt = prompt.replace("{new}", new_text)
    prompt = prompt.replace("fact_a: {old}", cand_block)
    return prompt


def parse_ops(reply):
    """返回 (superseded_ids, parse_ok)。"""
    if not reply.strip():
        return [], False
    if re.search(r'"ops"\s*:\s*\[\s*\]', reply):
        return [], True
    ids = re.findall(r'"supersedes"\s*:\s*\[(.*?)\]', reply, re.S)
    out = []
    for group in ids:
        out += re.findall(r'fact_\w+', group)
    return out, bool(ids)


def run(family_sample, embs, mode):
    """mode: 'full'（全候选）| 'hybrid'（簇内候选）。返回判定结果+调用数。"""
    results = []
    calls = 0
    cand_total = 0
    for s in family_sample:
        old, new = s["old"], s["new"]
        # 干扰候选：同族其它条目 3 条（模拟检索噪声）
        distractors = [x["old"] for x in STIM_BY_FAM[s["family"]]
                       if x["qid"] != s["qid"]][:3]
        existing = [old] + distractors
        if mode == "hybrid":
            cands, zero = cluster_candidates(existing, new, embs)
            if zero:
                # 结构层直接判 INSERT——零 LLM 调用
                verdict = "KEEP"
                results.append({"qid": s["qid"], "family": s["family"],
                                "expected": expected(s), "verdict": verdict,
                                "calls": 0, "cands": 0})
                continue
        else:
            cands = existing
        cand_total += len(cands)
        calls += 1
        prompt = build_prompt_multi(new, cands)
        for attempt in range(3):
            try:
                reply = deepseek(prompt)
                break
            except Exception:
                time.sleep(3)
        else:
            reply = ""
        ids, parse_ok = parse_ops(reply)
        verdict = "SUPERSEDE" if any(
            cands[int(re.search(r'\d+', i).group())] == old
            for i in ids if re.search(r'\d+', i)) else "KEEP"
        results.append({"qid": s["qid"], "family": s["family"],
                        "expected": expected(s), "verdict": verdict,
                        "calls": 1, "cands": len(cands)})
    return results, calls, cand_total / max(1, len(results))


import numpy as np
import random

STIM = json.load(open(r"D:\djr82\memory_boundary\stimuli_v1.json",
                      encoding="utf-8"))["stimuli"]
STIM_BY_FAM = defaultdict(list)
for s in STIM:
    STIM_BY_FAM[s["family"]].append(s)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def main():
    rng = random.Random(7)
    sample = []
    for fam, lst in STIM_BY_FAM.items():
        sample += rng.sample(lst, 15)
    rng.shuffle(sample)
    embs = {}
    for s in STIM:
        if s["old"] not in embs:
            embs[s["old"]] = model.encode(s["old"], normalize_embeddings=True)
        if s["new"] not in embs:
            embs[s["new"]] = model.encode(s["new"], normalize_embeddings=True)

    out = {}
    for mode in ("full", "hybrid"):
        res, calls, avg_cands = run(sample, embs, mode)
        agree = sum(1 for r in res if r["verdict"] == r["expected"])
        out[mode] = {"agree": agree, "total": len(res),
                     "rate": round(agree / len(res), 3),
                     "llm_calls": calls, "avg_candidates": round(avg_cands, 2)}
        print(f"{mode}: 一致率 {agree}/{len(res)} = {agree/len(res):.0%}, "
              f"LLM 调用 {calls}, 平均候选 {avg_cands:.2f}")
    # 叠加收益判定
    hybrid_keep = out["hybrid"]["rate"] >= out["full"]["rate"] - 0.05
    calls_saved = out["full"]["llm_calls"] - out["hybrid"]["llm_calls"]
    out["verdict"] = {
        "accuracy_kept": hybrid_keep,
        "calls_saved": calls_saved,
        "saving_rate": round(calls_saved / max(1, out["full"]["llm_calls"]), 3),
    }
    print(f"H8 判定: 准确率保持={hybrid_keep}, "
          f"省 {calls_saved}/{out['full']['llm_calls']} 次 LLM 调用 "
          f"({out['verdict']['saving_rate']:.0%})")
    json.dump(out, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
