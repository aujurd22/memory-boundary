"""H4 预验：实体簇内时间层级 vs 纯语义竞争（Cammarata 主张的工程翻译）

A1 基线：纯语义 top-1（已知 attribute 族 stale 46%）
A5 新臂：语义召回 + 实体簇内时间层级——
  召回 top-k（k=3），若多条目同实体（名字前缀相同），取写入时间最新者。
预测（PREREG v0.4）：A5 的 attribute stale <10%，unrelated 不受影响。
"""
import json, re
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

rng_seed = 7
STIM = json.load(open(r"D:\djr82\memory_boundary\stimuli_v1.json", encoding="utf-8"))["stimuli"]
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def make_q(text):
    m = re.match(r'(.+?(?:的|偏好|喜欢|当前|已).+?)是.*', text)
    return (m.group(1) + "是什么") if m else text + "的状态是什么"


def entity_of(text):
    m = re.match(r'(实体\d+)', text)
    return m.group(1) if m else text[:6]


def run(arm, stimuli):
    rows = []
    for s in stimuli:
        e_old = model.encode(s["old"], normalize_embeddings=True)
        e_new = model.encode(s["new"], normalize_embeddings=True)
        fillers = [f"填充记录{k}号关于无关主题{k}{c}"
                   for k, c in enumerate(["甲", "乙", "丙", "丁", "戊"])]
        F = [model.encode(f, normalize_embeddings=True) for f in fillers]
        entries = [(s["old"], e_old, 0), (s["new"], e_new, 6)] + \
                  [(f, fe, k + 1) for k, (f, fe) in enumerate(zip(fillers, F))]
        for ask_new in (True, False):
            src = s["new"] if ask_new else s["old"]
            q = make_q(src)
            qe = model.encode(q, normalize_embeddings=True)
            scored = sorted(((float(np.dot(qe, e)), t, txt)
                             for txt, e, t in entries), key=lambda x: -x[0])
            if arm == "A1":
                best = scored[0][2]
            else:  # A5: top-k 内同实体取最新
                topk = scored[:3]
                ents = defaultdict(list)
                for sc, t, txt in topk:
                    ents[entity_of(txt)].append((t, sc, txt))
                # 实体条目组：取该实体最新；若无实体组（全填充）取 top-1
                ent_groups = {k: max(v) for k, v in ents.items()
                              if k.startswith("实体")}
                if ent_groups:
                    # 同实体多版本 → 时间层级生效；否则退回纯语义
                    k_best, (t, sc, txt) = max(ent_groups.items(),
                                               key=lambda kv: kv[1][1])
                    versions = ents[k_best]
                    best = max(versions)[2] if len(versions) > 1 else txt
                else:
                    best = scored[0][2]
            # 判定
            if ask_new:
                outcome = ("correct" if best == s["new"]
                           else "stale" if best == s["old"] else "wrong")
            else:
                outcome = ("correct_old" if best == s["old"]
                           else "correct" if best == s["new"] else "wrong")
            rows.append({"family": s["family"], "cos": s["cos"],
                         "dir": "现在" if ask_new else "当时", "outcome": outcome})
    return rows


def summarize(rows, tag):
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        key = (r["family"], r["dir"])
        agg[key][1] += 1
        if r["outcome"] in ("stale", "wrong"):
            agg[key][0] += 1
    print(f"-- {tag}")
    for (fam, d), (bad, n) in sorted(agg.items()):
        print(f"   {fam:11s}|{d}: 失效 {bad}/{n} = {bad/n:.0%}")


def main():
    import random
    rng = random.Random(rng_seed)
    stimuli = list(STIM)
    rng.shuffle(stimuli)
    r1 = run("A1", stimuli)
    r5 = run("A5", stimuli)
    summarize(r1, "A1 纯语义（基线）")
    summarize(r5, "A5 簇内时间层级")
    # H4 判据
    a1_att = sum(1 for r in r1 if r["family"] == "attribute"
                 and r["dir"] == "现在" and r["outcome"] == "stale")
    a5_att = sum(1 for r in r5 if r["family"] == "attribute"
                 and r["dir"] == "现在" and r["outcome"] == "stale")
    n_att = sum(1 for r in r1 if r["family"] == "attribute" and r["dir"] == "现在")
    a5_unr = sum(1 for r in r5 if r["family"] == "unrelated"
                 and r["outcome"] in ("stale", "wrong"))
    n_unr = sum(1 for r in r5 if r["family"] == "unrelated")
    print(f"\nH4 判据: attribute stale {a1_att}/{n_att} -> {a5_att}/{n_att} "
          f"({'✓ <10%' if a5_att / n_att < 0.10 else '✗ 未达标'})")
    print(f"unrelated 副作用: {a5_unr}/{n_unr} ({'✓ 无新增失效' if a5_unr <= 2 else '✗ 引入失效'})")
    json.dump({"A1": r1, "A5": r5},
              open(r"D:\djr82\memory_boundary\h4_test.json", "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved -> h4_test.json")


if __name__ == "__main__":
    main()
