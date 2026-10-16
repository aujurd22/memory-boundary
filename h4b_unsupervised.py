"""H4b 预验：无监督实体成簇——不用名字前缀，靠嵌入聚类自己发现"同实体版本"。

问题（11 月 LLM 侧的核心）：真实记忆库没有"实体\d+"这种命名规则。
A5 的显式规则给了结构，H4b 问：结构能否从条目文本本身无监督地恢复？

方法（三步，全部只用嵌入）：
  1. 条目间余弦 ≥ 0.55 连边（同主题）
  2. 连通分量 = 候选簇（同一实体的 old/new 自然强连，跨实体经填充词
     不相连——填充词语义距离远）
  3. 簇内按写入时间排序取最新 = A5 层级

关键检验：
  - 簇纯度：每簇是否恰好含 {old, new} 同对（不把别的实体的条目吸进来）
  - 修复率：A5(无监督簇) 的 attribute stale 率是否同样归零
风险（预注册里写过的）：unrelated 族 old/new 语义远，不应被连进同簇——
  若连上，时间层级会把独立事实误合并（这就是 KIKO 发㟞e 的撤回问题）。
"""
import json, re
import numpy as np
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer

STIM = json.load(open(r"D:\djr82\memory_boundary\stimuli_v1.json",
                      encoding="utf-8"))["stimuli"]
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
OUT = r"D:\djr82\memory_boundary\h4b_result.json"


def make_q(text):
    m = re.match(r'(.+?(?:的|偏好|喜欢|当前|已).+?)是.*', text)
    return (m.group(1) + "是什么") if m else text + "的状态是什么"


def build_clusters(entries, emb_map, link_threshold=0.55):
    """连通分量聚类：条目两两 cos>=thresh 连边。entries=[(text, t), ...]"""
    texts = [t for t, _ in entries]
    n = len(texts)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if float(np.dot(emb_map[texts[i]], emb_map[texts[j]])) >= link_threshold:
                union(i, j)
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    return list(clusters.values())


def run_h4b(stimuli, link_threshold=0.55):
    rows = []
    purity_stats = Counter()
    for s in stimuli:
        e_old = model.encode(s["old"], normalize_embeddings=True)
        e_new = model.encode(s["new"], normalize_embeddings=True)
        fillers = [f"填充记录{k}号关于无关主题{k}{c}"
                   for k, c in enumerate(["甲", "乙", "丙", "丁", "戊"])]
        E = {f: model.encode(f, normalize_embeddings=True) for f in fillers}
        E[s["old"]] = e_old
        E[s["new"]] = e_new
        entries = [(s["old"], 0), (s["new"], 6)] + \
                  [(f, k + 1) for k, f in enumerate(fillers)]
        clusters = build_clusters(entries, E, link_threshold)

        # 簇纯度：old/new 应同簇，且不混入填充词
        old_ci = new_ci = None
        for ci, idxs in enumerate(clusters):
            if 0 in idxs:
                old_ci = ci
            if 1 in idxs:
                new_ci = ci
        same_cluster = old_ci == new_ci and old_ci is not None
        mixed = same_cluster and len(clusters[old_ci]) > 2  # 吸进填充词
        purity_stats["old_new_same" if same_cluster else "separated"] += 1
        if mixed:
            purity_stats["mixed_with_filler"] += 1

        # 查询判定
        for ask_new in (True, False):
            src = s["new"] if ask_new else s["old"]
            qe = model.encode(make_q(src), normalize_embeddings=True)
            # 无监督簇层级：找查询命中的簇，簇内取时间最新
            best = None
            scored = sorted(((float(np.dot(qe, E[t])), t) for t, _ in entries),
                            key=lambda x: -x[0])
            top_texts = [t for sc, t in scored[:3]]
            for ci, idxs in enumerate(clusters):
                ctexts = {entries[i][0] for i in idxs}
                hits = [t for t in top_texts if t in ctexts]
                # 只对含"非填充"成员的簇启用层级（簇=实体版本组）
                if hits and any(not t.startswith("填充") for t in ctexts):
                    members = [(entries[i][1], entries[i][0]) for i in idxs
                               if entries[i][0] in hits]
                    best = max(members)[1]
                    break
            if best is None:
                best = scored[0][1]
            if ask_new:
                outcome = ("correct" if best == s["new"]
                           else "stale" if best == s["old"] else "wrong")
            else:
                outcome = ("correct_old" if best == s["old"]
                           else "correct" if best == s["new"] else "wrong")
            rows.append({"family": s["family"],
                         "dir": "现在" if ask_new else "当时",
                         "outcome": outcome})
    return rows, purity_stats


def main():
    import random
    rng = random.Random(7)
    stimuli = list(STIM)
    rng.shuffle(stimuli)
    for th in (0.55, 0.60, 0.65):
        rows, purity = run_h4b(stimuli, th)
        att_stale = sum(1 for r in rows if r["family"] == "attribute"
                        and r["dir"] == "现在" and r["outcome"] == "stale")
        n_att = sum(1 for r in rows if r["family"] == "attribute"
                    and r["dir"] == "现在")
        bad_total = sum(1 for r in rows
                        if r["outcome"] in ("stale", "wrong"))
        print(f"threshold={th}: attribute stale {att_stale}/{n_att}, "
              f"总失效 {bad_total}/{len(rows)}, 簇统计 {dict(purity)}")
    json.dump({"note": "H4b unsupervised clustering pretest"},
              open(OUT, "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
