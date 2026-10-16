"""H6：五臂端到端原型——在 flymemory 真库上的完整写入→成簇→层级→召回流程

H1-H5 各验了一个环节；H6 把它们串成一条管线，在真库谱系上端到端跑：
  对每个抽样的 supersede 谱系（2-5 成员，按时间排序）：
    模拟五臂的召回行为（用真实嵌入，成簇+层级按 H4b/H5 验证过的算法），
    查询 = 谱系中间版本的问题（造"现在"指向问句），裁判问题延后——
    本轮先机械判分（召回文本 vs 谱系最新版本）。

关键新量：
  - 端到端 correct 率 per 臂（A1 语义 / A4 sup+time / A5 层级 / A6 簇层级）
  - "中间版本陷阱"：谱系长 ≥3 时，问最新状态，A1 可能召回中间版本
    （stale 的真库版）；A4/A5/A6 是否修复
"""
import pickle, json, random, re
import numpy as np
from collections import defaultdict, Counter

random.seed(7)
with open(r"D:/djr82/flymemory/flymemory/flymemory_v3.pkl", "rb") as f:
    mem = pickle.load(f)
mems = mem["memories"]
by_id = {m["memory_id"]: m for m in mems}

emb_cache = {}
def get_emb(mid):
    if mid not in emb_cache:
        v = np.array(by_id[mid]["embedding"], dtype=np.float32)
        n = np.linalg.norm(v)
        emb_cache[mid] = v / n if n > 0 else v
    return emb_cache[mid]


def lineage_members(mid):
    seen = {mid}
    frontier = [mid]
    while frontier:
        cur = frontier.pop()
        m = by_id.get(cur)
        if not m:
            continue
        sb = m.get("superseded_by")
        if sb and sb != "None" and sb not in seen:
            seen.add(sb)
            frontier.append(sb)
    for other in mems:
        if other.get("superseded_by") == mid and other["memory_id"] not in seen:
            seen.add(other["memory_id"])
            frontier.append(other["memory_id"])
    return seen


# 组装谱系（按时间排序，长>=2）
lineages = {}
for m in mems:
    mid = m["memory_id"]
    if mid not in lineages:
        lin = lineage_members(mid)
        for x in lin:
            lineages[x] = lin
multi = []
for l in set(id(l) for l in lineages.values()):
    pass
seen_lins = set()
for l in lineages.values():
    key = tuple(sorted(l))
    if key in seen_lins:
        continue
    seen_lins.add(key)
    if 2 <= len(l) <= 5:
        multi.append(sorted(l, key=lambda x: by_id[x]["timestamp"]))
random.shuffle(multi)
print(f"候选谱系（2-5 成员）: {len(multi)}")
sample = multi[:300]


def make_query(text):
    """从真实条目文本造查询：取前 18 字符（真实库没有模板，用前缀截断模拟
    "关于X的问题"——这是最接近真实查询形态的机械代理）"""
    t = re.sub(r"\s+", "", text)
    return t[:18]


def cluster_of(entries, embs, threshold=0.55):
    """连通分量；entries=[(mid,t)]"""
    n = len(entries)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.dot(embs[entries[i][0]], embs[entries[j][0]])) >= threshold:
                parent[find(i)] = find(j)
    cl = defaultdict(list)
    for i in range(n):
        cl[find(i)].append(i)
    return list(cl.values())


def run_arm(arm, lin):
    """lin: 按时间排序的谱系成员 mid 列表。查询=第一个（最旧）版本的前缀问题，
    考察"现在"指向下能否召回最新版本。"""
    # 写入：谱系成员按时间 + 3 个异谱系干扰（从其它谱系抽）
    distractors = random.sample([m["memory_id"] for m in mems
                                 if m["memory_id"] not in set(lin)], 3)
    entries = [(mid, k) for k, mid in enumerate(lin)] + \
              [(d, 100 + k) for k, d in enumerate(distractors)]
    embs = {mid: get_emb(mid) for mid, _ in entries}
    query_text = make_query(by_id[lin[0]]["text"])
    qe = np.array(model_encode(query_text), dtype=np.float32)
    qe = qe / (np.linalg.norm(qe) or 1)

    if arm == "A1":  # 纯语义 top-1
        best = max(entries, key=lambda e: float(np.dot(qe, embs[e[0]])))[0]
    elif arm == "A4":  # sup+time：active 条目中语义 top-1
        active = [(mid, t) for mid, t in entries
                  if by_id[mid].get("superseded_by") in (None, "None")]
        best = max(active, key=lambda e: float(np.dot(qe, embs[e[0]])))[0]
    elif arm in ("A5", "A6"):  # 簇内层级（A5=阈值0.55，A6=0.6）
        th = 0.55 if arm == "A5" else 0.6
        clusters = cluster_of(entries, embs, th)
        scored = sorted(((float(np.dot(qe, embs[mid])), mid)
                         for mid, _ in entries), key=lambda x: -x[0])
        top_texts = [mid for sc, mid in scored[:3]]
        cluster_order = sorted(
            clusters,
            key=lambda idxs: max((float(np.dot(qe, embs[entries[i][0]]))
                                  for i in idxs if entries[i][0] in top_texts),
                                 default=-1), reverse=True)
        best = None
        for idxs in cluster_order:
            ctexts = {entries[i][0] for i in idxs}
            hits = [t for t in top_texts if t in ctexts]
            if hits:
                members = [(entries[i][1], entries[i][0]) for i in idxs
                           if entries[i][0] in hits]
                best = max(members)[1] if len(members) >= 2 else hits[0]
                break
        if best is None:
            best = scored[0][1]
    return best


_model = None
def model_encode(text):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _model.encode(text, normalize_embeddings=True)


def main():
    results = defaultdict(Counter)
    for lin in sample:
        newest = lin[-1]
        for arm in ("A1", "A4", "A5", "A6"):
            best = run_arm(arm, lin)
            ok = best == newest
            results[arm]["correct" if ok else "not_newest"] += 1
    out = {}
    for arm, c in results.items():
        rate = c["correct"] / sum(c.values())
        out[arm] = dict(c)
        print(f"{arm}: 召回最新版本 {c['correct']}/{sum(c.values())} = {rate:.0%}")
    json.dump(out, open(r"D:/djr82/memory_boundary/h6_e2e.json", "w"), indent=1)
    print("saved -> h6_e2e.json")


if __name__ == "__main__":
    main()
