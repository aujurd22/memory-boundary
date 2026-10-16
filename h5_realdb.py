"""H5：flymemory 真库上的无监督成簇质量（supersede 谱系作 ground truth）

机械层（模板句）结论：cos≥0.55 连通分量可恢复实体版本簇（1.8% 残留）。
H5 问：真实记忆库（16507 条真实中文工作记忆，非模板句）上，
同一阈值的连通分量能否恢复 supersede 谱系（4515 个多成员组）？

Ground truth：superseded_by 字段定义的谱系（构造性真值，非人工标注）。
判分：
  - 谱系恢复率：同谱系条目是否被分进同一簇
  - 簇纯度：簇内是否混入异谱系条目（跨主题粘连=最危险的失效）
  - 阈值扫描：0.5 / 0.55 / 0.6 / 0.65
"""
import pickle, json, random
import numpy as np
from collections import defaultdict, Counter

random.seed(7)
with open(r"D:/djr82/flymemory/flymemory/flymemory_v3.pkl", "rb") as f:
    mem = pickle.load(f)
mems = mem["memories"]
by_id = {m["memory_id"]: m for m in mems}


def lineage(mid):
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
            if other.get("superseded_by") == cur and other["memory_id"] not in seen:
                seen.add(other["memory_id"])
                frontier.append(other["memory_id"])
    return seen


lineages = {}
for m in mems:
    mid = m["memory_id"]
    if mid not in lineages:
        lin = lineage(mid)
        for x in lin:
            lineages[x] = lin
multi = [l for l in lineages.values() if len(l) > 1]
# 抽样 200 个谱系（算力约束）
sample = random.sample(multi, min(200, len(multi)))
print(f"抽样 {len(sample)} 个谱系, "
      f"共 {sum(len(l) for l in sample)} 条目")

emb_cache = {}
def get_emb(mid):
    if mid not in emb_cache:
        emb_cache[mid] = np.array(by_id[mid]["embedding"], dtype=np.float32)
    return emb_cache[mid]

# 归一化
for mid in list(emb_cache):
    n = np.linalg.norm(emb_cache[mid])
    if n > 0:
        emb_cache[mid] = emb_cache[mid] / n


def evaluate(threshold):
    recover = 0          # 谱系内所有对都连通（谱系=单簇）
    partial = 0          # 谱系分裂成多簇
    impure = 0           # 簇内混入异谱系条目
    for lin in sample:
        mids = sorted(lin)
        # 连通分量（只在本谱系条目+全域近邻？为算力计：谱系内连边+与全域其它条目的最近邻检查）
        n = len(mids)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(n):
            for j in range(i + 1, n):
                if float(np.dot(get_emb(mids[i]), get_emb(mids[j]))) >= threshold:
                    parent[find(i)] = find(j)
        n_clusters = len({find(i) for i in range(n)})
        if n_clusters == 1:
            recover += 1
        else:
            partial += 1
        # 纯度：每个谱系成员在全域中的最近邻（top-3）是否有异谱系条目混入簇语义
        # 简化：谱系条目间的最近邻若低于阈值且最近邻是异谱系——不算失效（本来就该分离）
        # 真正的纯度失效=谱系内两成员 cos >= threshold 却被判不同簇（不可能，连通）→
        # 改测：异谱系条目对 cos>=threshold 的比例（过度连接风险）
    # 过度连接风险：随机抽 500 对异谱系条目，测 cos>=threshold 比例
    all_active = random.sample([m["memory_id"] for m in mems], 400)
    cross = 0
    pairs = 0
    for i in range(len(all_active)):
        for j in range(i + 1, len(all_active)):
            a, b = all_active[i], all_active[j]
            if lineages.get(a) is lineages.get(b) and len(lineages.get(a, ())) > 1:
                continue
            pairs += 1
            if pairs > 800:
                break
            if float(np.dot(get_emb(a), get_emb(b))) >= threshold:
                cross += 1
    return recover, partial, cross, min(pairs, 800)


def main():
    out = {}
    for th in (0.5, 0.55, 0.6, 0.65):
        rec, part, cross, npair = evaluate(th)
        n = rec + part
        out[th] = {"recover": rec, "partial": part,
                   "recover_rate": round(rec / max(1, n), 3),
                   "cross_link_rate": round(cross / max(1, npair), 3)}
        print(f"th={th}: 谱系完整恢复 {rec}/{n} = {rec/max(1,n):.0%}, "
              f"异谱系误连 {cross}/{npair} = {cross/max(1,npair):.1%}")
    json.dump(out, open(r"D:\djr82\memory_boundary\h5_realdb.json", "w",
                        encoding="utf-8"), indent=1)
    print("saved -> h5_realdb.json")


if __name__ == "__main__":
    main()
