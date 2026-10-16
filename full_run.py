"""记忆更新失效边界：五臂全量机械跑批 v1.0（200 对 × 时间轴双指向）

PREREG v0.3 三轴：语义Δ（实测 cos 分箱）× 状态类型 × 查询时间指向。
机械层先跑满全部 200 对——LLM 裁判只用于最终召回文本的判定，而
mini-registry 的召回行为（哪条被召回、是否被时间筛选过滤）是确定性的，
可以在 LLM 介入前完整量化。

产出：
  - 每臂×每族×每指向的判定分布
  - H1 信号：supersede 0.85 门在语义Δ 高区间（paraphrase）误触发率
  - H2 信号：时间筛选在【当时】指向下的 lost_update 率（vs A1-A3 基线）
"""
import json, random, re
import numpy as np
from collections import defaultdict, Counter

rng = random.Random(7)
STIM = json.load(open(r"D:\djr82\memory_boundary\stimuli_v1.json",
                      encoding="utf-8"))["stimuli"]
OUT = r"D:\djr82\memory_boundary\full_run_v1.json"

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def make_q(text):
    """模板问句（不含值）——从陈述句抽实体+属性。"""
    m = re.match(r'(.+?(?:的|偏好|喜欢|当前|已).+?)是.*', text)
    if m:
        return m.group(1) + "是什么"
    return text + "的状态是什么"


def delta_bin(cos_val):
    if cos_val >= 0.9: return "0.9+"
    if cos_val >= 0.7: return "0.7-0.9"
    if cos_val >= 0.5: return "0.5-0.7"
    return "<0.5"


class MiniRegistry:
    def __init__(self, mode_revise, mode_recall, sup_gate=0.85, threshold=0.45):
        self.mode_revise = mode_revise
        self.mode_recall = mode_recall
        self.sup_gate = sup_gate
        self.threshold = threshold
        self.entries = []

    def write(self, text, emb, t):
        if self.mode_revise == "supersede":
            best, bs = None, 0.0
            for e in self.entries:
                if e["active"]:
                    s = float(np.dot(emb, e["emb"]))
                    if s > bs:
                        bs, best = s, e
            if best is not None and bs > self.sup_gate:
                best["active"] = False
        self.entries.append({"text": text, "emb": emb, "t": t, "active": True})

    def recall(self, q_emb):
        cands = [(float(np.dot(q_emb, e["emb"])), e) for e in self.entries
                 if (e["active"] if self.mode_recall == "semantic+time" else True)]
        if not cands:
            return None, 0.0
        cands.sort(key=lambda x: -x[0])
        s, e = cands[0]
        return (e, s) if s >= self.threshold else (None, s)


def judge(pair, asked_about, rec, sup_pair_formed):
    """按 PREREG v0.3 三轴判分。
    asked_about=False →【当时】指向（问旧状态）; True →【现在】指向（问新状态）。
    """
    if rec is None:
        # 主题在记忆里存在（old 一定写过）→ 指向决定的失效
        return "lost_update"
    text = rec["text"]
    target_new, target_old = pair["new"], pair["old"]
    if asked_about:  # 【现在】
        if text == target_new:
            return "correct"
        if text == target_old:
            return "stale"
        return "wrong"   # 召回了第三条（填充词碰撞，记录之）
    else:            # 【当时】
        if text == target_old:
            return "correct_old"
        if text == target_new:
            return "correct"   # 同义改述场景 new 即同一内容；否则也记录
        return "wrong"


def run_arm(arm, stimuli):
    mode_revise = "supersede" if "sup" in arm else "append"
    mode_recall = "semantic+time" if "time" in arm else "semantic"
    results = defaultdict(Counter)
    sup_fired = Counter()   # H1: supersede 门在哪些 Δ 箱触发
    for pair in stimuli:
        reg = MiniRegistry(mode_revise, mode_recall)
        e_old = model.encode(pair["old"], normalize_embeddings=True)
        e_new = model.encode(pair["new"], normalize_embeddings=True)
        n_before = sum(1 for e in reg.entries if not e["active"])
        reg.write(pair["old"], e_old, 0)
        # supersede 门是否在 old→new 写入时触发（仅 A3/A4 有意义）
        if mode_revise == "supersede":
            # 先写 old，再直接探测 new 写入是否触发取代
            s = float(np.dot(e_new, e_old))
            sup_fired[delta_bin(pair["cos"])] += int(s > reg.sup_gate)
        for k in range(5):
            filler = f"填充记录{k}号关于无关主题{k}{rng.choice(['甲','乙','丙'])}"
            reg.write(filler, model.encode(filler, normalize_embeddings=True), k + 1)
        reg.write(pair["new"], e_new, 6)
        bin_key = delta_bin(pair["cos"])
        for ask_new in (True, False):
            src_text = pair["new"] if ask_new else pair["old"]
            q = make_q(src_text)
            q_emb = model.encode(q, normalize_embeddings=True)
            rec, s = reg.recall(q_emb)
            outcome = judge(pair, ask_new, rec,
                            sup_pair_formed=mode_revise == "supersede")
            results[(bin_key, pair["state_type"], "现在" if ask_new else "当时")][outcome] += 1
    return results, sup_fired


def main():
    all_out = {}
    for arm in ["A1_append_sem", "A2_append_time", "A3_sup_sem", "A4_sup_time"]:
        r, sup_fired = run_arm(arm, STIM)
        flat = {f"{b}|{st}|{tp}": dict(c) for (b, st, tp), c in sorted(r.items())}
        all_out[arm] = flat
        tot = Counter()
        for c in r.values():
            tot.update(c)
        print(f"{arm}: {dict(tot)}")
        if sup_fired:
            print(f"  supersede 门触发(按Δ箱): {dict(sup_fired)}")
    json.dump(all_out, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nsaved -> {OUT}")

    # 关键对比（预注册 H2 的机械信号）
    print("\n=== H2 信号：【当时】指向 lost_update（过度遗忘） ===")
    for arm in all_out:
        lost = sum(v.get("lost_update", 0) for k, v in all_out[arm].items()
                   if k.endswith("|当时"))
        n = sum(sum(v.values()) for k, v in all_out[arm].items() if k.endswith("|当时"))
        print(f"  {arm}: {lost}/{n}")


if __name__ == "__main__":
    main()
