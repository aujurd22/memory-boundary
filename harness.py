"""记忆更新失效边界：五臂 mini-registry harness v0.1

五臂（2×2 机制因子 + 基线）在 200 对刺激上的行为模拟：
  A1: append-only 修订 + 纯语义召回
  A2: append-only 修订 + 语义+时间筛选召回
  A3: 显式 supersede + 纯语义召回
  A4: 显式 supersede + 语义+时间筛选召回
  A0: 无记忆（对照，纯参数侧，本 harness 用"只看 new 不看 old"模拟）

召回模拟核心：查询 q（对 old 或 new 的问句）时，
  语义召回 = top-1 by cos(q, entry)，阈值 0.55
  时间筛选 = 语义 top-k 中保留 latest 状态（A2/A4 将 superseded 条目过滤）
本 harness 的意义：在接入真 LLM 记忆系统前，先机械验证五臂设计
能否在刺激集上产生预期方向的分离（H1/H2 的最小信号），并给出
判分 prompt 需要区分的输出类型样例。
"""
import json, random
import numpy as np
from collections import defaultdict

rng = random.Random(7)
STIM = json.load(open(r"D:\djr82\memory_boundary\stimuli_v1.json", encoding="utf-8"))["stimuli"]
EMB = None  # lazy


def get_emb():
    global EMB
    if EMB is None:
        from sentence_transformers import SentenceTransformer
        EMB = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return EMB


def cos(a, b):
    return float(np.dot(a, b))


class MiniRegistry:
    """A1/A2/A3/A4 的机械记忆模型。每条 entry: text, cos-emb, t, active."""

    def __init__(self, mode_revise, mode_recall, threshold=0.55):
        self.mode_revise = mode_revise      # "append" | "supersede"
        self.mode_recall = mode_recall      # "semantic" | "semantic+time"
        self.threshold = threshold
        self.entries = []                   # dicts

    def write(self, text, emb, t):
        if self.mode_revise == "supersede":
            # 找语义最近（>0.85 同事实族）的 active 条目 → 显式取代
            best, bs = None, 0.0
            for e in self.entries:
                if e["active"]:
                    s = cos(emb, e["emb"])
                    if s > bs:
                        bs, best = s, e
            if best is not None and bs > 0.85:
                best["active"] = False
                best["superseded_by"] = len(self.entries)
        self.entries.append({"text": text, "emb": emb, "t": t, "active": True})

    def recall(self, q_emb):
        cands = [(cos(q_emb, e["emb"]), e) for e in self.entries
                 if (e["active"] if self.mode_recall == "semantic+time" else True)]
        if not cands:
            return None, 0.0
        cands.sort(key=lambda x: -x[0])
        s, e = cands[0]
        if s < self.threshold:
            return None, s
        return e, s


def question_for(pair, ask_new):
    """问句：对 old 或 new 状态的提问。用陈述句模板（机械召回）。"""
    return pair["new"] if ask_new else pair["old"]


def judge_outcome(pair, asked_about, recalled_text, recalled_active):
    """机械判分（LLM 裁判前的规则代理）：
    correct_new   = 问新状态且召回 new
    correct_old   = 问旧状态且回召 old（append-only 合法）
    stale         = 问新状态但召回了已被取代的 old（时序错乱）
    lost_update   = 问旧状态但 old 已被 supersede 且无法召回（过度遗忘）
    missing       = 阈值下无召回
    """
    if recalled_text is None:
        return "missing"
    target = pair["new"] if asked_about else pair["old"]
    same = recalled_text == target
    if asked_about:
        if same:
            return "correct_new"
        return "stale" if not recalled_active else "wrong"
    else:
        if same:
            return "correct_old"
        return "lost_update" if not recalled_active else "wrong"


def run_arm(arm, stim_subset):
    """跑一臂，返回按族汇总的判定分布。"""
    embs = {}
    mode_revise = "supersede" if "sup" in arm else "append"
    mode_recall = "semantic+time" if "time" in arm else "semantic"
    results = defaultdict(Counter := __import__("collections").Counter)
    model = get_emb()

    for pair in stim_subset:
        reg = MiniRegistry(mode_revise, mode_recall)
        e_old = model.encode(pair["old"], normalize_embeddings=True)
        e_new = model.encode(pair["new"], normalize_embeddings=True)
        reg.write(pair["old"], e_old, t=0)
        # 干预写入：new 在 5 个无关写入后进入（时序压力最小化）
        for k in range(5):
            filler = f"填充记录{k}号关于无关主题{k}{rng.choice(['甲','乙','丙'])}"
            reg.write(filler, model.encode(filler, normalize_embeddings=True), t=k + 1)
        reg.write(pair["new"], e_new, t=6)

        for ask_new in (True, False):
            q = question_for(pair, ask_new)
            q_emb = model.encode(q, normalize_embeddings=True)
            rec, s = reg.recall(q_emb)
            active = rec["active"] if rec else False
            outcome = judge_outcome(pair, ask_new,
                                    rec["text"] if rec else None, active)
            results[(pair["family"], ask_new)][outcome] += 1
    return results


def main():
    import collections
    subset = rng.sample(STIM, 20)      # 试跑 20 对
    all_out = {}
    for arm in ["A1_append_sem", "A2_append_time", "A3_sup_sem", "A4_sup_time"]:
        r = run_arm(arm, subset)
        all_out[arm] = {f"{fam}|{'new' if an else 'old'}": dict(c)
                        for (fam, an), c in r.items()}
        # 汇总关键信号
        stale = sum(c.get("stale", 0) for c in r.values())
        lost = sum(c.get("lost_update", 0) for c in r.values())
        miss = sum(c.get("missing", 0) for c in r.values())
        print(f"{arm}: stale={stale} lost_update={lost} missing={miss}")
    json.dump(all_out, open(r"D:\djr82\memory_boundary\harness_dryrun.json", "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved -> harness_dryrun.json")


if __name__ == "__main__":
    main()
