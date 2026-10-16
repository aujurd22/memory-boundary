"""记忆更新失效边界：刺激集构建器 v0.2
四族（同义改述/属性更新/状态翻转/无关新条目）×200 对，每对带已知正确答案。
修复 v0.1 两个质检问题：state_type 全落 progress、unrelated 族 key 冲突。
产出: stimuli_v1.json
"""
import json, random
from collections import Counter

OUT = r"D:\djr82\memory_boundary\stimuli_v1.json"
rng = random.Random(7)

NAMES = [f"实体{i:02d}" for i in range(1, 41)]

# 模板显式带状态类型；每个模板标 key
TPL = {
    "fact":     [("容器", "{n}的容器是{v}"), ("位置", "{n}的位置在{v}"), ("颜色", "{n}的颜色是{v}")],
    "pref":     [("模式", "{n}偏好的模式是{v}"), ("版本", "{n}喜欢的版本是{v}")],
    "progress": [("状态", "{n}的状态是{v}"), ("进度", "{n}的进度是{v}")],
}
VALS = {
    "容器": ["盒子A", "盒子B", "盒子C", "箱子D"],
    "位置": ["架子1", "架子2", "抽屉3", "桌面4"],
    "颜色": ["红", "蓝", "绿", "黄"],
    "模式": ["快速", "精确", "均衡", "省电"],
    "版本": ["v1", "v2", "v3"],
    "状态": ["开启", "关闭", "暂停", "完成"],
    "进度": ["0%", "25%", "50%", "75%", "100%"],
}
# 可翻转的 key（语义上"新值替代旧值"成立）
FLIPPABLE = ("状态", "进度", "位置")
STYPES = ("fact", "pref", "progress")


def pick_tpl(st, key=None):
    """选 (key, template)；key 给定则在该 key 内选，否则任意。"""
    cands = [kt for kt in TPL[st] if key is None or kt[0] == key]
    return rng.choice(cands)


def synth(kind, n):
    out = []
    for i in range(n):
        st = rng.choice(STYPES)
        key_old, t_old = pick_tpl(st)
        n_ent = rng.choice(NAMES)
        v_old = rng.choice(VALS[key_old])
        old = t_old.format(n=n_ent, v=v_old)

        if kind == "paraphrase":
            # 同义改述：换模板（若可用）但同 key 同值 → Δ 最高
            alts = [kt for kt in TPL[st] if kt[0] == key_old and kt[1] != t_old]
            t_new = rng.choice(alts)[1] if alts else t_old
            new = t_new.format(n=n_ent, v=v_old)
            expected = "either_same_content"
        elif kind == "attribute":
            # 属性更新：同 key 换值
            v_new = rng.choice([v for v in VALS[key_old] if v != v_old])
            new = t_old.format(n=n_ent, v=v_new)
            expected = "new"
        elif kind == "flip":
            # 状态翻转：只用品类上可翻转的 key；若当前 st 无可翻转 key，
            # 换到有可翻转 key 的状态类型（progress 恒有）
            if key_old not in FLIPPABLE:
                flip_sts = [s2 for s2 in STYPES
                            if any(k in FLIPPABLE for k, _ in TPL[s2])]
                st = rng.choice(flip_sts)
                key_old, t_old = pick_tpl(
                    st, key=rng.choice([k for k, _ in TPL[st] if k in FLIPPABLE]))
                v_old = rng.choice(VALS[key_old])
                old = t_old.format(n=n_ent, v=v_old)
            v_new = rng.choice([v for v in VALS[key_old] if v != v_old])
            new = t_old.format(n=n_ent, v=v_new)
            expected = "new_strict"
        else:
            # unrelated：不同 key（值域独立），旧条目不动
            other_keys = [k for k in VALS if k != key_old]
            key_new = rng.choice(other_keys)
            all_t = [(k, t) for ts in TPL.values() for k, t in ts]
            _, t_new = rng.choice([kt for kt in all_t if kt[0] == key_new])
            new = t_new.format(n=n_ent, v=rng.choice(VALS[key_new]))
            expected = "both_independent"

        out.append({"qid": f"{kind[:3]}-{i:03d}", "family": kind,
                    "state_type": st, "old": old, "new": new,
                    "expected_recall": expected})
    return out


def main():
    stim = []
    for kind, n in [("paraphrase", 50), ("attribute", 50),
                    ("flip", 50), ("unrelated", 50)]:
        stim += synth(kind, n)
    rng.shuffle(stim)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"version": "v0.2", "n": len(stim), "stimuli": stim},
                  f, ensure_ascii=False, indent=1)
    print(f"v0.2: {len(stim)} 对")
    print("  族:", dict(Counter(s["family"] for s in stim)))
    print("  类型:", dict(Counter(s["state_type"] for s in stim)))
    print("  期望:", dict(Counter(s["expected_recall"] for s in stim)))
    for kind in ("paraphrase", "attribute", "flip", "unrelated"):
        for s in [x for x in stim if x["family"] == kind][:2]:
            print(f"  [{s['family']}] {s['old']}  ||  {s['new']}")


if __name__ == "__main__":
    main()
