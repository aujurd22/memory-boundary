# memory_boundary

动态记忆更新的失效边界——f(语义Δ, 状态类型) 相图。

## 文件
- PREREG.md        预注册草稿 v0.1（跑批前锁死）
- build_stimuli.py 刺激集生成器
- stimuli_v1.json  200 对刺激，v0.3（含实测嵌入余弦 `cos` 字段）

## 刺激集质检记录（2026-10-15）
- v0.1 质检问题: state_type 全落 progress（模板选择 bug）；paraphrase 100% 退化 old==new
- v0.2 修复类型分布，但 paraphrase 仍退化（同 key 无第二模板候选）
- v0.3 加同义词表层（SYNONYM_V 值变体 + SYNONYM_T 模板变体），退化 0/50

## 实测语义Δ（paraphrase-multilingual-MiniLM-L12-v2, cos(old,new)）
- paraphrase mean=0.845 [0.538, 0.965]   （预注册目标 0.9+，实测略低）
- attribute  mean=0.767 [0.496, 0.960]   （目标 0.7-0.9 ✓）
- flip       mean=0.731 [0.478, 0.940]   （目标 0.7-0.9 ✓）
- unrelated  mean=0.495 [0.281, 0.829]   （目标 <0.5 ✓ 边缘，max 0.829 有交叉）

## 预注册纪律决定
生成器不再按实测结果回调（防 stimulus p-hacking）。跑批按【实测 cos 分箱】
而非族名义分箱；族标签仅作辅助维度。跨族交叉（如 unrelated max 0.829）保留，
由实测分箱自然处理。

## 下一步
- 五臂 harness（对接 flymemory 本地库或独立 mini-registry）
- 裁判外置 prompt 设计（异源模型二选一判分）
- 2026-11 下旬正式跑批
