# memory-boundary

Failure-boundary phase diagram for memory update systems:
**f(semantic-Δ, state-type, query-time-direction) → which mechanism combination fails where.**

Seven pre-registered experiments (H1-H7) moving from synthetic stimuli to a
production memory store, culminating in a cross-system comparison against
Tencent Hunyuan's Hy-Memory / TriLink conflict reconciler.

## Headline results

| # | Experiment | Result |
|---|---|---|
| H1 | Supersede-gate Δ dependence | trigger rate monotonic in Δ: 64% at 0.9+ bin, 0% below 0.7 |
| H2 | Time-filter cost | 42 lost historical queries, concentrated in high-Δ bins |
| — | Stale decomposition | mid-Δ peak 26% (symmetry trap); attribute updates = 46% epicenter |
| H4 | Cluster-internal time hierarchy | attribute stale 46% → **0%**, zero side effects |
| H4b | Unsupervised clustering (no name rules) | reproduces H4 at **1.8%** residual |
| H5 | Production DB transfer (16,507 entries) | lineage recovery **99.5%**, cross-link **0.6%** at th=0.55 |
| H6 | End-to-end five-arm on real lineages | pure-semantic **14%** → structural **89%** (+75pp) |
| H7 | Cross-system (our stimuli × Hy-Memory reconciler prompt) | **90%** agreement; orthogonal layers, stacking gains |

## Key claims

1. **The fractal-structure claim (Cammarata 2026) transfers to memory stores**:
   entity-version clusters are recoverable unsupervised from embeddings alone,
   and exploiting them (cluster-internal time hierarchy) removes the largest
   failure class of pure-semantic recall.
2. **Query-time direction is a necessary experimental axis**: "recall the old
   version" is confounded between correct-history and stale unless the query
   carries a temporal qualifier ([now] vs [then]).
3. **Structure layer vs judgment layer are orthogonal**: TriLink's LLM-rule
   reconciler reaches 90% semantic agreement on our stimuli, but costs O(N×M)
   LLM calls and has no failure-boundary knowledge. The O(N) structural layer
   (unsupervised clustering + time hierarchy) serves as its candidate
   pre-filter — merged pipeline has stacking gains.

## Files

- `PREREG.md` — pre-registration v0.4 (design, hypotheses H1-H4, failure criteria)
- `build_stimuli.py`, `stimuli_v1.json` — 200 pairs, four families, with measured embedding cosine
- `harness.py`, `full_run.py` — five-arm mechanical harness + full run
- `h4_test.py`, `h4b_unsupervised.py` — explicit-rule and unsupervised cluster-hierarchy tests
- `h5_realdb.py` — production-store validation (flymemory, supersede lineages as ground truth)
- `h6_e2e.py` — end-to-end five-arm on real lineages
- `h7_cross.py` — cross-system comparison (Hy-Memory reconciler prompt, DeepSeek runner)
- `judge.py` — external judge protocol (temporal-qualifier version, 5/5 critical cells)
- `FULL_RUN_REPORT.md` — cumulative report v1.6
- `VERSUS_HYMEMORY.md` — six advantages flymemory holds that Hy-Memory lacks

## Normalize-before-compare

`flymemory_v3.pkl` stores **unnormalized** embeddings (norms 3-4). All
cosine comparisons must normalize first — the unnormalized inner product
(2.7-8.3) exceeds every threshold and produces garbage cluster diagnostics.
