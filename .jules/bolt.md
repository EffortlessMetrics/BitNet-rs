## 2025-05-23 - Sampling Strategy Context Caching
**Learning:** Rebuilding frequency maps from scratch for repetition penalty in every generation step is a significant O(N^2) bottleneck. Incremental updates using a cached context and diffing (finding common prefix) reduces per-step complexity to amortized O(1).
**Action:** When implementing stateful generation strategies that depend on full context history, always use incremental state updates instead of stateless re-computation.
