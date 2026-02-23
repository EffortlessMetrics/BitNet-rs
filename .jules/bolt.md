## 2025-05-18 - Sampling Performance Benchmarking
**Learning:** Performance benchmarking of tight loops (like sampling strategies) in `bitnet-inference` is misleading in debug mode.
**Action:** Always run benchmarks with `--release`. Algorithmic improvements that reduce complexity from O(N) to O(1) may only show 30% improvement in debug mode due to overhead, but show 10x improvement in release mode.
