## 2025-05-15 - HashMap Key Allocation in Hot Paths
**Learning:** In `BatchEngine` hot paths (like `optimize_batch_for_quantization`), using `String` keys for grouping requests causes unnecessary allocations ($O(N)$ where $N$ is batch size). Since `PendingRequest` outlives the local grouping map, `&str` keys should be used.
**Action:** Always check if `HashMap` keys can be borrowed (`&str` or `&T`) from source data instead of owned (`String`), especially in high-throughput loops.
