## 2024-10-25 - Atomic Cloning Anti-Pattern
**Learning:** `BatchEngine` uses `AtomicU64` fields that are manually cloned (value copied) when `self.clone()` is called for `tokio::spawn`. This creates independent counters in spawned tasks, leading to split-brain statistics where the main instance sees 0 for metrics updated by workers.
**Action:** When implementing shared state for `tokio::spawn`, always wrap atomic counters in `Arc<AtomicU64>` or `Arc<struct_with_atomics>` instead of holding them directly in the struct if the struct is cloned.

## 2024-10-25 - HashMap Key Allocations
**Learning:** `optimize_batch_for_quantization` was allocating `String` keys for every request in the candidate list to group them. In high-throughput batching systems, these allocations in the hot path add up.
**Action:** Use `HashMap<&str, ...>` with borrowed keys from the source data (which usually outlives the temporary map) to avoid allocations.
