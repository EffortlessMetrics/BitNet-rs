## 2025-10-21 - BatchEngine Clone Semantics
**Learning:** The `BatchEngine` struct uses `AtomicU64` for metrics but clones them by value (creating new counters) instead of wrapping them in `Arc`. This means metrics collected in spawned tasks (which use a cloned engine) are lost and not aggregated in the main instance.
**Action:** In future refactors, wrap shared metrics in `Arc<AtomicU64>` or a dedicated metrics struct behind `Arc`.

## 2025-10-21 - HashMap Keys Allocation
**Learning:** The codebase heavily uses `String` as map keys even when the data is already owned elsewhere. In `optimize_batch_for_quantization`, using `&str` keys saved significant allocations.
**Action:** Look for patterns where `String` is used as a key but a reference would suffice, especially in hot loops like batch formation.
