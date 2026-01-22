## 2024-05-23 - [BatchEngine Clone Bug & Allocation Optimization]
**Learning:** Rust's `Clone` implementation for structs with `Atomic` fields must be handled carefully. Using `AtomicU64::new(load())` in `Clone` creates a NEW independent counter, breaking shared state metrics. Wrapping atomics in `Arc` ensures shared state across clones.
**Action:** Always wrap shared atomic counters in `Arc` if the struct is meant to be cloned and shared (e.g., passed to spawned tasks). Also, watch out for dropped channels in async flows—ensure senders are stored or used before dropping.
