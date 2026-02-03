## 2024-11-14 - Atomic Counters in Structs
**Learning:** In Rust, `Atomic` types (like `AtomicU64`) are not reference-counted. Embedding them directly in a struct means `Clone` creates *new* independent counters if not wrapped in `Arc`.
**Action:** When implementing shared counters in a struct that will be cloned (e.g., for spawned tasks), always wrap them in `Arc<Atomic...>` to ensure updates are visible across clones.
