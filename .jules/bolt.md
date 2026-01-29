## 2025-10-21 - Shared State in Clones
**Learning:** `Atomic` types in structs that are cloned (e.g., for `tokio::spawn`) are copied by value (snapshot), not shared. This breaks metrics tracking in spawned tasks.
**Action:** Always wrap `Atomic` types in `Arc` (e.g., `Arc<AtomicU64>`) if they need to be shared across clones.

## 2025-10-21 - String Allocations in Loops
**Learning:** Using `String` keys in HashMaps during tight loops (like batch formation) causes excessive heap allocations.
**Action:** Use `HashMap<&str, V>` when keys can be borrowed from existing data (like `candidates` vector) to avoid `to_string()` calls.
