## 2024-10-24 - [Shared Atomics in Cloned Structs]
**Learning:** Atomic types (AtomicU64) inside a struct are deep-copied if the Clone implementation initializes new atomics with loaded values. To share state across clones (e.g. for spawned tasks), they must be wrapped in Arc.
**Action:** Always wrap global/shared counters in Arc<AtomicU64> when the containing struct is expected to be cloned.

## 2024-10-24 - [Avoid Clone by Moving with Option::take]
**Learning:** When filtering or selecting items from a Vec to move into a new structure, converting the Vec to Vec<Option<T>> allows using .take() to move items out by index without cloning, even if random access is needed.
**Action:** Use Vec<Option<T>> + take() for non-destructive moves from a collection when ownership transfer is needed but the source collection must remain valid (or indices must be preserved).
