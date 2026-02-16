## 2026-02-16 - Repetition Penalty Optimization
**Learning:** `HashMap<u32, usize>` inserts/lookups are too slow ($O(N)$ total) for per-token repetition penalty tracking in generation loops.
**Action:** Use a dense `Vec<u16>` (indexed by token ID) for counts and a sparse `Vec<u32>` for active tokens. This reduces complexity to $O(M)$ where M is number of unique tokens, avoiding hashing overhead entirely.
