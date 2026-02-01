## 2024-11-22 - HashMap Key Allocation Optimization
**Learning:** Using `String` as a `HashMap` key when `&str` is available from the source data (with sufficient lifetime) causes unnecessary heap allocations on every insertion. In high-frequency hot paths like batch formation, this adds up.
**Action:** Always prefer `HashMap<&str, V>` over `HashMap<String, V>` when keys can be borrowed from a longer-lived structure, especially in loops.
