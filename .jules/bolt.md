# Bolt's Journal - Critical Learnings

This journal tracks critical performance learnings, architectural bottlenecks, and optimization patterns for the BitNet-rs project.

## 2024-05-22 - [Initial Setup]
**Learning:** Performance optimizations should be measured and verified.
**Action:** Always verify with benchmarks or measurable impact.

## 2024-05-22 - [HashMap Key Allocation]
**Learning:** Using `String` keys in HashMaps for temporary grouping operations causes unnecessary allocations.
**Action:** Use `&str` keys referencing the original data source whenever the map's lifetime is shorter than the source data.
