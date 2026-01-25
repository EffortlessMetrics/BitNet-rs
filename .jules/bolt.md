## 2025-10-21 - [Zero-Allocation HashMap Keys]
**Learning:** In `BatchEngine`, request grouping used `String` keys derived from `&str` hints, causing N allocations per batch. Since the source strings outlive the map scope, `HashMap<&str, ...>` is perfectly safe and saves all key allocations.
**Action:** When grouping items based on string properties, always check if the property lifetime allows using `&str` keys before allocating `String`s.
