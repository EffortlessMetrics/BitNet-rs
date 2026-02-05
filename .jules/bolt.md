## 2024-05-22 - [Allocations in hot loops]
**Learning:** Using `String` keys in `HashMap` inside tight loops (like batch formation) causes unnecessary allocations. Using `&str` references to existing data (like in `candidates`) avoids this.
**Action:** Always check map keys in hot paths. Use borrowed types when ownership is not needed.
