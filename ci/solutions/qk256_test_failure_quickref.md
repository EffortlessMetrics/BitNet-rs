# QK256 Test Failure - Quick Reference

**Navigation:** [ci/](../) → [solutions/](./00_NAVIGATION_INDEX.md) → This Document
**Related:** [PR #475 Summary](../PR_475_FINAL_SUMMARY.md)

---

## The Problem
```
Test:        test_qk256_struct_creation (line 533)
Status:      ❌ FAILING
Root Cause:  128-byte tolerance vs strict size validation
Expected:    assert!(result.is_err()) when data is 1 byte short
Actual:      Returns Ok() because 1 byte ≤ 128-byte tolerance
```

## Numbers
```
Expected bytes:     10 × 128 = 1280
Short data:         1280 - 1 = 1279 bytes
Size difference:    1 byte
Tolerance:          128 bytes
Check:              1 ≤ 128 = TRUE → validation passes ✅
Test assertion:     Expects validation to FAIL ❌
```

## Locations
- **Test**: `crates/bitnet-models/tests/qk256_integration.rs:530-542`
- **Implementation**: `crates/bitnet-models/src/quant/i2s_qk256.rs:85-105`
- **Why it exists**: Tolerance added to support alignment padding in real GGUF files

## Introduced In
- **Commit**: `0c57da9d` (PR #468)
- **Date**: October 18, 2025
- **Status**: Pre-existing failure (not a new regression)

## Fix Recommendation
Update test to validate **actual behavior** (tolerance support):

```rust
// Change from:
let short_qs = vec![0u8; rows * row_stride_bytes - 1];
assert!(result.is_err());  // ❌ FAILS

// To:
let within_tolerance = vec![0u8; rows * row_stride_bytes - 64];
assert!(result.is_ok());  // ✅ PASSES

let beyond_tolerance = vec![0u8; rows * row_stride_bytes - 200];
assert!(result.is_err());  // ✅ PASSES
```

## Risk
- 🟢 **No production impact** - real GGUF files aren't off by 1 byte
- 🟢 **Low risk** - tolerance is intentional design
- 🟢 **Easy fix** - just update test expectations

## Next Steps
1. Document as pre-existing (no fix needed for this PR)
2. Create follow-up issue for test updates
3. Consider updating other similar tests (2 more affected)

---

**Full Analysis**: See `qk256_struct_creation_analysis.md`
