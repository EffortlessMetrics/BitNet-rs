# FFI Build Hygiene Quick Reference

**Status**: Ready for Implementation
**Priority**: P1 (Production-Ready FFI Build System)
**Time Estimate**: 1 hour
**Scope**: Unix/Linux/macOS (GCC/Clang)

---

## Three Priority 1 Fixes

### 1. Warning Visibility (5 minutes)
**File**: `crates/bitnet-ggml-ffi/build.rs`
**Lines**: 9, 14
**Change**: `eprintln!(` → `println!(`

### 2. Compiler Flag Spacing (5 minutes)
**File**: `crates/bitnet-ggml-ffi/build.rs`
**Lines**: 45-46
**Change**:
```rust
// Before
.flag("-isystemcsrc/ggml/include")
.flag("-isystemcsrc/ggml/src")

// After
.flag("-isystem")
.flag("csrc/ggml/include")
.flag("-isystem")
.flag("csrc/ggml/src")
```

### 3. Vendor Commit Population (10 minutes)
**File**: `crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT`
**Command**:
```bash
echo "b4247" > crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT
```

---

## Quick Verification

```bash
# Clean and rebuild
cargo clean -p bitnet-ggml-ffi
cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi

# Verify CI enforcement passes
CI=1 cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi

# Test FFI smoke build
cargo build --workspace --no-default-features --features ffi --exclude bitnet-sys --exclude crossval
```

---

## Success Criteria

- [x] Build warnings visible in cargo output
- [x] Compiler receives `-isystem csrc/ggml/include` (space-separated)
- [x] CI builds succeed (no "unknown" panic)
- [x] FFI smoke job passes on GCC/Clang

---

## Deferred to Priority 2

- **MSVC Support**: `/external:I` flags and Windows CI (2-3 hours)
- **Xtask Automation**: `cargo xtask vendor-ggml` command (45 minutes)

---

## Documentation

- **Full Plan**: `FFI_BUILD_HYGIENE_ACTION_PLAN.md` (comprehensive implementation guide)
- **Status Report**: `FFI_BUILD_HYGIENE_STATUS_REPORT.md` (complete assessment)
- **Audit**: `FFI_BUILD_HYGIENE_AUDIT.md` (detailed analysis)

