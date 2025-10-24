# FFI Build Hygiene: Exact Code Changes

**Implementation Guide**: This document shows the **exact code changes** needed for the three Priority 1 FFI build hygiene fixes.

---

## File 1: crates/bitnet-ggml-ffi/build.rs

### Change 1: Line 9 (Warning Visibility)

**Before**:
```rust
        let commit = fs::read_to_string(marker).unwrap_or_else(|e| {
            eprintln!(
                "cargo:warning=Failed to read VENDORED_GGML_COMMIT marker at '{}': {}",
                marker.display(),
                e
            );
```

**After**:
```rust
        let commit = fs::read_to_string(marker).unwrap_or_else(|e| {
            println!(
                "cargo:warning=Failed to read VENDORED_GGML_COMMIT marker at '{}': {}",
                marker.display(),
                e
            );
```

**Change**: `eprintln!(` → `println!(`

---

### Change 2: Line 14 (Warning Visibility)

**Before**:
```rust
            eprintln!(
                "cargo:warning=Using 'unknown' as fallback. Run 'cargo xtask vendor-ggml' to fix."
            );
```

**After**:
```rust
            println!(
                "cargo:warning=Using 'unknown' as fallback. Run 'cargo xtask vendor-ggml' to fix."
            );
```

**Change**: `eprintln!(` → `println!(`

---

### Change 3: Lines 45-46 (Compiler Flag Spacing)

**Before**:
```rust
            // AC6: Use -isystem for vendored GGML headers (third-party code)
            // This suppresses warnings from the vendored GGML implementation
            // while preserving warnings from our shim code.
            .flag("-isystemcsrc/ggml/include")
            .flag("-isystemcsrc/ggml/src")
```

**After**:
```rust
            // AC6: Use -isystem for vendored GGML headers (third-party code)
            // This suppresses warnings from the vendored GGML implementation
            // while preserving warnings from our shim code.
            .flag("-isystem")
            .flag("csrc/ggml/include")
            .flag("-isystem")
            .flag("csrc/ggml/src")
```

**Change**: Split concatenated flags into separate calls (POSIX-compliant)

---

## File 2: crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT

### Change 4: File Content

**Before**:
```
unknown
```

**After**:
```
b4247
```

**Command**:
```bash
echo "b4247" > crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT
```

---

## Complete build.rs After All Changes

```rust
fn main() {
    // Only build the C shim when the feature is on.
    if std::env::var("CARGO_FEATURE_IQ2S_FFI").is_ok() {
        use std::{fs, path::Path};

        // Inject vendored commit into the crate's env
        let marker = Path::new("csrc/VENDORED_GGML_COMMIT");
        let commit = fs::read_to_string(marker).unwrap_or_else(|e| {
            println!(                                                    // ← CHANGED (was eprintln!)
                "cargo:warning=Failed to read VENDORED_GGML_COMMIT marker at '{}': {}",
                marker.display(),
                e
            );
            println!(                                                    // ← CHANGED (was eprintln!)
                "cargo:warning=Using 'unknown' as fallback. Run 'cargo xtask vendor-ggml' to fix."
            );
            "unknown".into()
        });
        let commit = commit.trim().to_string();
        println!("cargo:rustc-env=BITNET_GGML_COMMIT={}", commit);

        // In CI, fail fast if the marker is missing or unknown
        if std::env::var("CI").is_ok() && commit == "unknown" {
            panic!(
                "VENDORED_GGML_COMMIT is 'unknown' in CI.\n\
                 Run: cargo xtask vendor-ggml --commit <sha>\n\
                 Or set crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT"
            );
        }
        let ggml_quants_src = fs::read_to_string("csrc/ggml/src/ggml-quants.c")
            .expect("Failed to read 'csrc/ggml/src/ggml-quants.c'. Ensure GGML sources are vendored via 'cargo xtask vendor-ggml'.");

        let mut build = cc::Build::new();
        if ggml_quants_src.contains("assert(k % QK_IQ2_S == 0)") {
            build.define("BITNET_IQ2S_DEQUANT_NEEDS_QK_MULTIPLE", None);
        }

        build
            .file("csrc/ggml_quants_shim.c")
            .file("csrc/ggml_consts.c") // Constants extraction
            .include("csrc") // Local includes (use -I, warnings visible)
            // AC6: Use -isystem for vendored GGML headers (third-party code)
            // This suppresses warnings from the vendored GGML implementation
            // while preserving warnings from our shim code.
            .flag("-isystem")                                            // ← CHANGED (split flag)
            .flag("csrc/ggml/include")                                   // ← CHANGED (split path)
            .flag("-isystem")                                            // ← CHANGED (split flag)
            .flag("csrc/ggml/src")                                       // ← CHANGED (split path)
            // Define for IQ quantization family
            .define("GGML_USE_K_QUANTS", None)
            .define("QK_IQ2_S", "256")
            // Optimization and warnings
            .flag_if_supported("-O3")
            .flag_if_supported("-fPIC")
            // AC6: Selective warning suppression for vendored code patterns
            // These affect the vendored GGML code but not our local shim code
            // (which uses -I, not -isystem)
            .flag_if_supported("-Wno-sign-compare")
            .flag_if_supported("-Wno-unused-parameter")
            .flag_if_supported("-Wno-unused-function")
            .flag_if_supported("-Wno-unused-variable")
            .flag_if_supported("-Wno-unused-but-set-variable")
            .compile("bitnet_ggml_quants_shim");

        println!("cargo:rerun-if-changed=csrc/ggml_quants_shim.c");
        println!("cargo:rerun-if-changed=csrc/ggml_consts.c");
        println!("cargo:rerun-if-changed=csrc/ggml/src/ggml-quants.c");
        println!("cargo:rerun-if-changed=csrc/ggml/src/ggml-quants.h");
        println!("cargo:rerun-if-changed=csrc/ggml/include/ggml/ggml.h");
        println!("cargo:rerun-if-changed=csrc/VENDORED_GGML_COMMIT");
    }
}
```

---

## Verification Commands

```bash
# 1. Verify changes applied
diff -u <(git show HEAD:crates/bitnet-ggml-ffi/build.rs) crates/bitnet-ggml-ffi/build.rs

# 2. Verify VENDORED_GGML_COMMIT content
cat crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT
# Expected: b4247

# 3. Clean build
cargo clean -p bitnet-ggml-ffi

# 4. Test build with feature flag
cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi

# 5. Verify CI enforcement passes
CI=1 cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi

# 6. Check compiler flags (optional - requires verbose logging)
CARGO_LOG=cargo::core::compiler=trace cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi 2>&1 | grep -- '-isystem'
# Expected output: -isystem csrc/ggml/include (space-separated)
```

---

## Git Diff Preview

```diff
diff --git a/crates/bitnet-ggml-ffi/build.rs b/crates/bitnet-ggml-ffi/build.rs
index abc123..def456 100644
--- a/crates/bitnet-ggml-ffi/build.rs
+++ b/crates/bitnet-ggml-ffi/build.rs
@@ -7,12 +7,12 @@ fn main() {
         // Inject vendored commit into the crate's env
         let marker = Path::new("csrc/VENDORED_GGML_COMMIT");
         let commit = fs::read_to_string(marker).unwrap_or_else(|e| {
-            eprintln!(
+            println!(
                 "cargo:warning=Failed to read VENDORED_GGML_COMMIT marker at '{}': {}",
                 marker.display(),
                 e
             );
-            eprintln!(
+            println!(
                 "cargo:warning=Using 'unknown' as fallback. Run 'cargo xtask vendor-ggml' to fix."
             );
             "unknown".into()
@@ -42,8 +42,10 @@ fn main() {
             // AC6: Use -isystem for vendored GGML headers (third-party code)
             // This suppresses warnings from the vendored GGML implementation
             // while preserving warnings from our shim code.
-            .flag("-isystemcsrc/ggml/include")
-            .flag("-isystemcsrc/ggml/src")
+            .flag("-isystem")
+            .flag("csrc/ggml/include")
+            .flag("-isystem")
+            .flag("csrc/ggml/src")
             // Define for IQ quantization family
             .define("GGML_USE_K_QUANTS", None)
             .define("QK_IQ2_S", "256")

diff --git a/crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT b/crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT
index 1234567..89abcdef 100644
--- a/crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT
+++ b/crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT
@@ -1 +1 @@
-unknown
+b4247
```

---

## Summary

**Total Changes**: 4 modifications across 2 files
**Lines Modified**: 6 lines in build.rs + 1 line in VENDORED_GGML_COMMIT
**Time to Apply**: 10 minutes (manual editing)
**Risk Level**: Low (pure refactoring, no logic changes)

**Files**:
1. `crates/bitnet-ggml-ffi/build.rs` (3 changes: lines 9, 14, 45-46)
2. `crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT` (1 change: content)

**Next Steps**: Apply changes, verify builds, commit to Git
