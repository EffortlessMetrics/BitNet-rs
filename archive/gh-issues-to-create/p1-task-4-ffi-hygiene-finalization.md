# Issue: FFI Hygiene Finalization - Zero Warnings in bitnet-ggml-ffi

## Context

The `bitnet-ggml-ffi` crate provides FFI bindings to vendored GGML quantization code for IQ2_S support. Currently, the build system uses `-isystem` flags to suppress third-party warnings from vendored GGML headers, but warning hygiene is not enforced in CI. This creates risk of regressions and reduces build signal-to-noise ratio.

Following PR #475 (comprehensive integration), we need to confirm zero-warning builds and add CI enforcement to catch future regressions immediately.

**Affected Components:**
- `crates/bitnet-ggml-ffi/build.rs` - Build system with `-isystem` flag handling
- `crates/bitnet-ggml-ffi/csrc/` - Vendored GGML code and shim files
- `.github/workflows/ci.yml` - CI enforcement for zero warnings
- `xtask/tests/` - Regression test for FFI build hygiene

**Inference Pipeline Impact:**
- Quantization stage - IQ2_S FFI bridge quality affects GGML quantization format compatibility
- Build system - Clean builds improve developer experience and CI reliability

**Performance Implications:**
- Build time: No change (warning suppression already implemented)
- CI reliability: Zero-warning enforcement catches regressions immediately
- Developer experience: Clean build output improves signal-to-noise ratio

## User Story

As a build maintainer, I need zero warnings in bitnet-ggml-ffi so that CI builds are clean and regressions are immediately visible.

## Acceptance Criteria

AC1: Confirm `cargo build -p bitnet-ggml-ffi --features iq2s-ffi` produces zero warnings on Linux (GCC/Clang)
AC2: Validate zero warnings on macOS (Clang) and Windows (MSVC) targets with platform-aware `-isystem`/`/external:I` flag handling
AC3: Add CI enforcement check that fails build if any warnings are detected in FFI crate
AC4: Create regression test in `xtask/tests/ffi_build_hygiene_test.rs` that verifies zero-warning builds
AC5: Document `-isystem` rationale and platform-specific flag handling in `crates/bitnet-ggml-ffi/README.md`
AC6: Ensure `VENDORED_GGML_COMMIT` is tracked and enforced in CI (fail if "unknown")
AC7: Verify `-isystem` flag suppresses warnings from vendored GGML headers while keeping shim code warnings visible
AC8: Add platform-aware compiler detection (GCC/Clang/MSVC) for correct flag selection

## Technical Implementation Notes

- **Affected crates**: `bitnet-ggml-ffi` (FFI bridge crate with C++ vendored dependencies)
- **Pipeline stages**: Quantization (IQ2_S FFI bridge for GGML quantization format)
- **Performance considerations**:
  - Build time: No change (warning suppression already implemented)
  - CI reliability: Zero-warning enforcement prevents regressions
  - Developer experience: Clean build output improves debugging
- **Quantization requirements**: IQ2_S FFI bridge must maintain GGML quantization format compatibility
- **Cross-validation**: Clean FFI builds ensure reliable cross-validation against C++ reference via `cargo run -p xtask -- crossval`
- **Feature flags**: Requires `iq2s-ffi` feature for FFI bridge compilation
- **GGUF compatibility**: FFI bridge maintains GGUF format compatibility for IQ2_S quantization
- **Testing strategy**:
  - TDD with `// AC:ID` tags for each acceptance criterion
  - Build validation: `cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi 2>&1 | grep -i "warning" && exit 1`
  - Regression test: `cargo test -p xtask --test ffi_build_hygiene_test -- --nocapture`
  - Multi-platform validation: Test on Linux (GCC/Clang), macOS (Clang), Windows (MSVC)
  - CI integration: Add zero-warning check to `.github/workflows/ci.yml`

**Platform-Aware Flag Handling (AC2, AC8):**
```rust
// crates/bitnet-ggml-ffi/build.rs (enhanced)
fn main() {
    if std::env::var("CARGO_FEATURE_IQ2S_FFI").is_ok() {
        use std::{fs, path::Path};

        let marker = Path::new("csrc/VENDORED_GGML_COMMIT");
        let commit = fs::read_to_string(marker)
            .unwrap_or_else(|_| "unknown".into())
            .trim()
            .to_string();

        println!("cargo:rustc-env=BITNET_GGML_COMMIT={}", commit);

        // AC6: CI hygiene - fail if vendored commit unknown
        if std::env::var("CI").is_ok() && commit == "unknown" {
            panic!("VENDORED_GGML_COMMIT is 'unknown' in CI. Run: cargo xtask vendor-ggml --commit <sha>");
        }

        let mut build = cc::Build::new();

        // AC8: Platform-aware -isystem flag handling
        let is_msvc = build.get_compiler().is_like_msvc();
        let isystem_flag = if is_msvc {
            "/external:I"  // MSVC uses /external:I for external headers (VS 2019+)
        } else {
            "-isystem"     // GCC/Clang use -isystem
        };

        build
            .file("csrc/ggml_quants_shim.c")
            .file("csrc/ggml_consts.c")
            .include("csrc") // AC7: Local shim code (warnings visible)
            // AC7: Vendored GGML headers (warnings suppressed via -isystem)
            .flag(&format!("{}csrc/ggml/include", isystem_flag))
            .flag(&format!("{}csrc/ggml/src", isystem_flag))
            .define("GGML_USE_K_QUANTS", None)
            .define("QK_IQ2_S", "256")
            .flag_if_supported("-O3")
            .flag_if_supported("-fPIC")
            // Suppress warnings only for vendored code (via -isystem)
            .flag_if_supported("-Wno-sign-compare")
            .flag_if_supported("-Wno-unused-parameter")
            // AC2: MSVC warning suppression for external headers
            .flag_if_supported("/external:W0") // Suppress all warnings from /external:I paths
            .compile("bitnet_ggml_quants_shim");

        // Rebuild triggers
        println!("cargo:rerun-if-changed=csrc/ggml_quants_shim.c");
        println!("cargo:rerun-if-changed=csrc/ggml_consts.c");
        println!("cargo:rerun-if-changed=csrc/VENDORED_GGML_COMMIT");
    }
}
```

**CI Enforcement (AC3):**
```yaml
# .github/workflows/ci.yml
- name: Build FFI crate with zero warnings
  run: |
    cargo clean -p bitnet-ggml-ffi
    BUILD_OUTPUT=$(cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi 2>&1)

    if echo "$BUILD_OUTPUT" | grep -i "warning:"; then
      echo "❌ FFI build has warnings:"
      echo "$BUILD_OUTPUT" | grep -i "warning:"
      exit 1
    fi

    echo "✅ FFI build clean (zero warnings)"
```

**Regression Test (AC4):**
```rust
// xtask/tests/ffi_build_hygiene_test.rs

#[test]
#[cfg(feature = "iq2s-ffi")]
fn test_ffi_build_zero_warnings() { // AC4
    use std::process::Command;

    // Clean build
    Command::new("cargo")
        .args(&["clean", "-p", "bitnet-ggml-ffi"])
        .status()
        .expect("Failed to clean FFI crate");

    // Build with warnings captured
    let output = Command::new("cargo")
        .args(&[
            "build", "-p", "bitnet-ggml-ffi",
            "--no-default-features", "--features", "iq2s-ffi"
        ])
        .output()
        .expect("Failed to build FFI crate");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // AC1: Check for warnings
    let has_warnings = stderr.contains("warning:") || stdout.contains("warning:");

    if has_warnings {
        eprintln!("❌ FFI build output:\n{}\n{}", stdout, stderr);
        panic!("FFI crate build produced warnings (expected zero)");
    }

    println!("✅ FFI build clean (zero warnings)");
}
```

**Validation Commands:**
```bash
# AC1: Clean build test (Linux)
cargo clean -p bitnet-ggml-ffi
cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi 2>&1 | grep -i "warning" && exit 1 || echo "✅ Zero warnings"

# AC2: Test with different compilers (if available)
CC=gcc cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi
CC=clang cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi

# AC4: Regression test
cargo test -p xtask --test ffi_build_hygiene_test -- --nocapture

# AC2: Multi-platform validation (requires cross-compilation setup)
cargo build -p bitnet-ggml-ffi --features iq2s-ffi --target x86_64-unknown-linux-gnu
cargo build -p bitnet-ggml-ffi --features iq2s-ffi --target x86_64-apple-darwin
cargo build -p bitnet-ggml-ffi --features iq2s-ffi --target x86_64-pc-windows-msvc
```

**Estimate**: 2-3 hours

---

<!-- gates:start -->
| Gate | Status | Evidence |
|------|--------|----------|
| spec | ✅ pass | Feature spec created in docs/explanation/specs/SPEC-2025-001-next-steps-priority-implementation.md (Story 4) |
| format | pending | Code formatting with cargo fmt --all --check |
| clippy | pending | Linting with cargo clippy --workspace --all-targets --no-default-features --features cpu -- -D warnings |
| tests | pending | TDD scaffolding with cargo test -p xtask --test ffi_build_hygiene_test |
| build | pending | Build validation with cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi |
| features | pending | Feature smoke testing: iq2s-ffi feature combo |
| benchmarks | pending | Build time baseline validation (no performance regression) |
| docs | pending | Documentation updates in crates/bitnet-ggml-ffi/README.md |
<!-- gates:end -->

<!-- hoplog:start -->
### Hop log
- Created feature spec: Story 4 in docs/explanation/specs/SPEC-2025-001-next-steps-priority-implementation.md
<!-- hoplog:end -->

<!-- decision:start -->
**State:** in-progress
**Why:** Feature spec created and validated, ready for implementation
**Next:** NEXT → implementation with TDD workflow (AC1-AC8 platform-aware flag handling)
<!-- decision:end -->
