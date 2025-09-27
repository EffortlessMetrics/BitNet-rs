# API Changes and Migration Guide

This document tracks all API changes in BitNet.rs and provides migration guidance for breaking changes.

## Version Policy

- **Major version (X.0.0)**: Breaking API changes
- **Minor version (0.X.0)**: New features, backward compatible
- **Patch version (0.0.X)**: Bug fixes, no API changes

## How to Handle API Changes

### For Contributors

1. **Before making API changes**:
   - Run `just api-check` to verify current baseline
   - Consider if the change is necessary
   - Prefer extending rather than modifying existing APIs

2. **When making intentional API changes**:
   - Update baselines: `cargo public-api -p <crate> > api/rust/<crate>.public-api.txt`
   - Update FFI header: `cbindgen crates/bitnet-ffi --config api/ffi/cbindgen.toml -o api/ffi/bitnet_ffi.h`
   - Update CLI help: `cargo run -p bitnet-cli -- --help > api/cli/help.txt`
   - Add entry to this file with migration notes
   - Bump version according to semver

3. **CI will check**:
   - Rust public API diffs
   - FFI header/symbol changes
   - CLI contract changes
   - Semver compliance

### For Users

Check this file when upgrading to understand what changed and how to migrate.

---

## Change Log

### [Unreleased]

#### Added
- Comprehensive API contract system with baselines
- Automatic breaking change detection in CI
- API snapshot testing with insta

#### Changed
- Default features are now empty (was `["cpu"]`)
- `gpu` feature renamed to `cuda` (alias kept for compatibility)

#### Migration Notes
- Explicitly enable features: `--features cpu` or `--features cuda`
- Update `gpu` references to `cuda` in Cargo.toml

---

### [1.0.0] - 2025-01-19

#### Initial Stable Release

**Rust API**:
- `bitnet-common`: Core types and traits
- `bitnet-kernels`: SIMD and CUDA kernels
- `bitnet-inference`: High-level inference API
- `bitnet-ffi`: C FFI bindings
- `bitnet-cli`: Command-line interface

**FFI API**:
- `bitnet_create_context`: Create inference context
- `bitnet_load_model`: Load GGUF models
- `bitnet_generate`: Generate text
- `bitnet_destroy_context`: Clean up resources
- `bitnet_get_last_error`: Error handling
- `bitnet_version`: Version information

**CLI Commands**:
- `infer`: Run inference
- `convert`: Convert model formats
- `benchmark`: Performance testing
- `server`: HTTP inference server

---

## Deprecation Schedule

Features marked as deprecated will be removed in the next major version.

### Currently Deprecated
- None

### Planned Deprecations
- None

---

## FFI ABI Stability

The FFI API maintains ABI compatibility within major versions:

- Struct sizes are fixed
- Function signatures are stable
- New functions added at end of symbol table
- Existing symbols never removed in minor versions

To verify ABI compatibility:
```bash
nm -D target/release/libbitnet_ffi.so | awk '{print $3}' | sort
diff api/ffi/ffi.symbols.txt <(nm -D target/release/libbitnet_ffi.so | awk '{print $3}' | sort)
```

---

## Python API Stability

The Python bindings follow the Rust API with these guarantees:

- Type stubs (.pyi) are always up-to-date
- Deprecated features raise `DeprecationWarning`
- Breaking changes require major version bump

---

## Experimental Features

Features behind `unstable-*` flags are not covered by semver:

- May change without notice
- Not included in API baselines
- Use at your own risk

Current experimental features:
- None

---

## Support Policy

- Latest major version: Full support
- Previous major version: Security fixes for 6 months
- Older versions: No support

Report API issues: https://github.com/yourusername/bitnet-rs/issues