# Test API Contract & CI Enforcement

This document describes the test harness API contract and enforcement mechanisms that prevent breaking changes.

## What's Protected

### 1. Test Result Types
- **Old/Banned**: `TestResult<T>` (generic fallible type)
- **New/Required**: `TestResultCompat<T>` (for fallible operations)
- **New/Required**: `TestRecord` (for test outcomes)

### 2. Constructor Methods
- **Old/Banned**: `TestResult::passed()`, `TestResult::failed()`, `TestResult::timeout()`
- **New/Required**: `TestRecord::passed()`, `TestRecord::failed()`, `TestRecord::timeout()`

### 3. Type Aliases
- **Old/Banned**: `TestResultData`
- **New/Required**: `TestRecord`

### 4. Code Style
- **Banned**: Split doc comments after closing braces (`} ///`)
- **Banned**: `&Vec<T>` parameters (use `&[T]` instead)

## Enforcement Layers

### Layer 1: Local Git Hooks
```bash
# Install the hooks
bash scripts/install-hooks.sh

# Or use Python pre-commit (more features)
pip install pre-commit
pre-commit install
```

**Pre-commit checks:**
- `cargo fmt --check`
- `cargo clippy` with `-D warnings -D clippy::ptr_arg`
- Banned patterns script
- Test compilation check

**Pre-push checks:**
- Test build verification
- cargo-deny security audit (if installed)

### Layer 2: CI/CD Pipeline
The GitHub Actions workflow mirrors local hooks:
- Format checking
- Clippy with strict flags
- Banned patterns enforcement
- Test compilation with `-Dwarnings`
- cargo-deny security checks

### Layer 3: Code Organization
- **Prelude Pattern**: All test utilities exported through `bitnet_tests::prelude::*`
- **Builder Pattern**: Test structs use `Default` to prevent field addition breakage
- **Private Fields**: Encapsulation prevents direct field access

## Configuration Files

### `rust-toolchain.toml`
Pins Rust version for consistent formatting and lints across all environments.

### `deny.toml`
Enforces supply chain security:
- Denies known vulnerabilities
- Blocks unmaintained crates
- Enforces license compliance
- Prevents yanked dependencies

### `.editorconfig`
Ensures consistent formatting across editors:
- LF line endings
- UTF-8 encoding
- 4-space indentation for Rust
- 2-space indentation for configs

### `taplo.toml`
TOML formatter configuration for consistent Cargo.toml formatting.

### `lefthook.yml` / `.pre-commit-config.yaml`
Defines git hook configuration for automated checks.

## Banned Patterns Script

Located at `scripts/hooks/banned-patterns.sh`, this script enforces:

1. **No legacy TestResult<T>** in tests/common
2. **No TestResultData** type alias
3. **No TestResult:: constructors**
4. **No split doc comments**
5. **No &Vec<T> parameters**

## Quick Reference

### Running Checks Locally
```bash
# Format code
cargo fmt --all

# Run clippy with strict checks
RUSTFLAGS="-Dwarnings" cargo clippy --workspace --all-features --all-targets -- -D warnings -D clippy::ptr_arg

# Check banned patterns
bash scripts/hooks/banned-patterns.sh

# Check tests compile
cargo check --no-default-features --workspace --tests --no-default-features --features cpu

# Run security audit
cargo deny check --hide-inclusion-graph
```

### Installing Tools
```bash
# Required
cargo install cargo-deny taplo-cli

# Optional (for Python pre-commit)
pip install pre-commit

# Install git hooks
bash scripts/install-hooks.sh
# OR
pre-commit install
```

## Migration Guide

If you encounter errors after this setup:

### TestResult Issues
```rust
// OLD (will fail)
fn test() -> TestResult<()> { ... }
TestResult::passed()

// NEW (correct)
fn test() -> TestResultCompat<()> { ... }
TestRecord::passed()
```

### Import Issues
```rust
// OLD (verbose)
use bitnet_tests::common::results::TestRecord;
use bitnet_tests::common::fixtures::FixtureManager;

// NEW (prelude)
use bitnet_tests::prelude::*;
```

### Parameter Issues
```rust
// OLD (will fail)
fn process(items: &Vec<String>) { ... }

// NEW (correct)  
fn process(items: &[String]) { ... }
```

## Rationale

This multi-layered approach ensures:
1. **Immediate feedback** during development (git hooks)
2. **Consistent enforcement** across all contributors (CI)
3. **Clear migration path** for existing code
4. **Future-proof design** with builder patterns and encapsulation

The test harness is now stable and maintainable long-term.