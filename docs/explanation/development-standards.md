# BitNet-rs Development Standards & CI/CD

This document outlines the development standards, CI/CD pipeline, and quality enforcement mechanisms for the BitNet-rs project.

## Quick Setup

### Features
- `cpu` (default): pure-Rust, no native deps. Use for local dev and CI.
- `ffi`: enables native backends (C/C++). Use `--features ffi` for smoke builds / perf comparison.
- `gpu`: enables CUDA support for GPU acceleration.
- `crossval`: enables cross-validation against C++ implementation (downloads & builds C++ code).

### Common Development Commands
```bash
# CPU-only quick test compile
cargo qg

# CPU-only benches
cargo bench -p bitnet-quantization --features cpu

# Build with specific features
cargo build --workspace --no-default-features --features cpu
cargo build --workspace --features ffi
```

### Install Git Hooks (Recommended)
```bash
# Install pre-commit (Python-based)
pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type pre-push

# Run pre-commit on all files to verify setup
pre-commit run --all-files

# OR use the simple bash hooks
bash scripts/install-hooks.sh
```

### Quick Quality Gate
Run all quality checks with one command:
```bash
# Unix/Linux/macOS
bash scripts/quality-gate.sh

# Windows PowerShell
scripts\quality-gate.ps1

# Or use the cargo alias (after updating Cargo.toml)
cargo qg  # Quick check for tests (CPU only)
```

### Install Development Tools
```bash
# Required tools
cargo install cargo-deny --locked
cargo install cargo-audit --locked

# Optional quality tools
cargo install cargo-outdated --locked
cargo install cargo-machete --locked
cargo install cargo-llvm-cov --locked
```

## Test API Contract Enforcement

### Protected Patterns

The following patterns are explicitly banned to maintain API stability:

1. **❌ `TestResult<T>` generics in tests/common**
   - Use `TestResultCompat<T>` instead

2. **❌ `TestResultData` type alias**
   - Use `TestRecord` instead

3. **❌ `TestResult::passed/failed/timeout` constructors**
   - Use `TestRecord::passed/failed/timeout` instead

4. **❌ Split doc comments after closing braces**
   ```rust
   // BAD
   }
   /// Documentation
   
   // GOOD
   } // Documentation
   ```

5. **❌ `&Vec<T>` parameters**
   - Use `&[T]` instead

### Enforcement Layers

1. **Local Git Hooks**: Immediate feedback during commit
2. **CI Pipeline**: Automated checks on every PR
3. **Banned Patterns Script**: Explicit pattern checking
4. **Clippy Lints**: Code quality enforcement

## Development Workflow

### Pre-Commit Checks

Run these checks before committing:

```bash
# Format code
cargo fmt --all

# Run clippy with strict checks
RUSTFLAGS="-Dwarnings" cargo clippy --workspace --all-features --all-targets -- -D warnings -D clippy::ptr_arg

# Check tests compile (CPU only to avoid C++ deps)
RUSTFLAGS="-Dwarnings" cargo check --workspace --tests --no-default-features --features cpu

# Check banned patterns
bash scripts/hooks/banned-patterns.sh

# Security audit
cargo deny check --hide-inclusion-graph
```

### Windows PowerShell

```powershell
# Format code
cargo fmt --all

# Set environment and run clippy
$env:RUSTFLAGS = "-Dwarnings"
cargo clippy --workspace --all-features --all-targets -- -D warnings -D clippy::ptr_arg

# Check tests compile
cargo check --workspace --tests --no-default-features --features cpu

# Check banned patterns
pwsh scripts/hooks/banned-patterns.ps1

# Security audit
cargo deny check --hide-inclusion-graph
```

## CI/CD Pipeline

### GitHub Actions Workflow

The CI pipeline runs on every push and PR with these jobs:

1. **Test Suite**: Cross-platform testing (Linux, Windows, macOS)
   - Formatting check (rustfmt)
   - Clippy with strict lints
   - Banned patterns check
   - Unit tests (CPU features only for speed)
   - Cross-compilation tests (ARM64)

2. **Security Audit**: Dependency vulnerability scanning
   - cargo-audit for security advisories
   - cargo-deny for license compliance

3. **Code Quality**: Comprehensive quality checks
   - Unused dependencies (cargo-machete)
   - Outdated dependencies (cargo-outdated)
   - Code coverage (cargo-llvm-cov)
   - Documentation generation

4. **Benchmarks**: Performance regression detection (main branch only)

## Configuration Files

### `rust-toolchain.toml`
Pins the Rust version and required components for reproducible builds.

### `rustfmt.toml`
Stable-only formatting rules for consistent code style.

### `clippy.toml`
Complexity thresholds and MSRV settings for linting.

### `deny.toml`
Security, license, and dependency policies.

### `.editorconfig`
Cross-editor formatting consistency.

### `.pre-commit-config.yaml`
Git hooks configuration for automated checks.

## Workspace Lints

The root `Cargo.toml` defines workspace-level lints:

```toml
[workspace.lints.clippy]
ptr_arg = "deny"         # Enforce &[T] over &Vec<T>
all = "warn"             # Enable all lints
pedantic = "warn"        # Pedantic lints
nursery = "warn"         # Experimental lints
# Allowed exceptions for ML/systems code
module_name_repetitions = "allow"
missing_errors_doc = "allow"
```

## Quality Standards

### Code Style
- Max line width: 100 characters
- Indentation: 4 spaces
- Unix line endings
- Explicit ABI declarations

### Testing
- All tests must compile with CPU-only features
- Tests must pass on all platforms
- No warnings allowed in CI

### Security
- No vulnerable dependencies
- No yanked crates
- Approved licenses only (MIT, Apache-2.0, BSD, etc.)

### Documentation
- Public APIs must be documented
- Examples for complex functions
- README updates for new features

## Troubleshooting

### Common Issues

1. **rustfmt warnings about unstable features**
   - Solution: Use only stable rustfmt options (see rustfmt.toml)

2. **cargo check fails with bitnet-sys**
   - Solution: Use `--no-default-features --features cpu` for CPU-only builds

3. **cargo deny errors**
   - Solution: Check deny.toml configuration and update if needed

4. **Banned pattern violations**
   - Solution: Review scripts/hooks/banned-patterns.sh for specific patterns

### Getting Help

- File issues at: https://github.com/microsoft/BitNet/issues
- Check CI logs for detailed error messages
- Run individual checks locally before pushing

## Maintenance

### Adding New Checks

1. Update `scripts/hooks/banned-patterns.sh` for new patterns
2. Add corresponding check to `.github/workflows/ci.yml`
3. Update this documentation

### Updating Dependencies

```bash
# Check for outdated dependencies
cargo outdated

# Update Cargo.lock
cargo update

# Audit for vulnerabilities
cargo audit
```

### Release Process

1. Run full test suite
2. Update CHANGELOG.md
3. Bump version in Cargo.toml
4. Create git tag
5. Push to trigger release CI

## Best Practices

1. **Always run pre-commit hooks** - They catch issues early
2. **Keep dependencies minimal** - Audit with cargo-machete
3. **Write tests first** - TDD helps catch API breaks
4. **Document breaking changes** - Update CHANGELOG.md
5. **Benchmark critical paths** - Use criterion for performance testing
6. **Review CI failures** - Don't ignore warnings
7. **Keep tools updated** - Regular cargo/tool updates

## Resources

- [Cargo Deny Documentation](https://embarkstudios.github.io/cargo-deny/)
- [Clippy Lints](https://rust-lang.github.io/rust-clippy/)
- [Rustfmt Configuration](https://rust-lang.github.io/rustfmt/)
- [Pre-commit Framework](https://pre-commit.com/)