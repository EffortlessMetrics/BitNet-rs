# Codemod Guide for Unit Constants

## Quick Start

```bash
# Preview what will change
./scripts/codemod-units.sh --preview

# Apply changes with per-class commits
./scripts/codemod-units.sh --apply

# Verify guards still work
./scripts/check-units.sh
./scripts/check-envlock.sh
```

## Manual One-Liners

For surgical fixes without the full script:

```bash
# Fix specific file's MB literals
perl -0777 -pe 's/\b1024\s*\*\s*1024\b/BYTES_PER_MB/g' -i tests/specific_file.rs

# Fix N * MB patterns
perl -0777 -pe 's/\b(\d+)\s*\*\s*1024\s*\*\s*1024\b/$1 * BYTES_PER_MB/g' -i tests/specific_file.rs

# Fix decimal constants
perl -0777 -pe 's/\b(1_048_576|1048576)\b/BYTES_PER_MB/g' -i tests/specific_file.rs
perl -0777 -pe 's/\b(1_073_741_824|1073741824)\b/BYTES_PER_GB/g' -i tests/specific_file.rs
```

## Type Casting Solutions

When you hit type mismatches after conversion:

### Option 1: Cast at use site
```rust
let size: usize = 10 * BYTES_PER_MB as usize;
```

### Option 2: Module-local aliases
```rust
// At top of test module
const MB: usize = bitnet_tests::common::BYTES_PER_MB as usize;
const GB: usize = bitnet_tests::common::BYTES_PER_GB as usize;

// Then use naturally
let allocation = vec![0u8; 10 * MB];
```

## Git Workflow

```bash
# Create feature branch
git checkout -b chore/codemod-units-$(date +%Y%m%d-%H%M)

# Run codemod with auto-commits
./scripts/codemod-units.sh --apply

# Review commit history
git log --oneline -10

# If something went wrong with a specific class
git revert HEAD~2  # Revert specific commit

# Push when ready
git push -u origin HEAD
```

## Pattern Reference

| Pattern | Example | Replacement |
|---------|---------|-------------|
| MB literal | `1024 * 1024` | `BYTES_PER_MB` |
| Scaled MB | `10 * 1024 * 1024` | `10 * BYTES_PER_MB` |
| MB decimal | `1048576` or `1_048_576` | `BYTES_PER_MB` |
| GB literal | `1024 * 1024 * 1024` | `BYTES_PER_GB` |
| Scaled GB | `5 * 1024 * 1024 * 1024` | `5 * BYTES_PER_GB` |
| GB decimal | `1073741824` or `1_073_741_824` | `BYTES_PER_GB` |
| Bit-shift MB | `1 << 20` | `BYTES_PER_MB` (manual review) |
| Bit-shift GB | `1 << 30` | `BYTES_PER_GB` (manual review) |

## Verification Commands

After running codemod:

```bash
# Check no raw conversions remain
./scripts/check-units.sh

# Verify tests still pass
cargo test -p bitnet --test test_configuration_scenarios
cargo test -p bitnet --features fixtures --test test_configuration_scenarios

# Check for type errors
cargo check --workspace --tests
```

## Rollback

If you need to undo:

```bash
# Undo last commit
git reset --hard HEAD~1

# Undo all codemod commits (if on feature branch)
git reset --hard origin/main

# Cherry-pick only good commits
git cherry-pick <sha1> <sha2> ...
```