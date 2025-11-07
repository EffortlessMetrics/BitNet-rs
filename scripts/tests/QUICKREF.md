# fix-locked.sh - Quick Reference

## One-Liners

```bash
# Apply changes to all workflows
scripts/fix-locked.sh .github/workflows/*.yml

# Preview changes
scripts/fix-locked.sh --dry-run .github/workflows/*.yml

# Check if changes needed (CI)
scripts/fix-locked.sh --check .github/workflows/*.yml

# Run test suite
scripts/tests/test-fix-locked.sh
```

## Modes

| Mode | Flag | Exit Code | Side Effects | Use Case |
|------|------|-----------|--------------|----------|
| Apply | (none) | 0 | Modifies files | Apply fixes |
| Dry-run | `--dry-run`, `--preview` | 0 | None | Preview changes |
| Check | `--check` | 0=clean, 1=dirty | None | CI validation |

## What Gets Modified

✅ `cargo build`, `cargo test`, `cargo run`, `cargo bench`, `cargo clippy`
✅ `cross build`, `cross test`, `cross run`, `cross bench`, `cross clippy`
❌ Other commands (shell scripts, Python, etc.)

## Insertion Rules

```bash
# Rule 1: Before " -- " separator
cargo test -- --nocapture
→ cargo test --locked -- --nocapture

# Rule 2: At end of command
cargo build -p pkg
→ cargo build -p pkg --locked

# Rule 3: Preserve comments
cargo build  # comment
→ cargo build --locked  # comment

# Rule 4: Preserve backslash continuations
cargo build \
  -p pkg
→ cargo build --locked \
  -p pkg

# Rule 5: Skip if already present (idempotent)
cargo build --locked
→ cargo build --locked
```

## Test Fixtures

| Fixture | Tests |
|---------|-------|
| 01-simple | Basic single-line commands |
| 02-multiline | Multi-line with `\` continuation |
| 03-with-comments | Inline comments |
| 04-double-dash | `cargo test -- args` |
| 05-already-locked | Idempotency |
| 06-cross-tool | Cross commands |
| 07-cargo-run-with-args | `cargo run -p pkg -- args` |
| 08-non-cargo | Non-cargo (ignored) |

## CI Integration

```yaml
# Option 1: Add to existing workflow
- name: Check --locked flags
  run: scripts/fix-locked.sh --check .github/workflows/*.yml

# Option 2: Add test suite
- name: Test fix-locked.sh
  run: scripts/tests/test-fix-locked.sh

# Option 3: Both
- name: Verify workflows have --locked
  run: |
    scripts/fix-locked.sh --check .github/workflows/*.yml
    scripts/tests/test-fix-locked.sh
```

## Common Workflows

### Fix a Single Workflow

```bash
# 1. Preview
scripts/fix-locked.sh --dry-run .github/workflows/ci.yml

# 2. Review diff output

# 3. Apply
scripts/fix-locked.sh .github/workflows/ci.yml

# 4. Verify
scripts/fix-locked.sh --check .github/workflows/ci.yml
```

### Batch Fix All Workflows

```bash
# Preview all
scripts/fix-locked.sh --dry-run .github/workflows/*.yml > /tmp/changes.diff
less /tmp/changes.diff

# Apply all
scripts/fix-locked.sh .github/workflows/*.yml
```

### Develop New Test Case

```bash
# 1. Create fixtures
editor scripts/tests/fixtures/09-my-test.yml
editor scripts/tests/fixtures/09-my-test.expected.yml

# 2. Run tests
scripts/tests/test-fix-locked.sh

# 3. Debug if needed
scripts/fix-locked.sh --dry-run scripts/tests/fixtures/09-my-test.yml
```

## Test Suite Results

```
========================================
Test Summary
========================================
Tests run:    16
Tests passed: 16  ← All passing!
Tests failed: 0

✓ All tests passed!
```

## Exit Codes

| Code | Mode | Meaning |
|------|------|---------|
| 0 | apply | Success |
| 0 | dry-run | Success (changes shown) |
| 0 | check | No changes needed |
| 1 | check | Changes needed |
| 1 | any | Script error or usage error |

## File Locations

```
scripts/
├── fix-locked.sh              # Main script
└── tests/
    ├── fixtures/              # 16 test YAML files
    ├── test-fix-locked.sh     # Test harness (executable)
    ├── README.md              # Test suite docs
    ├── USAGE.md               # User guide
    ├── IMPLEMENTATION.md      # Implementation summary
    └── QUICKREF.md            # This file
```

## Line Counts

```
Scripts:
  fix-locked.sh:         142 lines
  test-fix-locked.sh:    268 lines

Documentation:
  README.md:             170 lines
  USAGE.md:              379 lines
  IMPLEMENTATION.md:     425 lines
  QUICKREF.md:            ~150 lines
```

## Documentation

- **User Guide**: `scripts/tests/USAGE.md` - Complete reference with examples
- **Test Docs**: `scripts/tests/README.md` - Test suite details
- **Implementation**: `scripts/tests/IMPLEMENTATION.md` - Technical details
- **Quick Ref**: `scripts/tests/QUICKREF.md` - This file

## Help

```bash
# Show usage
scripts/fix-locked.sh

# Output:
Usage: scripts/fix-locked.sh [--dry-run|--check] <file1> [file2 ...]

Modes:
  (default)  Apply changes in-place
  --dry-run  Show what would be changed (no modifications)
  --check    Exit with non-zero if changes would be made (CI mode)
```

## Examples from Tests

See `scripts/tests/fixtures/*.yml` for 8 real examples covering:
- ✅ Simple commands
- ✅ Multi-line continuations
- ✅ Comments
- ✅ Double-dash separators
- ✅ Already-locked (idempotency)
- ✅ Cross tool
- ✅ Cargo run with args
- ✅ Non-cargo commands

Each fixture has:
- `.yml` - Input (without --locked)
- `.expected.yml` - Expected output (with --locked)
