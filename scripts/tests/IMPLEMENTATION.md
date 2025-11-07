# fix-locked.sh Hardening - Implementation Summary

## Overview

This document describes the hardening improvements made to `scripts/fix-locked.sh`, including comprehensive test coverage, new operational modes, and CI integration support.

## Changes Implemented

### 1. Enhanced Script Features

#### New Modes

**Dry-Run Mode (`--dry-run` / `--preview`)**
- Shows what would be changed without modifying files
- Displays unified diffs for review
- Safe for exploratory testing

**Check Mode (`--check`)**
- Exits with status 0 if no changes needed
- Exits with status 1 if changes would be made
- Perfect for CI/CD pipelines
- Shows which files need updating

**Apply Mode (default)**
- Original behavior: modifies files in-place
- Adds `--locked` flags where missing

#### Improved AWK Processing

Enhanced the AWK script to handle:
- **Trailing backslashes**: Preserves line continuations in multi-line commands
- **Inline comments**: Comments remain at end of line after `--locked` insertion
- **Comment-only lines**: Skips lines starting with `#` (don't modify)
- **Double-dash separator**: Inserts `--locked` before ` -- ` when present
- **Trailing whitespace**: Properly handles whitespace before comments/backslashes

### 2. Comprehensive Test Suite

#### Test Fixtures (`scripts/tests/fixtures/`)

Created 8 test fixture pairs (16 files total):

| Fixture | Purpose |
|---------|---------|
| 01-simple | Basic single-line cargo commands |
| 02-multiline | Multi-line commands with `\` continuation |
| 03-with-comments | Commands with inline comments |
| 04-double-dash | Commands with ` -- ` separator |
| 05-already-locked | Idempotency test (already has --locked) |
| 06-cross-tool | Cross tool commands (treated like cargo) |
| 07-cargo-run-with-args | Complex `cargo run ... -- ...` patterns |
| 08-non-cargo | Non-cargo commands (should be ignored) |

Each fixture has:
- `.yml` - Input file (without --locked)
- `.expected.yml` - Expected output (with --locked added correctly)

#### Test Harness (`scripts/tests/test-fix-locked.sh`)

Comprehensive test harness with 16 tests:

**Fixture Tests (8 tests)**
- Runs script on each fixture
- Compares output with expected result
- Reports differences with unified diffs

**Functional Tests (8 tests)**
1. Script exists and is executable
2. Fixtures directory exists
3. Idempotency (running twice produces same result)
4. Dry-run mode doesn't modify files
5. Check mode exits non-zero when changes needed
6. Check mode exits zero when no changes needed
7. Handles non-existent files gracefully
8. Shows usage message with no arguments

**Output Format**
- Color-coded results (green ✓, red ✗, yellow ⊘)
- Detailed diffs for failures
- Summary statistics

### 3. Documentation

Created three documentation files:

**scripts/tests/README.md**
- Test suite overview
- Fixture descriptions
- Test harness details
- Instructions for adding new tests
- CI integration examples

**scripts/tests/USAGE.md**
- Complete user guide
- Mode descriptions with examples
- Behavior documentation
- CI integration patterns
- Troubleshooting guide
- Real-world examples

**scripts/tests/IMPLEMENTATION.md** (this file)
- Implementation summary
- Change log
- Technical details
- Validation results

## Technical Details

### AWK Processing Flow

1. **Line Analysis**
   - Check if line contains cargo/cross command
   - Check if line already has `--locked`
   - Check if line is a comment (starts with `#`)

2. **Element Extraction**
   - Extract trailing comment (if present)
   - Extract trailing backslash (if present)
   - Remove extracted elements from working line

3. **Flag Insertion**
   - If line has ` -- ` separator: insert `--locked` before it
   - Otherwise: append `--locked` at end
   - Remove trailing whitespace before insertion

4. **Reconstruction**
   - Append backslash (if was present)
   - Append comment (if was present)
   - Output reconstructed line

### File Processing

For each file:
1. Create temporary file
2. Process with AWK
3. Compare with original using `diff -q`
4. Based on mode:
   - **apply**: Replace original with modified
   - **dry-run**: Show diff, delete temp
   - **check**: Check if different, delete temp

## Validation Results

### Test Suite Results

```
========================================
Testing fix-locked.sh
========================================

✓ PASS: Script exists and is executable
✓ PASS: Fixtures directory exists
  Found 8 test fixtures

--- Fixture Tests ---
✓ PASS: Fixture: 01-simple
✓ PASS: Fixture: 02-multiline
✓ PASS: Fixture: 03-with-comments
✓ PASS: Fixture: 04-double-dash
✓ PASS: Fixture: 05-already-locked
✓ PASS: Fixture: 06-cross-tool
✓ PASS: Fixture: 07-cargo-run-with-args
✓ PASS: Fixture: 08-non-cargo

--- Functional Tests ---
✓ PASS: Idempotency test
✓ PASS: Dry-run mode (no modifications)
✓ PASS: Check mode (changes needed)
✓ PASS: Check mode (no changes needed)
✓ PASS: Handles non-existent files
✓ PASS: Shows usage message with no args

========================================
Test Summary
========================================
Tests run:    16
Tests passed: 16
Tests failed: 0

✓ All tests passed!
```

### Mode Validation

**Dry-Run Mode**
```bash
$ scripts/fix-locked.sh --dry-run scripts/tests/fixtures/01-simple.yml
Would update: scripts/tests/fixtures/01-simple.yml
--- Diff ---
[Shows unified diff]
⚠ Changes would be made (see diffs above)
```
✅ Files not modified, diff displayed correctly

**Check Mode (Clean File)**
```bash
$ scripts/fix-locked.sh --check scripts/tests/fixtures/05-already-locked.expected.yml
✓ All cargo commands have --locked flags
$ echo $?
0
```
✅ Exits with 0 when no changes needed

**Check Mode (Dirty File)**
```bash
$ scripts/fix-locked.sh --check scripts/tests/fixtures/01-simple.yml
Changes needed in: scripts/tests/fixtures/01-simple.yml
❌ Some files are missing --locked flags
Run: scripts/fix-locked.sh .github/workflows/*.yml
$ echo $?
1
```
✅ Exits with 1 when changes needed

## CI Integration Recommendations

### Option 1: Dedicated Workflow Guard

Create `.github/workflows/workflow-guards.yml`:

```yaml
name: Workflow Guards

on:
  pull_request:
    paths:
      - '.github/workflows/**'

jobs:
  verify-locked-flags:
    name: Verify --locked flags
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check --locked flags in workflows
        run: scripts/fix-locked.sh --check .github/workflows/*.yml

      - name: Run fix-locked test suite
        run: scripts/tests/test-fix-locked.sh
```

### Option 2: Add to Existing Guards Workflow

Add to `.github/workflows/guards.yml`:

```yaml
  verify-locked-flags:
    name: Verify --locked flags
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check --locked flags
        run: scripts/fix-locked.sh --check .github/workflows/*.yml
```

### Option 3: Pre-commit Hook

Install as pre-commit hook to catch issues before commit:

```bash
# Add to .git/hooks/pre-commit
changed_workflows=$(git diff --cached --name-only | grep '^.github/workflows/.*\.yml$')
if [ -n "$changed_workflows" ]; then
  scripts/fix-locked.sh --check $changed_workflows || exit 1
fi
```

## Usage Examples

### Daily Development

```bash
# After modifying a workflow, preview changes
scripts/fix-locked.sh --dry-run .github/workflows/ci-core.yml

# Apply if satisfied
scripts/fix-locked.sh .github/workflows/ci-core.yml

# Verify idempotency
scripts/fix-locked.sh --check .github/workflows/ci-core.yml
```

### Batch Operations

```bash
# Fix all workflows at once
scripts/fix-locked.sh .github/workflows/*.yml

# Preview all changes first
scripts/fix-locked.sh --dry-run .github/workflows/*.yml > /tmp/changes.diff
less /tmp/changes.diff
```

### CI Pipeline

```bash
# In CI, fail if any workflow is missing --locked
scripts/fix-locked.sh --check .github/workflows/*.yml || {
  echo "Some workflows need --locked flags"
  echo "Run locally: scripts/fix-locked.sh .github/workflows/*.yml"
  exit 1
}
```

## Edge Cases Handled

✅ Multi-line commands with backslash continuation
✅ Commands with inline comments
✅ Commands with `--` separator (e.g., `cargo test -- --nocapture`)
✅ Already-locked commands (idempotent)
✅ Cross tool commands (treated like cargo)
✅ Complex `cargo run -p pkg -- args` patterns
✅ Non-cargo commands (ignored)
✅ Comment-only lines (skipped)
✅ Empty lines (preserved)
✅ YAML indentation (preserved)

## Known Limitations

1. **Text-based processing**: Uses AWK pattern matching, not full YAML parsing
   - Complex multi-line YAML strings may have edge cases
   - Unusual formatting might not be handled perfectly

2. **Comment detection**: Simple regex for inline comments
   - Assumes `#` starts a comment (correct for YAML)
   - Quoted strings with `#` might have edge cases

3. **Line-oriented**: Processes line by line
   - Cannot handle commands split across multiple lines without backslash

For the vast majority of GitHub Actions workflows, these limitations are not an issue.

## Files Created/Modified

### Created Files

```
scripts/tests/
├── fixtures/
│   ├── 01-simple.yml
│   ├── 01-simple.expected.yml
│   ├── 02-multiline.yml
│   ├── 02-multiline.expected.yml
│   ├── 03-with-comments.yml
│   ├── 03-with-comments.expected.yml
│   ├── 04-double-dash.yml
│   ├── 04-double-dash.expected.yml
│   ├── 05-already-locked.yml
│   ├── 05-already-locked.expected.yml
│   ├── 06-cross-tool.yml
│   ├── 06-cross-tool.expected.yml
│   ├── 07-cargo-run-with-args.yml
│   ├── 07-cargo-run-with-args.expected.yml
│   ├── 08-non-cargo.yml
│   └── 08-non-cargo.expected.yml
├── test-fix-locked.sh           # Test harness
├── README.md                     # Test suite documentation
├── USAGE.md                      # User guide
└── IMPLEMENTATION.md             # This file
```

### Modified Files

```
scripts/fix-locked.sh             # Enhanced with --dry-run and --check modes
```

## Running the Tests

```bash
# Run full test suite
scripts/tests/test-fix-locked.sh

# Manual testing
scripts/fix-locked.sh --dry-run scripts/tests/fixtures/*.yml
scripts/fix-locked.sh --check scripts/tests/fixtures/05-already-locked.expected.yml
```

## Success Criteria

All criteria met:

✅ Test suite with sample YAML files testing:
   - Multi-line cargo commands
   - Commands with comments
   - Commands with `cargo run ... -- ...` patterns
   - Already-correct files (idempotency)

✅ --dry-run / --check modes:
   - Shows what would be changed
   - Exits with non-zero if changes would be made
   - Doesn't modify files

✅ Test harness script:
   - Runs fix-locked.sh against test fixtures
   - Validates expected output
   - Can be run in CI
   - Color-coded output with statistics

✅ Documentation:
   - Test fixture files with inputs and expected outputs
   - Test harness with comprehensive test coverage
   - User guide with examples
   - Implementation summary

## Next Steps

To integrate into CI:

1. Choose integration approach (dedicated workflow or add to existing)
2. Add CI job using `--check` mode
3. Optionally add test suite run: `scripts/tests/test-fix-locked.sh`
4. Document in project CI guidelines

Example CI addition:

```yaml
- name: Verify --locked flags in workflows
  run: scripts/fix-locked.sh --check .github/workflows/*.yml

- name: Run fix-locked test suite
  run: scripts/tests/test-fix-locked.sh
```
