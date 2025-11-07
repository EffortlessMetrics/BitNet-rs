# Test Suite for fix-locked.sh

Comprehensive test suite for validating the `fix-locked.sh` script behavior.

## Quick Start

```bash
# Run all tests
scripts/tests/test-fix-locked.sh

# Test the script manually with dry-run mode
scripts/fix-locked.sh --dry-run scripts/tests/fixtures/*.yml

# Test check mode (CI usage)
scripts/fix-locked.sh --check scripts/tests/fixtures/05-already-locked.expected.yml
```

## Test Structure

### Fixtures (`fixtures/`)

Test fixtures are organized as pairs:
- `XX-name.yml` - Input file (without --locked flags)
- `XX-name.expected.yml` - Expected output (with --locked flags added)

Each fixture tests a specific scenario:

1. **01-simple.yml** - Basic single-line cargo commands
   - `cargo build`, `cargo test`, `cargo run`, etc.

2. **02-multiline.yml** - Multi-line commands with backslash continuation
   - Tests that `--locked` is added on the first line

3. **03-with-comments.yml** - Commands with inline comments
   - Ensures comments are preserved

4. **04-double-dash.yml** - Commands with `--` separator
   - Tests `--locked` insertion before the `--` separator
   - Examples: `cargo test -- --nocapture`

5. **05-already-locked.yml** - Idempotency test
   - Files already containing `--locked` should remain unchanged

6. **06-cross-tool.yml** - Cross tool commands
   - Tests `cross build`, `cross test` (same behavior as cargo)

7. **07-cargo-run-with-args.yml** - Complex `cargo run ... -- ...` patterns
   - Tests `cargo run -p xtask -- download-model`
   - Ensures `--locked` goes before `--` separator

8. **08-non-cargo.yml** - Non-cargo commands
   - Shell scripts, Python, echo statements with "cargo" in strings
   - Should NOT be modified

## Test Harness (`test-fix-locked.sh`)

The test harness validates:

### Fixture Tests
- Runs `fix-locked.sh` on each fixture
- Compares output with expected results
- Reports differences with unified diffs

### Functional Tests
- **Idempotency**: Running twice produces identical results
- **Dry-run mode**: Files are not modified with `--dry-run`
- **Check mode (dirty)**: Exits non-zero when changes needed
- **Check mode (clean)**: Exits zero when no changes needed
- **Error handling**: Gracefully handles non-existent files
- **Usage message**: Shows help when run without arguments

### Output Format

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

## Adding New Tests

To add a new test case:

1. Create input fixture: `scripts/tests/fixtures/XX-name.yml`
2. Create expected output: `scripts/tests/fixtures/XX-name.expected.yml`
3. Run test harness to validate: `scripts/tests/test-fix-locked.sh`

The test harness will automatically discover and run the new fixture.

## CI Integration

Add to `.github/workflows/guards.yml` or similar:

```yaml
- name: Verify --locked flags in workflows
  run: scripts/fix-locked.sh --check .github/workflows/*.yml

- name: Run fix-locked.sh test suite
  run: scripts/tests/test-fix-locked.sh
```

This ensures:
- All workflow files have `--locked` flags (via `--check` mode)
- The `fix-locked.sh` script itself is tested

## Manual Testing

```bash
# Preview changes without modifying files
scripts/fix-locked.sh --dry-run .github/workflows/*.yml

# Apply changes
scripts/fix-locked.sh .github/workflows/*.yml

# Verify no changes needed (CI mode)
scripts/fix-locked.sh --check .github/workflows/*.yml && echo "All good!"
```

## Edge Cases Tested

- ✅ Single-line cargo commands
- ✅ Multi-line commands with `\` continuation
- ✅ Commands with inline comments (`# comment`)
- ✅ Commands with `--` separator (`cargo test -- --nocapture`)
- ✅ Already-locked commands (idempotency)
- ✅ Cross tool commands (`cross build`)
- ✅ Complex `cargo run ... -- ...` patterns
- ✅ Non-cargo commands (should be ignored)
- ✅ Files with mixed cargo/non-cargo commands

## Known Limitations

The script uses AWK pattern matching and has these known limitations:

1. **Line-based processing**: Multi-line YAML strings with unusual formatting might not be handled correctly
2. **Comment preservation**: Inline comments are preserved, but complex comment patterns may have edge cases
3. **No YAML parsing**: The script does text-based pattern matching, not full YAML parsing

For most GitHub Actions workflows, these limitations are not an issue.
