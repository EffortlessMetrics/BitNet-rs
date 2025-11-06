# fix-locked.sh - Usage Guide

## Overview

The `fix-locked.sh` script automatically adds the `--locked` flag to cargo/cross commands in GitHub Actions workflow files. This ensures deterministic builds by using exact dependency versions from `Cargo.lock`.

## Quick Start

```bash
# Apply changes to all workflow files
scripts/fix-locked.sh .github/workflows/*.yml

# Preview changes without modifying files
scripts/fix-locked.sh --dry-run .github/workflows/*.yml

# Check if changes are needed (CI mode)
scripts/fix-locked.sh --check .github/workflows/*.yml
```

## Modes

### Apply Mode (Default)

Modifies files in-place, adding `--locked` where needed.

```bash
scripts/fix-locked.sh .github/workflows/*.yml
```

**Output:**
```
✓ Updated: .github/workflows/ci-core.yml
✓ Updated: .github/workflows/test.yml
✓ Applied --locked where missing
```

### Dry-Run Mode (`--dry-run` or `--preview`)

Shows what would be changed without modifying files. Useful for:
- Reviewing changes before applying
- Understanding the script's behavior
- Generating change reports

```bash
scripts/fix-locked.sh --dry-run .github/workflows/*.yml
```

**Output:**
```
Would update: .github/workflows/ci-core.yml
--- Diff ---
--- .github/workflows/ci-core.yml
+++ .github/workflows/ci-core.yml
@@ -75,7 +75,7 @@
       - name: Build core crates
         run: |
-          cargo build \
+          cargo build --locked \
             -p bitnet-common \
             --no-default-features --features cpu

⚠ Changes would be made (see diffs above)
```

### Check Mode (`--check`)

Exits with non-zero status if changes would be made. Ideal for CI/CD pipelines.

```bash
scripts/fix-locked.sh --check .github/workflows/*.yml
```

**Success (no changes needed):**
```
✓ All cargo commands have --locked flags
Exit code: 0
```

**Failure (changes needed):**
```
Changes needed in: .github/workflows/ci-core.yml
Changes needed in: .github/workflows/test.yml
❌ Some files are missing --locked flags
Run: scripts/fix-locked.sh .github/workflows/*.yml
Exit code: 1
```

## Behavior

### Commands Targeted

The script modifies these cargo/cross commands:
- `cargo build`
- `cargo test`
- `cargo run`
- `cargo bench`
- `cargo clippy`
- `cross build`
- `cross test`
- `cross run`
- `cross bench`
- `cross clippy`

### Insertion Rules

1. **Before `--` separator**: If command has ` -- `, insert `--locked` before it
   ```bash
   cargo test -- --nocapture
   # Becomes:
   cargo test --locked -- --nocapture
   ```

2. **At end of command**: Otherwise, append `--locked` at the end
   ```bash
   cargo build -p bitnet-common
   # Becomes:
   cargo build -p bitnet-common --locked
   ```

3. **Preserve comments**: Inline comments are preserved
   ```bash
   cargo build  # This builds the project
   # Becomes:
   cargo build --locked  # This builds the project
   ```

4. **Preserve backslashes**: Multi-line continuations work correctly
   ```bash
   cargo build \
     -p bitnet-common
   # Becomes:
   cargo build --locked \
     -p bitnet-common
   ```

5. **Skip if already present**: Commands with `--locked` are not modified (idempotent)

6. **Skip non-cargo commands**: Shell scripts, Python, etc. are ignored

### Edge Cases Handled

✅ **Multi-line commands with backslash continuation**
```yaml
run: |
  cargo test \
    -p bitnet-common \
    --features cpu
```

✅ **Commands with inline comments**
```yaml
run: cargo build  # Build the project
```

✅ **Complex `cargo run ... -- ...` patterns**
```yaml
run: cargo run -p xtask -- download-model
```

✅ **Already-locked commands (idempotency)**
```yaml
run: cargo test --locked -p bitnet-common
```

✅ **Non-cargo commands (ignored)**
```yaml
run: |
  echo "Building with cargo"
  ./scripts/build.sh
```

## CI Integration

### GitHub Actions

Add to `.github/workflows/guards.yml` or a dedicated workflow:

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
```

This ensures PRs adding/modifying workflows include `--locked` flags.

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Check workflow files for missing --locked flags

changed_workflows=$(git diff --cached --name-only --diff-filter=ACM | grep '^.github/workflows/.*\.yml$')

if [ -n "$changed_workflows" ]; then
  echo "Checking workflow files for --locked flags..."
  if ! scripts/fix-locked.sh --check $changed_workflows; then
    echo ""
    echo "❌ Some workflow files are missing --locked flags"
    echo "Run: scripts/fix-locked.sh $changed_workflows"
    echo ""
    exit 1
  fi
fi
```

## Testing

### Run Test Suite

```bash
scripts/tests/test-fix-locked.sh
```

This runs 16 tests covering:
- 8 fixture scenarios (various command patterns)
- 8 functional tests (idempotency, modes, error handling)

See `scripts/tests/README.md` for detailed test documentation.

### Manual Testing Workflow

1. **Preview changes**:
   ```bash
   scripts/fix-locked.sh --dry-run .github/workflows/ci-core.yml
   ```

2. **Review diff output** to ensure changes are correct

3. **Apply changes**:
   ```bash
   scripts/fix-locked.sh .github/workflows/ci-core.yml
   ```

4. **Verify idempotency** (running again should make no changes):
   ```bash
   scripts/fix-locked.sh --check .github/workflows/ci-core.yml
   ```

## Examples

### Example 1: Fixing a new workflow

You've just created `.github/workflows/new-feature.yml`:

```yaml
name: New Feature
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Build
        run: cargo build -p new-feature
      - name: Test
        run: cargo test -p new-feature -- --nocapture
```

**Preview changes:**
```bash
scripts/fix-locked.sh --dry-run .github/workflows/new-feature.yml
```

**Apply:**
```bash
scripts/fix-locked.sh .github/workflows/new-feature.yml
```

**Result:**
```yaml
name: New Feature
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Build
        run: cargo build -p new-feature --locked
      - name: Test
        run: cargo test -p new-feature --locked -- --nocapture
```

### Example 2: Batch processing

Fix all workflow files at once:

```bash
# Preview all changes
scripts/fix-locked.sh --dry-run .github/workflows/*.yml > /tmp/changes.diff

# Review the diff
less /tmp/changes.diff

# Apply if satisfied
scripts/fix-locked.sh .github/workflows/*.yml
```

### Example 3: CI enforcement

In your CI pipeline:

```bash
# Fail build if any workflows are missing --locked
scripts/fix-locked.sh --check .github/workflows/*.yml

if [ $? -ne 0 ]; then
  echo "Fix with: scripts/fix-locked.sh .github/workflows/*.yml"
  exit 1
fi
```

## Troubleshooting

### Issue: Script not executable

```bash
chmod +x scripts/fix-locked.sh
```

### Issue: No changes detected but --locked is missing

Check if:
1. The command is actually `cargo` or `cross` (not in a comment/string)
2. The command is one of: build, test, run, bench, clippy
3. The `--locked` flag isn't already present elsewhere in the line

Debug with dry-run:
```bash
scripts/fix-locked.sh --dry-run your-file.yml
```

### Issue: Unexpected modifications

The script uses text-based pattern matching, not full YAML parsing. Complex multi-line strings or unusual formatting may cause issues.

For edge cases:
1. Review with `--dry-run` first
2. Manually fix if needed
3. Consider adding a test case to `scripts/tests/fixtures/`

## Development

### Adding Test Cases

To add new test fixtures:

1. Create input: `scripts/tests/fixtures/XX-name.yml`
2. Create expected: `scripts/tests/fixtures/XX-name.expected.yml`
3. Run: `scripts/tests/test-fix-locked.sh`

The test harness auto-discovers new fixtures.

### Modifying the Script

After changes:

1. Run test suite: `scripts/tests/test-fix-locked.sh`
2. Test against real workflows: `scripts/fix-locked.sh --dry-run .github/workflows/*.yml`
3. Verify no regressions

## References

- Test Suite Documentation: `scripts/tests/README.md`
- Project Conventions: `CLAUDE.md` (see "Repository Contracts")
- GitHub Actions Docs: https://docs.github.com/en/actions
