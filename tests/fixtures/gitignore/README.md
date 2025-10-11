# Gitignore Test Fixtures (Issue #439 AC8)

## Purpose

Validate that `.gitignore` includes proper patterns for proptest regression files and incremental test cache to prevent accidental commits.

## Fixture Files

### Valid Pattern

1. **`.gitignore.valid`**
   - ✓ Includes `**/*.proptest-regressions` pattern
   - ✓ Includes `tests/tests/cache/incremental/last_run.json`
   - ✓ Comprehensive ignore patterns
   - Usage: Reference `.gitignore` for Issue #439 AC8

### Invalid Pattern (Missing Critical Entries)

2. **`.gitignore.missing`**
   - ✗ Missing `**/*.proptest-regressions` pattern
   - ✗ Incomplete test artifact coverage
   - Problem: Proptest regression files may be committed accidentally
   - Tests: AC8 validation should detect missing patterns

## Testing Usage

### Validate Gitignore Contents
```rust
use std::fs;

#[test]
fn validate_gitignore_has_proptest_pattern() {
    let gitignore = fs::read_to_string(
        "tests/fixtures/gitignore/.gitignore.valid"
    ).unwrap();

    assert!(
        gitignore.contains("**/*.proptest-regressions"),
        "Gitignore must include proptest regression pattern"
    );
}
```

### Check for Missing Patterns
```bash
# Search for proptest pattern in .gitignore
rg "proptest-regressions" .gitignore

# Verify incremental cache pattern
rg "incremental/last_run.json" .gitignore
```

## Integration with Tests

These fixtures are consumed by:
- Workspace-level integration tests
- CI/CD gitignore validation

## Specification Reference

- **Issue**: #439 GPU feature-gate hardening
- **Acceptance Criteria**: AC8 - Gitignore updates
- **Specification**: `docs/explanation/issue-439-spec.md#ac8-repository-hygiene`

## Critical Patterns

### Proptest Regressions
```gitignore
# Proptest regression files (deterministic testing)
**/*.proptest-regressions
```

**Why important**: Proptest generates regression files during test failures. These should not be committed as they're test-specific and may cause flakiness.

### Incremental Test Cache
```gitignore
# Incremental test cache (should not be committed)
tests/tests/cache/incremental/last_run.json
```

**Why important**: Test cache files are machine-specific and should not be in version control.

## Common Gitignore Mistakes

1. **Missing proptest pattern**: Forget `**/*.proptest-regressions`
2. **Wrong proptest glob**: Using `*.proptest-regressions` instead of `**/*.proptest-regressions`
3. **Missing incremental cache**: Forget `last_run.json` pattern
4. **Too specific paths**: Using absolute paths instead of patterns

## Validation Checklist

.gitignore should include:
- [ ] `**/*.proptest-regressions` (wildcard for all directories)
- [ ] `tests/tests/cache/incremental/last_run.json`
- [ ] `target/` (Rust build artifacts)
- [ ] IDE-specific files (`.vscode/`, `.idea/`)
- [ ] OS-specific files (`.DS_Store`, `Thumbs.db`)

## Example Validation Test

```rust
#[test]
fn ac8_gitignore_includes_required_patterns() {
    let gitignore = std::fs::read_to_string(".gitignore")
        .expect("Failed to read .gitignore");

    // Check for proptest regression pattern
    assert!(
        gitignore.contains("**/*.proptest-regressions")
            || gitignore.contains("*.proptest-regressions"),
        "AC:8 FAIL - .gitignore missing proptest-regressions pattern"
    );

    // Check for incremental cache pattern
    assert!(
        gitignore.contains("incremental/last_run.json")
            || gitignore.contains("last_run.json"),
        "AC:8 FAIL - .gitignore missing incremental cache pattern"
    );

    println!("AC:8 PASS - .gitignore includes required test artifact patterns");
}
```
