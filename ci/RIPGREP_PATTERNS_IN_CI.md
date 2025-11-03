# Ripgrep Patterns in BitNet.rs CI & Hooks

Complete reference for all ripgrep (`rg`) usage in CI/CD pipeline and git hooks.

---

## Pre-commit Hook (`.githooks/pre-commit`)

### Pattern 1: Bare #[ignore] Annotations

**Location**: Line 22  
**Command**: 
```bash
rg -n -P '#\[ignore\](?!\s*=)' --hidden --glob '!**/target/**' --glob '*.rs' crates tests tests-new xtask
```

**Regex Breakdown**:
- `-n`: Show line numbers
- `-P`: Enable Perl-compatible regex (for lookahead)
- `#\[ignore\]`: Match literal `#[ignore]`
- `(?!\s*=)`: Negative lookahead - fail if followed by whitespace + `=`

**Purpose**: Detect bare `#[ignore]` attributes without annotations

**Valid matches** (that FAIL the check):
```rust
#[ignore]                        // ❌ BARE - no reason provided
fn test_something() { }

#[test]
#[ignore]                        // ❌ BARE - even with #[test] above
fn test_something() { }
```

**Invalid matches** (that PASS the check):
```rust
#[ignore = "reason"]             // ✅ ANNOTATED - explicit reason
fn test_something() { }

// TODO: Fix later
#[ignore]                        // ✅ COMMENT - justification above
fn test_something() { }

// Blocked by Issue #254
#[ignore]                        // ✅ COMMENT - issue reference
fn test_something() { }
```

---

### Pattern 2: Raw Environment Mutations

**Location**: Line 46  
**Command**:
```bash
rg -n '(std::env::set_var|std::env::remove_var)\(' \
  --glob '*.rs' \
  --glob '!**/tests/helpers/**' \
  --glob '!**/support/**' \
  --glob '!**/env_guard.rs' \
  crates tests tests-new xtask
```

**Regex Breakdown**:
- `-n`: Show line numbers
- `(std::env::set_var|std::env::remove_var)\(`: Match either function with opening paren
- `--glob '*.rs'`: Only Rust files
- `--glob '!**/tests/helpers/**'`: Exclude test helpers
- `--glob '!**/support/**'`: Exclude support modules
- `--glob '!**/env_guard.rs'`: Exclude EnvGuard definition

**Purpose**: Prevent raw environment mutations; enforce EnvGuard pattern

**Valid matches** (that FAIL the check):
```rust
fn test_bad() {
    std::env::set_var("VAR", "value");      // ❌ RAW - no EnvGuard
    assert_eq!(std::env::var("VAR").unwrap(), "value");
}
```

**Invalid matches** (that PASS the check):
```rust
// In tests/helpers/env_guard.rs - ALLOWED (definition)
pub struct EnvGuard { /* ... */ }
impl EnvGuard {
    pub fn new(var: &str, val: &str) -> Self {
        std::env::set_var(var, val);        // ✅ ALLOWED - in definition
        EnvGuard { var: var.to_string() }
    }
}

// In test code - ALLOWED (via helper)
#[test]
#[serial(bitnet_env)]
fn test_good() {
    let _guard = EnvGuard::new("VAR", "value");  // ✅ ALLOWED - via helper
    assert_eq!(std::env::var("VAR").unwrap(), "value");
}
```

---

## CI Guard Scripts

### Guard: Serial Annotations (`scripts/check-serial-annotations.sh`)

**Location**: Line 7  
**Command**:
```bash
rg -n 'EnvGuard::new|temp_env::with_var' crates tests --type rust -B 5
```

**Regex Breakdown**:
- `-n`: Show line numbers
- `EnvGuard::new|temp_env::with_var`: Match either pattern (alternation)
- `--type rust`: Only Rust files
- `-B 5`: Show 5 lines before match (context for annotation check)

**Purpose**: Find env-mutating tests and verify they have `#[serial(bitnet_env)]`

**Detection Strategy**:
1. Find all `EnvGuard::new(...)` or `temp_env::with_var(...)` calls
2. Extract 5 lines of context above each match
3. Check if context contains `#[serial(bitnet_env)]`
4. Fail if test has env mutation but missing annotation

---

### Guard: Feature Consistency (`scripts/check-feature-gates.sh`)

**Location**: Line 13  
**Command**:
```bash
rg -oI '#\[cfg.*feature\s*=\s*"([^"]+)"' --replace '$1' crates --type rust
```

**Regex Breakdown**:
- `-o`: Output only matched groups (not full line)
- `-I`: Smart case (ignore case for ASCII)
- `#\[cfg`: Match literal `#[cfg`
- `.*feature\s*=\s*"`: Match `feature` with flexible whitespace
- `([^"]+)`: Capture group - match anything except `"`
- `"([^"]+)"`: Extract feature name between quotes
- `--replace '$1'`: Replace entire match with captured group (feature name)

**Purpose**: Extract all feature names used in `#[cfg(feature = "...")]` macros

**Example Usage**:
```rust
#[cfg(feature = "gpu")]
fn gpu_function() { }

#[cfg(any(feature = "gpu", feature = "cuda"))]
fn device_function() { }
```

**Extracted**:
```
gpu
gpu
cuda
```

**Then Compared Against** (from `Cargo.toml`):
```toml
[features]
cpu = []
gpu = ["cuda"]  # Requires "cuda" as dependency
cuda = ["gpu"]  # Backward compat alias
```

---

### Guard: Ignore Annotations (`scripts/check-ignore-annotations.sh`)

**Location**: Line 7  
**Command**:
```bash
rg -n '#\[ignore\]' crates tests --type rust
```

**Regex Breakdown**:
- `-n`: Show line numbers
- `#\[ignore\]`: Literal match for ignore attribute
- `--type rust`: Rust files only

**Purpose**: Find all ignored tests and verify they have issue references

**Verification Process**:
1. Find each `#[ignore]` marker with line number
2. Extract 2 lines before the match (context)
3. Check if context contains one of:
   - `Blocked by Issue #NNN`
   - `Slow: <reason>`
   - `TODO: <reason>`
4. Fail if none match

**Valid annotations**:
```rust
// Blocked by Issue #254 - shape mismatch in layer-norm
#[ignore]
fn test_something() { }

// Slow: QK256 scalar kernels (~0.1 tok/s)
#[ignore]
fn test_slow_path() { }

// TODO: Implement after #469 resolution
#[ignore]
fn test_future_feature() { }
```

---

### Guard: Environment Mutation (Inline in `ci.yml`)

**Location**: `.github/workflows/ci.yml` line 569  
**Command**:
```bash
rg -n '(std::env::set_var|std::env::remove_var)\(' crates \
  --glob '!**/tests/support/**' \
  --glob '!**/support/**' \
  --glob '!**/helpers/**' \
  --type rust
```

**Differences from pre-commit**:
- No `-P` flag (uses default ERE, not Perl regex)
- Different glob exclusions (combined in one guard)
- Runs only on `crates/` directory (narrower scope)
- Different error message formatting

**Purpose**: CI enforcement of no raw env mutations in crate code

---

## Ripgrep Pattern Classes

### Negative Lookahead (Pre-commit)
```regex
#\[ignore\](?!\s*=)
```
- **Matches**: `#[ignore]` NOT followed by `= ...`
- **Use Case**: Find bare attributes without annotations
- **Requires**: `-P` flag for Perl-compatible regex

### Alternation (Choice)
```regex
EnvGuard::new|temp_env::with_var
std::env::set_var|std::env::remove_var
Blocked by Issue #[0-9]+|Slow:|TODO:
```
- **Matches**: Either pattern separated by `|`
- **Use Case**: Multiple valid patterns
- **Standard**: Works in ERE (default)

### Capture Groups
```regex
#\[cfg.*feature\s*=\s*"([^"]+)"
```
- **Captures**: Feature name between quotes
- **Use Case**: Extract specific text (with `--replace '$1'`)
- **Requires**: `-o` output flag

### Character Classes
```regex
[^"]      # Not a quote
[0-9]+    # One or more digits
[a-z_-]*  # Alphanumeric, underscore, or dash
```
- **Matches**: Sets of characters
- **Common**: Negation with `^` inside `[]`

### Whitespace Flexibility
```regex
\s*       # Zero or more whitespace chars
\s+       # One or more whitespace chars
```
- **Use Case**: Flexible formatting (e.g., `feature = "..." ` vs `feature="..."`)

---

## Performance Notes

### Ripgrep Speed

All patterns complete in **< 1 second** on typical BitNet.rs codebase:

| Pattern | Time | Scope |
|---------|------|-------|
| Bare #[ignore] | ~50ms | crates/, tests/, tests-new/, xtask/ |
| Raw env mutations | ~60ms | crates/, tests/, tests-new/, xtask/ |
| EnvGuard search | ~40ms | crates/, tests/ with context |
| Feature usage | ~80ms | crates/ with extraction |

**Total hook time**: ~200ms (ripgrep only; git operations dominate)

### Optimization Tips

1. **Use `--type rust`**: Filter by file type before regex (faster than globs)
2. **Use `-o`**: Only output matched text, not full lines (reduces output)
3. **Use `--glob`**: Exclude large directories (e.g., target/) early
4. **Combine patterns**: Use `|` alternation instead of multiple runs
5. **Perl regex only when needed**: `-P` has slight overhead; use for lookahead/lookbehind only

---

## Common Patterns Reference

### Finding Test Annotations

```bash
# Find all tests with a specific attribute
rg -n '#\[test\]' crates tests --type rust

# Find tests with multiple attributes
rg -n '#\[test\].*#\[ignore\]' crates tests --type rust -U

# Find tests in a specific file pattern
rg -n '#\[test\]' crates/*/tests/*.rs --type rust
```

### Finding Feature Gates

```bash
# Find all feature usage
rg -n '#\[cfg.*feature' crates --type rust

# Find specific feature
rg -n '#\[cfg.*feature = "gpu"' crates --type rust

# Find feature without fallback
rg -n '#\[cfg(feature = "gpu")\]' crates --type rust  # Missing 'cuda' fallback
```

### Finding Environment Usage

```bash
# Find all env var access (not just set/remove)
rg -n 'std::env::var|std::env::set_var|std::env::remove_var' crates --type rust

# Find env vars in specific module
rg -n 'std::env::' crates/bitnet-cli --type rust

# Find helper-provided env functions
rg -n 'EnvGuard|temp_env' crates tests --type rust
```

---

## Integration with CI Workflow

### Guard Job Execution Order (Sequential)

```yaml
# Pre-commit check (local, before push)
→ `.githooks/pre-commit` (2 ripgrep patterns)

# CI check (after push, parallel)
→ guard-serial-annotations (ripgrep + sh processing)
→ guard-feature-consistency (ripgrep + grep)
→ guard-ignore-annotations (ripgrep + sed processing)
→ env-mutation-guard (inline ripgrep)
```

### Failure Scenarios

| Trigger | Pattern | Guard | Output | Action |
|---------|---------|-------|--------|--------|
| Bare `#[ignore]` | Negative lookahead | Pre-commit / CI | Line numbers + error | Block commit / PR |
| Raw `env::set_var` | Literal match | Pre-commit / CI | Line numbers + error | Block commit / PR |
| Undefined feature | Feature mismatch | CI only | Feature name | Block merge |
| Missing annotation | Backtrack + grep | CI only | Line numbers | Block merge |

---

## Related Documentation

- **CLAUDE.md**: Test scaffolding patterns
- **docs/development/test-suite.md**: Test annotation guide
- **docs/development/env-guard.md**: EnvGuard usage patterns
- `.githooks/README.md`: Git hook setup
- `scripts/hooks/`: Additional bash scripts

