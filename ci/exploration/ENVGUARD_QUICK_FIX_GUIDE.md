# EnvGuard Compilation Issues - Quick Fix Guide

**Last Updated**: 2025-10-22 | **Status**: Ready to Apply | **Est. Fix Time**: 15-30 minutes

## TL;DR - The Three Fixes

```bash
# 1. Add missing dependency
# File: crates/bitnet-tokenizers/Cargo.toml
# Add to [dependencies]:
once_cell.workspace = true

# 2. Fix type annotation
# File: tests/support/env_guard.rs:130
# Change from:
let lock = ENV_LOCK.lock().unwrap_or_else(|poisoned| {
# Change to:
let lock = ENV_LOCK.lock().unwrap_or_else(|poisoned: std::sync::PoisonError<std::sync::MutexGuard<'static, ()>>| {

# 3. Fix all 26 incorrect API usages
# See "API Usage Fixes" section below
```

---

## Issue 1: Missing Dependency (5 seconds)

### File
`crates/bitnet-tokenizers/Cargo.toml`

### What to add
In the `[dependencies]` section, add:
```toml
once_cell.workspace = true
```

### Why
- `env_guard.rs` uses `use once_cell::sync::Lazy;` (line 74)
- Crate includes `env_guard.rs` via `include!()` macro
- Must declare the dependency for the crate that uses it

---

## Issue 2: Type Annotation (2 minutes)

### File
`tests/support/env_guard.rs`, lines 130-135

### Current Code
```rust
let lock = ENV_LOCK.lock().unwrap_or_else(|poisoned| {
    poisoned.into_inner()
});
```

### Fixed Code
```rust
let lock = ENV_LOCK.lock().unwrap_or_else(|poisoned: std::sync::PoisonError<std::sync::MutexGuard<'static, ()>>| {
    poisoned.into_inner()
});
```

### Why
- Rust compiler cannot infer complex nested type: `PoisonError<MutexGuard<()>>`
- Closure parameter types must be explicit for `unwrap_or_else()`

---

## Issue 3: API Usage Fixes (10-20 minutes)

### The Problem
26 incorrect usages of `EnvGuard::set()` and `EnvGuard::remove()` across 7 files.

### The Correct Pattern
```rust
// CORRECT ✅
let guard = EnvGuard::new("VAR_NAME");      // Create guard
guard.set("value");                         // Call method on instance
// ... use variable ...
// guard drops here, restoring original state

// INCORRECT ❌
let _guard = EnvGuard::set("VAR_NAME", "value");  // Calls static method that doesn't exist
```

### Files to Fix (26 total occurrences)

#### 1. `crates/bitnet-tokenizers/src/fallback.rs` (1 fix)
**Line 486**:
```rust
// BEFORE:
let _guard = EnvGuard::set("BITNET_STRICT_TOKENIZERS", "1");

// AFTER:
let guard = EnvGuard::new("BITNET_STRICT_TOKENIZERS");
guard.set("1");
```

#### 2. `xtask/tests/verify_receipt.rs` (3 fixes)
Search for: `EnvGuard::set("BITNET_ALLOW_CORRECTIONS"` and `EnvGuard::remove("BITNET_ALLOW_CORRECTIONS"`

**Pattern**:
```rust
// BEFORE:
let _guard = EnvGuard::set("BITNET_ALLOW_CORRECTIONS", "1");
let _guard = EnvGuard::remove("BITNET_ALLOW_CORRECTIONS");

// AFTER:
let guard = EnvGuard::new("BITNET_ALLOW_CORRECTIONS");
guard.set("1");

// For remove:
let guard = EnvGuard::new("BITNET_ALLOW_CORRECTIONS");
guard.remove();
```

#### 3. `tests/common/github_cache.rs` (1 fix)
**Pattern**:
```rust
// BEFORE:
let _guard = EnvGuard::set("GITHUB_ACTIONS", "true");

// AFTER:
let guard = EnvGuard::new("GITHUB_ACTIONS");
guard.set("true");
```

#### 4. `tests/common/env.rs` (2 doc comment fixes)
These are in documentation/examples - update similarly.

#### 5. `crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs` (11 fixes)
```rust
// BEFORE (multiple instances):
let _g1 = EnvGuard::set("BITNET_DETERMINISTIC", "1");
let _g2 = EnvGuard::set("BITNET_SEED", "42");
let _g3 = EnvGuard::set("RAYON_NUM_THREADS", "1");

// AFTER (same pattern for all):
let g1 = EnvGuard::new("BITNET_DETERMINISTIC");
g1.set("1");
let g2 = EnvGuard::new("BITNET_SEED");
g2.set("42");
let g3 = EnvGuard::new("RAYON_NUM_THREADS");
g3.set("1");
```

#### 6. `crates/bitnet-inference/tests/issue_254_ac6_determinism_integration.rs` (5 fixes)
Same pattern as above.

#### 7. `crates/bitnet-inference/tests/issue_254_ac4_receipt_generation.rs` (4 fixes)
Same pattern as above.

---

## Verification Steps

After applying all fixes:

### 1. Check compilation
```bash
cargo check --tests 2>&1 | grep "error\[E0433\]\|error\[E0282\]"
# Should output nothing (no errors)
```

### 2. Verify no incorrect usages remain
```bash
grep -r "EnvGuard::set\|EnvGuard::remove" --include="*.rs" \
  | grep -v "pub fn set\|pub fn remove" | grep -v "//" | wc -l
# Should output: 0
```

### 3. Run env_guard tests
```bash
cargo test --workspace --lib env_guard --no-default-features
```

### 4. Run affected test suites
```bash
cargo test -p bitnet-tokenizers --no-default-features --features cpu fallback
cargo test -p bitnet-inference --no-default-features --features cpu issue_254
```

---

## Quick Fix Checklist

- [ ] **Dependency Fix**: Add `once_cell.workspace = true` to `crates/bitnet-tokenizers/Cargo.toml`
- [ ] **Type Annotation**: Fix `|poisoned|` to `|poisoned: std::sync::PoisonError<std::sync::MutexGuard<()>|` in `tests/support/env_guard.rs:130`
- [ ] **API Usage**: Fix all 26 occurrences across 7 files (see files list above)
- [ ] **Verify Compilation**: Run `cargo check --tests` 
- [ ] **Verify No Incorrect Usage**: Run grep command to confirm all fixed
- [ ] **Run Tests**: Confirm env_guard and affected test suites pass

---

## Implementation Strategy

### Option A: Manual (15-20 minutes, more thorough)
1. Fix issue #1 (30 seconds)
2. Fix issue #2 (2 minutes)
3. Fix issue #3 by going through each file and updating manually
4. Verify all fixes (5 minutes)

### Option B: Semi-Automated (10-15 minutes, faster)
1. Fix issues #1 and #2 (2.5 minutes)
2. Use Find & Replace (IDE or sed) with verification:
   ```
   Find: EnvGuard::set\("([^"]+)", "([^"]+)"\)
   Replace: EnvGuard::new("$1");\nguard.set("$2");
   ```
3. **IMPORTANT**: Manually verify each change (10-15 replacements to review)
4. Verify all fixes (5 minutes)

### Option C: Pure Manual (20-30 minutes, safest)
Use your IDE's Find & Replace feature:
1. Search for: `EnvGuard::set(`
2. For each result: manually apply the fix pattern
3. Repeat for `EnvGuard::remove(`
4. Verify

---

## Common Mistakes to Avoid

### ❌ Don't do this:
```rust
let _guard = EnvGuard::set("VAR", "1");  // Looks like static method but isn't
let guard: EnvGuard = EnvGuard::set("VAR", "1");  // Won't compile
guard.set("1");  // Won't work - set() returns (), not Self
```

### ✅ Do this:
```rust
let guard = EnvGuard::new("VAR");  // Creates guard
guard.set("1");                     // Calls method on instance
```

---

## Need More Details?

See the comprehensive analysis in:
- **Full Analysis**: `ci/exploration/issue_envguard_compilation.md` (769 lines)
- **Summary**: `ci/exploration/ENVGUARD_ANALYSIS_SUMMARY.txt`

These documents contain:
- Complete root cause analysis for each issue
- Design philosophy and safety guarantees
- All 26 incorrect usages listed by file and line
- Complete API reference
- Testing and verification procedures

---

## Timeline Estimate

| Task | Time | Notes |
|------|------|-------|
| Add `once_cell` dependency | 30 sec | 1 line change |
| Fix type annotation | 2 min | 1 line change |
| Fix 26 API usages | 10-20 min | 26 changes, 7 files |
| Verify compilation | 2 min | Run cargo check |
| Test fixes | 5 min | Run test suites |
| **Total** | **15-30 min** | All changes complete |

---

## Questions?

The comprehensive analysis document has answers for:
- Why does this happen?
- How does the RAII pattern work?
- What's the two-tiered isolation strategy?
- Why was EnvGuard designed this way?
- How can we prevent this in the future?

See: `ci/exploration/issue_envguard_compilation.md`
