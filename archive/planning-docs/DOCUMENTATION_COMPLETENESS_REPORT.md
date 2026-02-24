# BitNet.rs CI Hardening Documentation Completeness Report

**Date**: 2025-11-05  
**Scope**: Verification of documentation alignment with CI hardening work (guards.yml, property-tests.yml updates)  
**Status**: CRITICAL INCONSISTENCIES FOUND

---

## Executive Summary

The CI hardening work has been successfully implemented in workflows but **documentation has inconsistencies and gaps** that could confuse contributors and cause CI failures:

### Critical Issues Found
1. **MSRV Inconsistency** - Multiple documents state different MSRV values (1.89.0 vs 1.90.0)
2. **Missing --locked Documentation** - Key docs don't mention the new `--locked` requirement
3. **PR Template Mismatch** - Two PR templates with conflicting requirements
4. **Guard Requirements Not Documented** - CONTRIBUTING.md lacks comprehensive CI guard details
5. **Documentation Gaps** - Several docs lack examples of guard-compliant commands

### Risk Level
- **High**: MSRV confusion could cause CI blocks on toolchain issues
- **High**: Missing `--locked` guidance could cause failed PRs after hardening
- **Medium**: PR template confusion could lead to rejected submissions
- **Medium**: Incomplete guard documentation reduces contributor confidence

---

## Detailed Findings

### 1. MSRV Inconsistency (CRITICAL)

**Problem**: Multiple documents specify different MSRV values, conflicting with `rust-toolchain.toml`

| Document | MSRV | Status |
|----------|------|--------|
| `rust-toolchain.toml` | 1.89.0 | **AUTHORITATIVE** |
| `Cargo.toml` (`rust-version`) | 1.89.0 | ✅ Consistent |
| `CLAUDE.md` | 1.89.0 | ✅ Correct |
| `CONTRIBUTING.md` | 1.89.0 | ✅ Correct |
| `README.md` | 1.89.0 (badge) | ✅ Badge correct |
| `README.md` (Quick Start section) | 1.90.0+ | ❌ **WRONG** |
| `README.md` (Status line) | 1.90.0 | ❌ **WRONG** |
| `.github/PULL_REQUEST_TEMPLATE.md` | 1.90.0 | ❌ **WRONG** |
| `Cargo.toml` (`cargo-mutants` msrv) | 1.90.0 | ❌ Test tool artifact |

**Impact**: 
- Contributors may try to build with 1.90.0 and encounter non-standard features
- CI guards enforce 1.89.0, so hardcoded 1.90.0 in docs will cause confusion
- Contributors may submit PRs that hardcode 1.90.0, triggering guards check failure

**Requires Fix**:
- `README.md` - 2 locations (lines 30, 513)
- `.github/PULL_REQUEST_TEMPLATE.md` - 1 location (line 23)

**Guards Check**: 
```yaml
# From guards.yml line 58-62
if rg -n --color=never --glob '!guards.yml' \
     -e 'toolchain:\s*"?1\.90\.0"?' \
     .github/workflows; then
  echo "❌ Found stale toolchain 1.90.0 in workflows (MSRV is 1.89.0)"
```
This check will BLOCK PRs that add 1.90.0 references to workflows.

---

### 2. Missing --locked Documentation (HIGH)

**Problem**: New `--locked` requirement (guards.yml lines 99-115) is not well-documented in contributor guides

**Coverage Analysis**:

✅ **Well Documented**:
- `CONTRIBUTING.md` - Line 509: Explicit requirement in PR checklist
- `CONTRIBUTING.md` - Lines 534-535: Instructions to use `--locked`
- `pull_request_template.md` - Line 10: Requirement checklist item

❌ **Missing or Insufficient**:
- `docs/development/build-commands.md` - Lines 8-97: Shows cargo commands **without** `--locked` flag
  - None of the 30+ example commands include `--locked`
  - This is the primary reference for build commands
  - Contradicts guards enforcement
  
- `CLAUDE.md` - Lines 49-117: **Quick Reference** section shows mixed usage
  - Line 58: `cargo test` missing `--locked`
  - Lines 66, 71, 80: `cargo run` missing `--locked`
  - Lines 50-51: `cargo build` missing `--locked`
  - Only some commands properly include it

- `README.md` - Lines 41-48: Build instructions lack `--locked`
  - All 4 build examples missing flag
  - This is the first thing new contributors see

**Impact**: 
- Contributors following documented examples will fail guards check
- GitHub Actions runs with `--locked` but local development doesn't match
- Workflow reproducibility not enforced locally

**Guard Failure Example**:
```bash
# This will FAIL guards check:
cargo build --release --no-default-features --features cpu

# This will PASS:
cargo build --release --no-default-features --features cpu --locked
```

**Requires Fix**:
- `README.md` - All build examples (4 locations, lines 41-48)
- `CLAUDE.md` - Quick Reference section (8+ locations)
- `docs/development/build-commands.md` - All 30+ examples
- Policy: Add note "All cargo invocations in CI use `--locked`" near top of each doc

---

### 3. PR Template Duplication & Mismatch (MEDIUM-HIGH)

**Problem**: Two PR templates exist with conflicting requirements

**Files Found**:
1. `.github/pull_request_template.md` (48 lines) - **NEWER**, comprehensive
2. `.github/PULL_REQUEST_TEMPLATE.md` (48 lines) - **OLDER**, minimal guards info

**Template Differences**:

| Aspect | `.github/pull_request_template.md` | `.github/PULL_REQUEST_TEMPLATE.md` |
|--------|------|------|
| Guards Requirements | ✅ Explicit (lines 5-12) | ❌ Minimal (line 22 only) |
| MSRV Version | ❌ 1.89.0 (line 11) | ❌ 1.90.0 (line 23) |
| --locked Flag | ✅ Mentioned (line 10) | ❌ Not explicitly mentioned |
| Scope | Generic PRs | Build/Format focused |
| Test Command | `cargo test` (line 25) | `cargo nextest run` (line 28) |
| Fixture Integrity | ✅ Yes (lines 33-34) | ❌ No |
| EnvGuard Pattern | ✅ Yes (lines 35-36) | ❌ No |

**Git Behavior**: GitHub recognizes both templates and may pick either one randomly

**Impact**: 
- Contributors see different requirements depending on which template loads
- One template references actions that need SHA-pinning but doesn't clarify it
- Fixture-specific requirements missing from one template
- EnvGuard requirements missing from one template

**Requires Fix**:
- **Recommendation**: Delete `PULL_REQUEST_TEMPLATE.md` (uppercase)
- Use `.github/pull_request_template.md` (lowercase) as single source of truth
- Ensure all requirements align with guards.yml enforcement

---

### 4. Guards Requirements Not Clearly Documented (MEDIUM)

**Problem**: CONTRIBUTING.md has guards info but it's scattered and incomplete

**Current Coverage in CONTRIBUTING.md**:

✅ **Present**:
- Lines 142-148: "CI Alignment" section mentioning same validation scripts
- Lines 485-502: "CI and Supply Chain Requirements" section with details
- Lines 508-511: "PR Checklist (CI Requirements)" with 4 items
- Lines 567: Reference to `hooks.yml` in troubleshooting section

❌ **Missing**:
- No explicit explanation of what the "Guards" check does
- No details on individual guard names (no-floating-actions, MSRV-consistency, --locked-everywhere)
- No explanation of failure modes
- No troubleshooting section for guard failures
- No link to guards.yml for reference

**Current Structure Problem**:
- Guards information scattered across "Pre-Commit Hooks", "CI and Supply Chain", and "PR Checklist"
- Not clear that some guards are local (pre-commit) and some are CI-only
- EnvGuard pattern documented but not linked to guards check

**Requires Addition**:
- New section: "Understanding CI Guards" with:
  - Overview of 4 guards enforced in guards.yml
  - When each guard runs (local vs CI)
  - Common violation patterns with examples
  - Troubleshooting guide
  - Reference to `.github/workflows/guards.yml`

---

### 5. Documentation of Guard Details (MEDIUM)

**Problem**: guards.yml has hardening work but contributor-facing documentation doesn't explain it

**What guards.yml enforces** (from workflow lines 43-115):
1. **No floating action refs** - All GitHub Actions must use SHA pins
2. **MSRV consistency** - Only 1.89.0 allowed
3. **--locked everywhere** - All cargo commands in workflows must use --locked
4. **Dev-only flags warning** - Informational check for deprecated flags
5. **Bare cfg(cuda) warning** - Informational check for feature consistency
6. **Cargo --locked validation** - Enforces deterministic builds

**Current Documentation**:
- ❌ No doc mentions "Guards" workflow by name
- ❌ No doc explains guard enforcement mechanism
- ✅ Some requirements mentioned in CONTRIBUTING.md but not linked to specific guards
- ❌ No troubleshooting guide for guard failures
- ❌ No examples of guard-compliant vs non-compliant code

**Requires Addition**:
- Dedicated section "CI Guards" in CONTRIBUTING.md
- Reference table showing which guard enforces which requirement
- Examples of violations and fixes
- Link to guards.yml as authoritative source

---

### 6. Documentation Gaps for Cross-Validation (LOW-MEDIUM)

**Problem**: Cross-validation requirements from property-tests.yml not documented

**What changed**:
- property-tests.yml now uses SHA-pinned actions (line 37-42)
- Toolchain hardened to 1.89.0 (line 42)

**Documentation Status**:
- ✅ CLAUDE.md documents cross-validation commands
- ✅ CONTRIBUTING.md mentions "Cross-validation Tests" (line 214)
- ❌ No mention that property tests now enforce stricter CI standards
- ❌ No guidance on maintaining MSRV in cross-validation code

**Impact**: Low - mainly affects maintainers, not contributors

---

## Detailed Recommendations

### Priority 1: Fix MSRV Inconsistency (DO FIRST)

**Files to update**:
1. `README.md` - Line 30: Change "1.90.0+" to "1.89.0+"
2. `README.md` - Line 513: Change "1.90.0" to "1.89.0"
3. `.github/PULL_REQUEST_TEMPLATE.md` - Line 23: Change "1.90.0" to "1.89.0"

**Verification**:
```bash
# After fix, these should all show 1.89.0:
grep -n "MSRV\|rust-version\|toolchain" \
  README.md CONTRIBUTING.md CLAUDE.md .github/PULL_REQUEST_TEMPLATE.md \
  rust-toolchain.toml Cargo.toml
```

**Why**: Guards check (guards.yml:58-62) will BLOCK PRs that harden toolchain to 1.90.0

---

### Priority 2: Update Build Command Documentation (DO SECOND)

**Files to update** (add `--locked` to all cargo commands):
1. `README.md` - Lines 41-48 (4 build examples)
2. `CLAUDE.md` - Lines 49-117 (Quick Reference section, ~8 commands)
3. `docs/development/build-commands.md` - All examples (~30 commands)

**Strategy**: Use a unified note at top of each section:
```markdown
> Note: All cargo commands in CI workflows use `--locked` to ensure deterministic builds.
> Follow the same pattern in local development for consistency.
```

**Template fix**:
```bash
# BEFORE:
cargo build --release --no-default-features --features cpu

# AFTER:
cargo build --release --no-default-features --features cpu --locked
```

**Verification**:
```bash
# Should have zero results after fix:
grep -r "cargo build\|cargo test\|cargo run" docs/ README.md CLAUDE.md \
  | grep -v "\-\-locked" | grep -v "# " | wc -l
```

**Why**: guards.yml lines 99-115 enforce this across workflows

---

### Priority 3: Consolidate PR Templates

**Action**:
1. Keep `.github/pull_request_template.md` (lowercase)
2. Delete `.github/PULL_REQUEST_TEMPLATE.md` (uppercase)
3. Fix MSRV in kept template: 1.90.0 → 1.89.0 (line 11 of lowercase file)

**Verification**:
```bash
# After deletion, only lowercase template should exist:
ls -la .github/*pull_request* 2>/dev/null
```

**Why**: Prevents confusion when contributors see conflicting requirements

---

### Priority 4: Document Guards Workflow

**Add new section to CONTRIBUTING.md** (after line 481, before "## Local Development Setup"):

```markdown
## Understanding CI Guards

The **Guards** workflow (`.github/workflows/guards.yml`) enforces supply chain security 
and reproducibility standards on all PRs. Violations will block merge.

### Guard Enforcements

| Guard | Enforces | Scope | Example |
|-------|----------|-------|---------|
| **No floating action refs** | All GitHub Actions use SHA pins | Workflows | ❌ `@v3` → ✅ `@abc123` |
| **MSRV consistency** | Toolchain is 1.89.0 only | Workflows | ❌ 1.90.0 → ✅ 1.89.0 |
| **--locked everywhere** | All cargo commands use --locked | Workflows | ❌ `cargo test` → ✅ `cargo test --locked` |
| **Dev-only flags** | Deprecated environment variables not set | Workflows | ⚠️ `BITNET_CORRECTION_POLICY` |
| **Feature gate hygiene** | No bare `cfg(cuda)` in tests | Test code | ⚠️ Use `#[cfg(any(feature = "gpu", feature = "cuda"))]` |

### How Guards Work

1. **Local**: Pre-commit hooks validate some guards locally
2. **CI**: Full validation runs on every PR in the `Guards` job (5-minute timeout)
3. **Blocking**: Guard failure prevents merge to `main`

### Common Violations & Fixes

**Violation 1: Floating GitHub Action Reference**
```yaml
# ❌ FAILS GUARDS
- uses: actions/checkout@v4

# ✅ PASSES GUARDS  
- uses: actions/checkout@08eba0b27e820071cde6df949e0beb9ba4906955  # v4
```

**Violation 2: Missing --locked Flag**
```bash
# ❌ FAILS GUARDS
cargo test --workspace --no-default-features --features cpu

# ✅ PASSES GUARDS
cargo test --workspace --no-default-features --features cpu --locked
```

**Violation 3: MSRV Hardcoded**
```yaml
# ❌ FAILS GUARDS (in .github/workflows/*.yml)
toolchain: "1.90.0"

# ✅ PASSES GUARDS (use rust-toolchain.toml)
toolchain: 1.89.0
```

### Troubleshooting Guard Failures

**"Floating GitHub Action refs detected"**
- Problem: Action uses `@v3`, `@main`, `@stable`, or `@latest`
- Solution: Replace with full SHA from action's release page (find via GitHub API or UI)
- Tool: Use action's commit SHA when browsing GitHub releases

**"Found stale toolchain 1.90.0"**
- Problem: Hardcoded `1.90.0` in workflow file
- Solution: Use `rust-toolchain.toml` for toolchain management
- Reference: `.github/workflows/guards.yml` line 58-62

**"Missing --locked in workflow cargo command"**
- Problem: `cargo build`/`test`/`run`/`bench`/`clippy` missing `--locked`
- Solution: Add `--locked` flag to all cargo invocations
- Why: Ensures reproducible builds using `Cargo.lock` versions
- Reference: `.github/workflows/guards.yml` line 99-115

### When to Use Pre-Commit Hooks vs Guards

- **Pre-commit hooks** (local): Catch issues before commit, fast feedback
- **CI Guards** (remote): Comprehensive validation, runs on every PR

Both use the same validation scripts, so local errors will match CI failures.
```

---

### Priority 5: Update CLAUDE.md Quick Reference

**Current issues** (lines 49-117):
- Lines 50-51: `cargo build` missing `--locked`
- Line 58: `cargo test` missing `--locked`  
- Lines 71, 80: `cargo run` missing `--locked`
- Missing note about CI guard enforcement

**Fix approach**:
1. Add `--locked` to all cargo commands
2. Add comment about guard compliance
3. Add note that examples show CI requirements

**Example fix**:
```bash
# BEFORE (line 58):
cargo test --workspace --no-default-features --features cpu

# AFTER:
# Note: All cargo commands use --locked for CI determinism
cargo test --workspace --no-default-features --features cpu --locked
```

---

### Priority 6: Update docs/development/build-commands.md

**Current status**: 30+ examples without `--locked` flag

**Fix strategy**:
1. Add note at top: "All examples use `--locked` for deterministic builds (CI requirement)"
2. Update all `cargo build`, `cargo test`, `cargo bench`, `cargo run`, `cargo clippy` commands
3. Add reference comment linking to guards.yml

**Coverage**: Lines 8-97 contain most examples that need updates

---

## Verification Checklist

### Before Merging CI Hardening Documentation Updates

- [ ] All MSRV references changed to 1.89.0 (not 1.90.0)
  - [ ] README.md line 30
  - [ ] README.md line 513
  - [ ] `.github/PULL_REQUEST_TEMPLATE.md` line 23

- [ ] All `cargo` command examples include `--locked`
  - [ ] README.md (4 examples)
  - [ ] CLAUDE.md Quick Reference (8+ examples)
  - [ ] docs/development/build-commands.md (30+ examples)

- [ ] PR template consolidation complete
  - [ ] `.github/PULL_REQUEST_TEMPLATE.md` deleted (uppercase)
  - [ ] `.github/pull_request_template.md` verified as lowercase
  - [ ] MSRV corrected in lowercase template

- [ ] CI Guards documentation added to CONTRIBUTING.md
  - [ ] New "Understanding CI Guards" section
  - [ ] Guard enforcement table
  - [ ] Common violations with examples
  - [ ] Troubleshooting guide

- [ ] Cross-references updated
  - [ ] CONTRIBUTING.md links to guards.yml
  - [ ] Documentation mentions guard enforcement
  - [ ] Example commands match guard requirements

### Testing Documentation Consistency

```bash
# Verify MSRV consistency
echo "=== MSRV References ==="
grep -rn "MSRV\|1\.89\|1\.90" \
  README.md CONTRIBUTING.md CLAUDE.md .github/PULL_REQUEST_TEMPLATE.md \
  rust-toolchain.toml Cargo.toml | grep -E "(1\.89|1\.90)"

# Verify --locked usage in examples  
echo "=== Cargo Commands Without --locked ==="
grep -rn "cargo build\|cargo test\|cargo run\|cargo bench" \
  README.md CLAUDE.md docs/development/build-commands.md | \
  grep -v "\-\-locked" | grep -v "^#" | head -20

# Verify no uppercase PR template
echo "=== PR Templates ==="
ls -la .github/*pull_request* 2>/dev/null
```

---

## Risk Assessment

### If Documentation NOT Updated

| Issue | Impact | Likelihood | Mitigation |
|-------|--------|------------|-----------|
| MSRV confusion | Contributors hardcode 1.90.0 → PR fails guards check | High | CI will block, clear error message |
| Missing --locked | Contributors follow examples → PR fails guards check | High | CI will block, clear error message |
| PR template conflict | Inconsistent requirements shown to different contributors | Medium | Resolve template duplication |
| Guards not documented | Contributors don't understand why PR failed | High | Add Guards documentation section |
| Contributor frustration | Unclear requirements lead to rework | High | Complete documentation updates |

### Severity Escalation Path
1. **Immediate**: MSRV inconsistency blocks PRs (guards check will fail)
2. **Short-term**: Missing --locked blocks PRs (guards check will fail)
3. **Long-term**: Undocumented guards reduce contributor satisfaction

---

## Summary of Changes Required

### Documentation Files to Update

1. **README.md** - 2 lines
   - Line 30: "1.90.0+" → "1.89.0+"
   - Line 513: "1.90.0" → "1.89.0"
   - Lines 41-48: Add `--locked` to all build examples

2. **CONTRIBUTING.md** - Add/update sections
   - Add "Understanding CI Guards" section (750+ words)
   - Verify "CI and Supply Chain Requirements" section
   - Update examples with `--locked` flag

3. **CLAUDE.md** - Lines 49-117
   - Add `--locked` to all cargo commands in Quick Reference

4. **docs/development/build-commands.md** - All cargo examples
   - Add `--locked` to ~30+ examples
   - Add note about guard compliance

5. **.github/pull_request_template.md** (lowercase)
   - Fix MSRV 1.90.0 → 1.89.0
   - Verify guards requirements clearly stated

6. **.github/PULL_REQUEST_TEMPLATE.md** (uppercase)
   - **RECOMMEND DELETION** - Keep only lowercase template

---

## References

**Workflows Modified**:
- `.github/workflows/guards.yml` - NEW: 5-minute guard validation
- `.github/workflows/property-tests.yml` - UPDATED: SHA pins, MSRV enforcement

**Authoritative Sources**:
- `rust-toolchain.toml` - MSRV 1.89.0 (canonical)
- `Cargo.toml` - rust-version = "1.89.0"
- `.github/workflows/guards.yml` - Guard enforcement rules

**Current Documentation Status**:
- CONTRIBUTING.md - Primary contributor guide (partially updated)
- CLAUDE.md - Internal development guide (partial --locked coverage)
- README.md - User-facing documentation (MSRV inconsistency)
- docs/ - Architecture and development docs (mixed compliance)

---

## Conclusion

The CI hardening work successfully enforces reproducible builds and supply chain security through the new Guards workflow. However, **documentation has not fully caught up with these requirements**, creating risk of:

1. Contributor confusion about MSRV version (1.89.0 vs 1.90.0)
2. Failed PRs due to missing `--locked` flags in examples
3. Unclear understanding of what Guards enforce and why

**Recommended approach**: Execute Priority 1-2 fixes immediately (MSRV + --locked), add Priority 4 documentation (Guards explanation), and clean up Priority 3 (PR template). Total effort: ~3-4 hours for comprehensive documentation update.

