# BitNet.rs Pull Request Guidelines

## Overview

This document establishes guidelines for pull request size and scope to maintain code quality, enable effective reviews, and prevent oversized PRs that are difficult to review and merge.

## PR Size Guidelines

### 🎯 Target Sizes

| Category | Line Changes | Files | Review Time | Approval Required |
|----------|-------------|--------|-------------|------------------|
| **Tiny** | <100 lines | <5 files | 5-15 minutes | 1 reviewer |
| **Small** | 100-500 lines | 5-15 files | 15-30 minutes | 1 reviewer |
| **Medium** | 500-1,000 lines | 15-30 files | 30-60 minutes | 1-2 reviewers |
| **Large** | 1,000-5,000 lines | 30-50 files | 1-2 hours | 2 reviewers |
| **Oversized** | 5,000-50,000 lines | 50+ files | 2+ hours | **Requires justification** |
| **Massive** | 50,000+ lines | 100+ files | Unreviewable | **Must be split** |

### 📏 Size Calculation

PR size is calculated as: `additions + deletions = total changes`

**Examples:**
- Adding 200 lines to a new file = 200 changes ✅ Small
- Modifying 50 lines (25 additions, 25 deletions) = 50 changes ✅ Tiny  
- Refactoring 1,000 lines (500 additions, 500 deletions) = 1,000 changes ✅ Medium

## Common Causes of Oversized PRs

### 🚨 Anti-Patterns to Avoid

1. **Build Artifacts**
   ```
   ❌ fixtures/target/debug/
   ❌ test_data/models/large.gguf
   ❌ .cargo/registry/
   ```

2. **Test Fixture Bloat**
   ```
   ❌ tests/fixtures/tokenizer_data.json (2MB)
   ❌ test_data/large_corpus.txt (10MB)
   ❌ benchmark_data/model.safetensors (500MB)
   ```

3. **Vendored Dependencies**
   ```
   ❌ vendor/llama.cpp/ (entire repo)
   ❌ 3rdparty/ggml/ (C++ source)
   ❌ thirdparty/cuda/ (SDK files)
   ```

4. **Scope Creep**
   ```
   ❌ Adding tokenizer + quantizer + server changes in one PR
   ❌ Feature implementation + refactoring + CI updates
   ❌ Multiple unrelated bug fixes
   ```

5. **Branch Merge Issues**
   ```
   ❌ Accidentally including other PRs' changes
   ❌ Merge conflicts with unrelated files
   ❌ Rebasing onto wrong base branch
   ```

## BitNet.rs Specific Guidelines

### 🎯 Focused PR Examples

**✅ Good PR Examples:**
- `feat(kernels): add AVX-512 quantization` (~200 lines)
- `fix(cli): handle GGUF parsing errors` (~50 lines)  
- `docs: update GPU setup guide` (~100 lines)
- `test(inference): add streaming token tests` (~150 lines)

**❌ Bad PR Examples:**
- `Implement everything for v2.0` (50,000+ lines)
- `Add tokenizer + fix CI + update docs + refactor kernels` (10,000+ lines)
- Including `fixtures/target/` build artifacts (300MB of generated files)

### 🔧 Crate-Specific Guidelines

#### bitnet-kernels
- **New kernel**: 100-300 lines per algorithm
- **SIMD optimization**: 50-150 lines per architecture  
- **GPU kernels**: 200-500 lines including tests

#### bitnet-inference
- **Engine features**: 200-400 lines
- **Streaming**: 100-300 lines
- **Model loading**: 300-600 lines

#### bitnet-cli
- **New command**: 50-200 lines
- **Command option**: 20-100 lines
- **Output formatting**: 30-150 lines

#### bitnet-server
- **New endpoint**: 100-300 lines
- **Middleware**: 50-150 lines
- **SSE streaming**: 200-400 lines

## Automated Checks

### 🤖 PR Size Guard

The repository includes automated PR size checking:

- **Large PRs (1k-5k lines)**: Warning comment, normal review
- **Oversized PRs (5k-50k lines)**: Warning comment, requires justification
- **Massive PRs (50k+ lines)**: Warning comment, strongly recommend splitting

### 🚫 Blocked Patterns

The following patterns trigger warnings:

```bash
# Build artifacts
**/target/
**/build/
**/.cargo/

# Large data files
**/*.gguf
**/*.safetensors  
**/*.bin (>1MB)

# Test fixture bloat
**/fixtures/**/*.json (>100KB)
test_data/large/

# Vendored code
vendor/
3rdparty/
thirdparty/
```

## Best Practices

### ✅ Before Creating a PR

1. **Review changed files**
   ```bash
   git status
   git diff --stat
   git diff --numstat | awk '{sum+=$1+$2} END {print sum " total changes"}'
   ```

2. **Check for artifacts**
   ```bash  
   find . -name "target" -type d
   find . -name "*.gguf" -size +1M
   git ls-files | grep -E "\.(bin|so|dll|dylib)$"
   ```

3. **Validate scope**
   - Does this PR do exactly one thing?
   - Are all changes related to the stated goal?
   - Can any changes be deferred to separate PRs?

### ✅ When Your PR is Flagged as Large

1. **Immediate Actions**
   ```bash
   # Check what's actually changed
   git diff --name-status origin/main
   
   # Look for unexpected files
   git diff --name-only origin/main | sort
   
   # Check file sizes
   git diff --name-only origin/main | xargs ls -lh
   ```

2. **Identify the Problem**
   - Build artifacts: Remove and update `.gitignore`
   - Test fixtures: Use synthetic data or external storage
   - Scope creep: Split into multiple PRs
   - Merge issues: Rebase onto correct branch

3. **Split the PR**
   ```bash
   # Create focused branches
   git checkout -b feature/core-functionality
   git checkout -b feature/documentation  
   git checkout -b feature/tests
   
   # Cherry-pick specific commits
   git cherry-pick <commit-hash>
   ```

### ✅ Creating Focused PRs

1. **One Feature Per PR**
   - ✅ "Add I2S quantization"
   - ❌ "Add quantization and fix CI and update docs"

2. **Logical Boundaries**
   - Core implementation
   - Tests for implementation  
   - Documentation updates
   - Examples and demos

3. **Incremental Development**
   - PR 1: Basic functionality (200 lines)
   - PR 2: SIMD optimization (150 lines)
   - PR 3: GPU acceleration (300 lines)
   - PR 4: Integration tests (100 lines)

## Emergency Procedures

### 🆘 If Your PR Becomes Massive

1. **Stop Development**
   - Don't add more changes
   - Don't try to "fix everything in one go"

2. **Analyze the Damage**
   ```bash
   # Get size breakdown by file type
   git diff --name-only origin/main | \
     xargs wc -l | \
     sort -nr | \
     head -20
   ```

3. **Create Recovery Plan**
   - Identify core changes (usually <500 lines)
   - List accidental inclusions (build artifacts, etc.)
   - Plan focused PRs for remaining work

4. **Execute Recovery**
   ```bash
   # Save current work
   git branch emergency-backup
   
   # Start fresh focused branch
   git checkout -b feature/focused-implementation main
   
   # Cherry-pick only core changes
   git cherry-pick -n <core-commit>
   git add <only-core-files>
   git commit -m "focused implementation"
   ```

## Historical Context

This guideline was established after resolving massive PRs #92, #101, #103, and #104, which grew to 300k+ lines due to:

- **Build artifacts** in `fixture_test/target/` (PR #92)
- **Vendored GGML C code** without proper `.gitignore` (PR #101)  
- **Huge JSON test fixtures** and scope creep (PR #103)
- **Branch merge issues** accumulating unrelated changes (PR #104)

These PRs were closed and the valuable core changes were extracted into focused implementations.

## FAQ

**Q: What if my feature genuinely requires many changes?**
A: Break it into logical increments. Most large features can be implemented in 3-5 focused PRs of 200-500 lines each.

**Q: What about generated code or data migration?**
A: These are valid exceptions but require explicit justification in the PR description and pre-approval from maintainers.

**Q: Can I temporarily exceed size limits?**
A: Oversized PRs (5k-50k lines) are allowed with justification. Massive PRs (50k+ lines) should always be split.

**Q: What if the PR size guard is wrong?**
A: The automated check can have false positives. Human reviewers make the final decision, but large PRs still require extra scrutiny.

---

**Remember:** Small PRs are faster to review, easier to test, simpler to revert, and reduce merge conflicts. When in doubt, split it out! 🔀