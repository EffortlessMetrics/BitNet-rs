# Environment Export → build.rs → HAS_* Constants: Complete Analysis Index

## Document Overview

This directory contains a comprehensive analysis of the **environment propagation gap** in the BitNet.rs auto-repair workflow. The gap prevents environment variables from `setup-cpp-auto` being propagated to child cargo build processes, causing the build.rs detection to fail.

### Documents in This Analysis

#### 1. **QUICK-REF-env-gap.md** (Start Here - 5 min read)
**Purpose**: Executive summary for busy developers  
**Covers**:
- The problem in 30 seconds (diagram)
- The three gaps explained
- 4-step solution overview
- Key files to modify
- Testing checklist

**Best For**: Getting up to speed quickly, understanding scope

---

#### 2. **env-export-build-gap-analysis.md** (Main Analysis - 20 min read)
**Purpose**: Detailed technical analysis of the complete pipeline  
**Covers**:
- **Executive Summary**: The core problem
- **Current Workflow**: 
  - setup-cpp-auto command output (lines 1-50)
  - Auto-repair flow with gap identification (lines 51-100)
  - build.rs detection logic (lines 101-150)
  - HAS_BITNET/HAS_LLAMA constants (lines 151-200)
- **The Gap in Detail**: 
  - Gap 1: Shell output not captured (lines 201-250)
  - Gap 2: Environment variables not applied (lines 251-300)
  - Gap 3: No env parsing infrastructure (lines 301-350)
- **Target Workflow**: Post-fix diagram (lines 351-400)
- **Implementation Plan**: 4 locations needing changes
- **Acceptance Criteria**: 4 validation checkpoints (AC1-AC4)
- **Risk Assessment**: Mitigation strategies

**Best For**: Understanding the root cause, making implementation decisions

---

#### 3. **env-gap-code-snippets.md** (Implementation Guide - 30 min read)
**Purpose**: Ready-to-use code snippets for fixing the gap  
**Covers**:
- **Current Broken Implementation** (with line numbers):
  - setup-cpp-auto output format
  - attempt_repair_once() showing Gap 1
  - rebuild_xtask() showing Gap 2
  - preflight_with_auto_repair() showing gap location
  
- **Fixed Implementation**:
  - parse_sh_exports() function (with unit tests)
  - rebuild_xtask_with_env() function
  - Modified attempt_repair_with_retry()
  - Modified attempt_repair_once()
  - Modified preflight_with_auto_repair()
  
- **Summary Table**: What changes, what fixes, impact

**Best For**: Implementation, code review, testing

---

## The Problem at a Glance

### Current Broken Flow
```
setup-cpp-auto                  rebuild_xtask()
(subprocess)                    (child cargo)
      |
      ├─ Installs libs          Inherits STALE env ✗
      ├─ Outputs exports        BITNET_CPP_DIR NOT set ✗
      |
      └─ stdout DISCARDED ✗     build.rs detection FAILS ✗
                                HAS_BITNET = false ✗
```

### Fixed Flow (Target)
```
setup-cpp-auto                  rebuild_xtask()
(subprocess)                    (child cargo)
      |
      ├─ Installs libs
      ├─ Outputs exports
      |
      └─→ [PARSE]               Gets env vars ✓
           [APPLY]              BITNET_CPP_DIR SET ✓
           |
           └──→ Command::new("cargo")
                .env("BITNET_CPP_DIR", ...)
                .spawn()
                |
                └─→ build.rs detection SUCCEEDS ✓
                    HAS_BITNET = true ✓
```

## Key Files Affected

### Primary Changes
- `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/preflight.rs`
  - Lines 1393-1407: Integration point (preflight_with_auto_repair)
  - Lines 1617-1639: rebuild_xtask function
  - Lines 1970-1976: attempt_repair_once (capture stdout)
  - NEW: parse_sh_exports() function
  - NEW: rebuild_xtask_with_env() function

### Reference Files (Read-Only)
- `/home/steven/code/Rust/BitNet-rs/xtask/src/cpp_setup_auto.rs` (lines 787-866)
  - Shows emit_exports() format
  - Input to parse_sh_exports()

- `/home/steven/code/Rust/BitNet-rs/crossval/build.rs` (lines 131-189)
  - Shows detection logic that depends on env vars
  - HAS_BITNET/HAS_LLAMA emission (lines 340-344)

## Implementation Roadmap

### Phase 1: Create Infrastructure (2-3 hours)
1. Implement `parse_sh_exports()` function
   - Handle all shell formats (sh, fish, pwsh, cmd)
   - Add unit tests (5 tests minimum)
2. Implement `rebuild_xtask_with_env()` function
   - Accept exports HashMap
   - Apply to child cargo process

### Phase 2: Integrate into Repair (1-2 hours)
1. Modify `attempt_repair_with_retry()` return type
   - Change to `Result<String, RepairError>`
   - Capture and return stdout
2. Modify `attempt_repair_once()` return type
   - Change to `Result<String, RepairError>`
   - Return setup_result.stdout
3. Modify `preflight_with_auto_repair()`
   - Integrate parse_sh_exports call
   - Integrate apply_env_exports call
   - Use rebuild_xtask_with_env instead of rebuild_xtask

### Phase 3: Test & Verify (2-3 hours)
1. Unit tests for parse_sh_exports()
   - sh format: ✓
   - fish format: ✓
   - pwsh format: ✓
   - Multiple exports: ✓
2. Integration test for environment propagation
   - Verify exports reach child process: ✓
   - Verify build.rs detection succeeds: ✓
   - Verify HAS_BITNET updated: ✓
3. End-to-end test
   - Full repair + rebuild + re-exec workflow

## Acceptance Criteria Checklist

### AC1: Shell Export Parsing
- [ ] Correctly parses sh format: `export VAR="value"`
- [ ] Correctly parses fish format: `set -gx VAR "value"`
- [ ] Correctly parses PowerShell format: `$env:VAR = "value"`
- [ ] Correctly parses cmd format: `set VAR=value`
- [ ] Handles variable references (e.g., `${LD_LIBRARY_PATH:-}`)
- [ ] Preserves value escaping
- [ ] Unit tests pass

### AC2: Environment Propagation
- [ ] Exports applied to current process
- [ ] Exports reach child Command processes via .env()
- [ ] Handles path-like variables correctly
- [ ] No crash on invalid env values
- [ ] Integration tests pass

### AC3: Integration with Repair Flow
- [ ] setup-cpp-auto output captured
- [ ] Exports parsed before rebuild_xtask call
- [ ] Child cargo receives BITNET_CPP_DIR
- [ ] build.rs detection succeeds with applied env
- [ ] build.rs warnings show libraries found (not STUB)

### AC4: Persistent Detection
- [ ] After repair + rebuild + re-exec: HAS_BITNET = true
- [ ] No "BITNET_STUB mode" warning after successful repair
- [ ] preflight check succeeds on second invocation

## Related Issues & Context

**Issue Context**: This analysis addresses the environment propagation gap that prevents auto-repair from functioning correctly. When `setup-cpp-auto` installs C++ libraries and outputs environment variable exports, those exports are lost in the subprocess stdout. The subsequent `rebuild_xtask()` call inherits the parent's stale environment, preventing build.rs from discovering the newly-installed libraries.

**Use Case**: 
```bash
# User runs preflight with auto-repair
cargo run -p xtask -- preflight --backend bitnet --repair=auto

# Expected flow:
# 1. Detects HAS_BITNET = false
# 2. Runs setup-cpp-auto (installs libs, outputs exports)
# 3. Parses exports from stdout [← MISSING]
# 4. Rebuilds xtask with env vars [← MISSING]
# 5. Re-exec with updated binary
# 6. HAS_BITNET = true [← FAILS without fix]
```

## Files Referenced

### Source Code
- `xtask/src/cpp_setup_auto.rs` - setup-cpp-auto implementation
- `xtask/src/crossval/preflight.rs` - auto-repair workflow
- `crossval/build.rs` - build-time detection logic

### Analysis Documents
- `docs/analysis/env-export-build-gap-analysis.md` - Full technical analysis
- `docs/analysis/QUICK-REF-env-gap.md` - Quick reference
- `docs/analysis/env-gap-code-snippets.md` - Implementation guide

### Test Location
- `xtask/tests/preflight_auto_repair_tests.rs` - Integration tests

## How to Use This Analysis

### For Specification Writers
1. Start with **QUICK-REF-env-gap.md** (overview)
2. Read **env-export-build-gap-analysis.md** sections:
   - Specific Locations Needing Changes
   - Implementation Plan
   - Acceptance Criteria
3. Use **env-gap-code-snippets.md** for precise specifications

### For Implementers
1. Start with **QUICK-REF-env-gap.md** (scope)
2. Read **env-gap-code-snippets.md** (copy/paste snippets)
3. Reference **env-export-build-gap-analysis.md** for context
4. Follow Implementation Roadmap (Phases 1-3)

### For Reviewers
1. Check against Acceptance Criteria (AC1-AC4)
2. Verify against env-gap-code-snippets.md changes
3. Run integration tests from Phase 3

### For QA/Testing
1. Use testing checklist from **QUICK-REF-env-gap.md**
2. Run unit tests from **env-gap-code-snippets.md** parse_tests
3. Run integration tests from Phase 3

## Quick Links

| Document | Purpose | Length | Audience |
|----------|---------|--------|----------|
| QUICK-REF-env-gap.md | Executive summary | 5 min | Everyone |
| env-export-build-gap-analysis.md | Technical deep-dive | 20 min | Architects, Implementers |
| env-gap-code-snippets.md | Implementation guide | 30 min | Developers |

## Implementation Estimate

- **Analysis Review**: 30 minutes
- **Code Implementation**: 2-3 hours
- **Testing**: 2-3 hours
- **Code Review**: 1-2 hours
- **Total**: 5-9 hours

## Risk Level: MEDIUM

**Mitigation Strategies**:
- Keep `rebuild_xtask()` as fallback (non-env version)
- Comprehensive unit tests for parsing
- Integration tests verify env propagation
- Graceful error handling (fallback to defaults)
- Verbose logging of env vars passed to cargo

## Success Metrics

After implementation:
1. Preflight auto-repair succeeds on first attempt
2. No "BITNET_STUB mode" warnings after repair
3. HAS_BITNET = true after rebuild + re-exec
4. All 4 ACs verified ✓
5. Unit + integration tests pass ✓

---

**Document Last Updated**: 2025-10-27  
**Status**: Analysis Complete - Ready for Implementation  
**Next Step**: Create precise AC specifications based on env-gap-code-snippets.md
