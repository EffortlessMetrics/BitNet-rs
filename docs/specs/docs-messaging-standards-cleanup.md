# Documentation Cleanup and Message Standards Specification

**Specification ID**: SPEC-DOCS-MSG-001
**Version**: 1.0.0
**Status**: Draft
**Date**: 2025-10-26
**Target Release**: v0.2.0

---

## Executive Summary

This specification addresses systematic cleanup of ambiguous "when available" phrasing across the BitNet-rs codebase and establishes comprehensive message standards for CLI tools, error reporting, and documentation. The goal is to eliminate timing ambiguity, provide consistent user experiences, and enable robust CI/CD integration through clear exit codes and recovery guidance.

**Scope**: Affects 70+ instances of ambiguous phrasing across CLI help text, error messages, diagnostic output, and documentation (CLAUDE.md, docs/howto/, docs/explanation/).

**Key Problems Addressed**:
1. **Ambiguous timing**: "when available" doesn't clarify build-time vs runtime detection
2. **Inconsistent messaging**: Error messages lack structured recovery steps
3. **Missing exit code documentation**: Help text doesn't explain exit codes or recovery paths
4. **Fragmented terminology**: Multiple terms for the same concept (e.g., "backend available", "libraries found")

**Key Outcomes**:
1. Zero "when available" phrasing in codebase (verified with grep)
2. Consistent "detected at build time" terminology for library availability
3. 4-part error message template (Status, Error Detail, Recovery Steps, Documentation)
4. 8-section help text standard template with exit codes and recovery
5. Updated CLAUDE.md with new workflows and preflight examples
6. Updated docs/howto/cpp-setup.md reflecting dual-backend setup patterns

---

## Table of Contents

1. [Acceptance Criteria](#acceptance-criteria-ac1-ac10)
2. [Terminology Replacement Strategy](#terminology-replacement-strategy)
3. [Error Message Templates](#error-message-templates-4-part-structure)
4. [Help Text Standards](#help-text-standards-8-section-template)
5. [Documentation Updates](#documentation-updates-file-by-file)
6. [Automated Cleanup Scripts](#automated-cleanup-scripts)
7. [Testing and Verification](#testing-and-verification)
8. [Exit Code Reference](#exit-code-reference-table)
9. [Implementation Roadmap](#implementation-roadmap)

---

## Acceptance Criteria (AC1-AC10)

### AC1: Zero "When Available" Phrasing in Codebase

**Verification Command**:
```bash
# Should return zero results
grep -rn "when available\|if available\|runtime availability\|as available" \
  xtask/src bitnet-cli/src tests docs --include="*.rs" --include="*.md"
```

**Success Criteria**:
- Exit code 1 (no matches found)
- All 70+ instances replaced with context-specific terminology

**Test Script**: `scripts/verify_no_ambiguous_phrasing.sh`

---

### AC2: Consistent "Detected at Build Time" Terminology

**Verification Pattern**:
```bash
# Should find consistent build-time language
grep -rn "detected at build time\|compiled with" \
  xtask/src/crossval/preflight.rs bitnet-cli/src/main.rs
```

**Success Criteria**:
- Library availability described as "detected at build time"
- Feature flags described as "compiled if X feature enabled"
- Runtime library resolution described as "resolved via LD_LIBRARY_PATH/rpath"

**Examples**:
| Before | After |
|--------|-------|
| "GPU support when available" | "GPU support (compiled if gpu feature enabled)" |
| "Backend available" | "Backend libraries detected at build time" |
| "Runtime availability" | "Runtime library resolution via dynamic loader" |

---

### AC3: Runtime Fallback Documented with Rebuild Guidance

**Verification**:
```bash
# Should find rebuild instructions in error messages
grep -rn "cargo build -p xtask\|cargo clean -p xtask" \
  xtask/src/crossval/preflight.rs
```

**Success Criteria**:
- Error messages explain why rebuild is needed (build-time constants)
- Exact rebuild commands provided in error output
- Environment variable alternatives documented (BITNET_CROSSVAL_LIBDIR)

**Example Message**:
```
❌ Backend 'bitnet.cpp' UNAVAILABLE

Build-time detection failed. Libraries not present when xtask was compiled.

Recovery Steps:
1. Libraries installed after build:
   cargo clean -p xtask && cargo build -p xtask --features crossval-all

2. Override detection with explicit library path:
   export BITNET_CROSSVAL_LIBDIR=/path/to/libs
   cargo run -p xtask -- preflight --backend bitnet

Documentation:
  See: docs/howto/cpp-setup.md
```

---

### AC4: RepairMode Documented in All Relevant Help Text

**Verification**:
```bash
# Should find RepairMode in help text
cargo run -p xtask -- preflight --help | grep -i "repair"
```

**Success Criteria**:
- Help text documents all RepairMode variants (Auto, Never, Always)
- CI-aware defaults explained (Auto in local dev, Never in CI)
- Override flag `--repair` documented with examples

**Help Text Section**:
```
REPAIR MODES:
  auto (default in local dev)   - Automatically provision missing backends
  never (default in CI)          - Fail fast if backend missing
  always                         - Force refresh even if backend present

ENVIRONMENT DETECTION:
  CI=true, GITHUB_ACTIONS=true   → RepairMode::Never (safe default)
  Interactive terminal           → RepairMode::Auto (user-friendly)

EXAMPLES:
  # Let preflight auto-repair missing backend
  cargo run -p xtask -- preflight --backend bitnet --repair=auto

  # Disable auto-repair (CI-friendly)
  cargo run -p xtask -- preflight --backend bitnet --repair=never

  # Force refresh of existing backend
  cargo run -p xtask -- preflight --backend bitnet --repair=always
```

---

### AC5: Exit Code Reference Table in Docs and CLI Help

**Verification**:
```bash
# Check docs existence
test -f docs/reference/exit-codes.md && echo "PASS: Reference table exists"

# Check CLI help includes exit codes
cargo run -p xtask -- preflight --help | grep -i "exit code"
```

**Success Criteria**:
- `docs/reference/exit-codes.md` created with complete taxonomy
- All command help text includes EXIT CODES section
- Exit codes documented with recovery actions

**Exit Codes Section in Help**:
```
EXIT CODES:
  0 - Backend available (ready for cross-validation)
  1 - Backend unavailable (repair disabled or failed)
  2 - Invalid arguments (check --help for valid options)
  3 - Auto-repair failed: network error (retryable)
  4 - Auto-repair failed: permission error (requires user action)
  5 - Auto-repair failed: build error (install dependencies)
  6 - Recursion detected (internal error, report bug)

RECOVERY BY EXIT CODE:
  Exit 0: No action needed
  Exit 1: Enable auto-repair with --repair=auto
  Exit 2: Check command syntax with --help
  Exit 3: Check network with 'ping github.com', retry
  Exit 4: Fix ownership with 'chown -R $USER /path'
  Exit 5: Install cmake, gcc, git, then retry
  Exit 6: Report issue to maintainers

See: docs/reference/exit-codes.md for complete taxonomy
```

---

### AC6: Error Message Templates with 4-Part Structure

**Verification**:
```bash
# Check error messages follow template
grep -A 10 "RepairError::" xtask/src/crossval/preflight.rs | \
  grep -E "Error:|Recovery Steps:|Documentation:"
```

**Success Criteria**:
- All error types follow 4-part structure:
  1. Status message with icon (❌/⚠️/✓)
  2. Error Detail (specific cause)
  3. Recovery Steps (numbered, actionable)
  4. Documentation (links to relevant guides)

**Template Structure**:
```
[ICON] STATUS MESSAGE

Error Detail:
  <specific error with context>

Recovery Steps:
  1. <immediate action>
  2. <alternative if #1 fails>
  3. <documentation/support>

Documentation:
  See: <relative path to docs>

Exit code: N (semantic description)
```

---

### AC7: Help Text Follows 8-Section Standard Template

**Verification**:
```bash
# Check help text has all sections
cargo run -p xtask -- preflight --help | \
  grep -E "USAGE:|DESCRIPTION:|OPTIONS:|EXAMPLES:|EXIT CODES:|RECOVERY:|DOCUMENTATION:|CONTACT:"
```

**Success Criteria**:
- All commands use 8-section template:
  1. COMMAND NAME (with brief description)
  2. USAGE (syntax)
  3. DESCRIPTION (detailed explanation)
  4. OPTIONS (flags with defaults)
  5. EXAMPLES (concrete use cases)
  6. EXIT CODES (complete list)
  7. RECOVERY BY EXIT CODE (actionable steps)
  8. DOCUMENTATION (links to guides)

**Template**:
```
COMMAND NAME
  Brief description (1-2 sentences)

USAGE:
  cargo run -p xtask -- command [OPTIONS]

DESCRIPTION:
  Detailed description of what the command does

  Paragraph 2: More context if needed

  Key behavior: Specific to note

OPTIONS:
  --flag              Description (example values)
  --verbose, -v       Enable verbose diagnostics
  --help, -h          Show this help message

EXAMPLES:
  cargo run -p xtask -- command --flag value
  cargo run -p xtask -- command --verbose

EXIT CODES:
  0 - Success
  1 - General failure
  2 - Usage error
  3 - <domain-specific>

RECOVERY BY EXIT CODE:
  Exit 0: No action needed
  Exit 1: Enable verbose mode for more information
  Exit 2: Check available options with --help
  Exit 3: <specific recovery steps>

DOCUMENTATION:
  docs/howto/cpp-setup.md        - C++ reference setup
  docs/development/xtask.md      - xtask tooling reference

CONTACT:
  Report issues: https://github.com/bitnet-rs/BitNet-rs/issues
```

---

### AC8: Verbose Output with Timestamp and Phase Markers

**Verification**:
```bash
# Check verbose output includes timestamps
cargo run -p xtask -- preflight --backend bitnet --verbose 2>&1 | \
  grep -E "\[2025-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\]"
```

**Success Criteria**:
- All verbose output uses `[TIMESTAMP]` prefix
- Phase markers clearly separate stages (Config, Progress, Status, Result)
- Consistent timestamp format: `[YYYY-MM-DD HH:MM:SS]`

**Example Verbose Output**:
```
[2025-10-26 14:32:15] Operation: preflight check
[2025-10-26 14:32:15] Config: backend=bitnet, repair=auto
[2025-10-26 14:32:16] Progress: checking build-time detection
[2025-10-26 14:32:16] Status: HAS_BITNET=false (libraries not detected)
[2025-10-26 14:32:16] Progress: starting auto-repair
[2025-10-26 14:32:20] Status: cloning bitnet.cpp (45% complete)
[2025-10-26 14:32:50] Status: building with cmake
[2025-10-26 14:33:15] Status: build complete
[2025-10-26 14:33:16] Progress: rebuilding xtask
[2025-10-26 14:33:25] Status: xtask rebuild complete
[2025-10-26 14:33:26] Progress: re-checking detection
[2025-10-26 14:33:26] Status: HAS_BITNET=true (libraries detected)
[2025-10-26 14:33:26] Result: AVAILABLE (auto-repaired in 71s)
```

---

### AC9: CLAUDE.md Updated with New Workflows

**Verification**:
```bash
# Check CLAUDE.md has updated preflight workflows
grep -A 10 "preflight" CLAUDE.md | grep -E "repair=auto|RepairMode"
```

**Success Criteria**:
- CLAUDE.md documents preflight auto-repair workflows
- Examples show RepairMode variants
- C++ backend setup reflects dual-backend patterns (bitnet.cpp + llama.cpp)
- Exit code handling documented for CI/CD integration

**New Sections**:
1. **Preflight Auto-Repair** - One-command backend provisioning
2. **RepairMode Variants** - Auto/Never/Always with CI detection
3. **Dual-Backend Setup** - BitNet.cpp + llama.cpp patterns
4. **Exit Code Handling** - CI integration examples

---

### AC10: docs/howto/cpp-setup.md Reflects Dual-Backend Setup

**Verification**:
```bash
# Check cpp-setup.md has dual-backend guidance
grep -A 5 "BitNet.cpp\|llama.cpp" docs/howto/cpp-setup.md | \
  grep -E "preflight|setup-cpp-auto"
```

**Success Criteria**:
- Document updated with preflight auto-repair workflows
- Dual-backend setup patterns (BitNet.cpp for BitNet models, llama.cpp for LLaMA models)
- Backend selection heuristics documented (path-based auto-detection)
- Manual setup alternatives provided

**New Sections**:
1. **Quick Start: Auto-Provisioning** - One-command setup with preflight
2. **Backend Selection** - BitNet.cpp vs llama.cpp use cases
3. **Manual Setup** - Step-by-step for both backends
4. **Troubleshooting** - Common issues with preflight and setup-cpp-auto

---

## Terminology Replacement Strategy

### Context-Specific Replacements

| ❌ Ambiguous | ✅ Replacement | Context | File Pattern |
|-------------|---------------|---------|--------------|
| "when available" | "detected at build time" | Library detection | `**/preflight.rs`, `**/build.rs` |
| "if available" | "if gpu feature enabled" | Feature gates | `**/*.rs` (near `#[cfg(feature = ...)]`) |
| "backend available" | "backend libraries found" | Preflight status | `**/preflight.rs`, help text |
| "runtime availability" | "runtime library resolution" | Dynamic loading | `**/preflight.rs`, CLAUDE.md |
| "as available" | "in the detected configuration" | Adaptive behavior | CLAUDE.md, docs/ |
| "GPU when available" | "GPU (compiled if gpu feature enabled)" | Feature docs | CLAUDE.md, FEATURES.md |
| "C++ backend when available" | "C++ backend (detected at build time)" | Cross-validation | `**/crossval/*.rs` |

### Replacement Patterns by File Type

#### Rust Source Code (`*.rs`)

**Pattern 1: Feature Gate Documentation**
```rust
// Before:
/// Enable GPU support when available
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_function() { }

// After:
/// Enable GPU support (compiled if gpu or cuda feature enabled)
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_function() { }
```

**Pattern 2: Library Detection Messages**
```rust
// Before:
eprintln!("Backend available: using cross-validation");

// After:
eprintln!("Backend libraries detected at build time: using cross-validation");
```

**Pattern 3: Error Messages**
```rust
// Before:
bail!("Backend not available when needed");

// After:
bail!(
    "Backend libraries not detected at build time\n\
     \n\
     Rebuild xtask to refresh detection:\n\
       cargo clean -p xtask && cargo build -p xtask --features crossval-all"
);
```

#### Markdown Documentation (`*.md`)

**Pattern 1: CLAUDE.md Feature Descriptions**
```markdown
<!-- Before -->
- GPU acceleration (when available)
- C++ cross-validation (if backend available)

<!-- After -->
- GPU acceleration (compiled if gpu feature enabled)
- C++ cross-validation (backend libraries detected at build time)
```

**Pattern 2: Help Text in docs/howto/**
```markdown
<!-- Before -->
Cross-validation is enabled when the C++ backend is available.

<!-- After -->
Cross-validation is enabled when C++ backend libraries are detected at build time.
To provision backends automatically, use:
  cargo run -p xtask -- preflight --repair=auto
```

**Pattern 3: Environment Variable Documentation**
```markdown
<!-- Before -->
BITNET_GPU_LAYERS: Use GPU if available

<!-- After -->
BITNET_GPU_LAYERS: Configure GPU layer offloading (requires CUDA runtime)
  0: CPU-only inference (default)
  N: Offload first N layers to GPU
  -1: Auto-detect and offload all layers
```

---

## Error Message Templates (4-Part Structure)

### Template Definition

All error messages follow this 4-part structure:

```
[ICON] STATUS MESSAGE

Error Detail:
  <specific error with context>

Recovery Steps:
  1. <immediate action>
  2. <alternative if #1 fails>
  3. <documentation/support>

Documentation:
  See: <relative path to docs>

Exit code: N (semantic description)
```

### Template Variables

```rust
pub struct ErrorTemplate {
    icon: &'static str,           // ✓ (success), ❌ (failure), ⚠️  (warning)
    status: String,               // AVAILABLE, UNAVAILABLE, BUSY, etc.
    reason: Option<String>,       // cached, auto-repaired, network error, etc.
    backend: String,              // bitnet, llama
    error_detail: String,         // Specific error message or pattern
    recovery_steps: Vec<String>,  // Numbered recovery steps
    doc_path: &'static str,       // Relative path to relevant documentation
    exit_code: i32,               // 0-17, 130
    exit_description: &'static str, // Semantic exit code description
}
```

### Error Message Examples by Type

#### Network Error (Exit 3)

```
❌ Backend 'bitnet.cpp' UNAVAILABLE (network error during repair)

Error Detail:
  Connection timeout: github.com unreachable (30s timeout)
  Failed to clone: https://github.com/microsoft/BitNet.git

Recovery Steps:
  1. Check internet connectivity:
     ping github.com
  2. Verify firewall allows git clone:
     curl -I https://github.com
  3. Retry with backoff:
     cargo run -p xtask -- preflight --backend bitnet --repair=auto
  4. For persistent issues, see manual setup:
     docs/howto/cpp-setup.md

Exit code: 3 (repair failed: network error)
```

#### Permission Error (Exit 4)

```
❌ Backend 'bitnet.cpp' UNAVAILABLE (permission error during repair)

Error Detail:
  Permission denied: /home/user/.cache/bitnet_cpp
  Cannot create directory (EACCES)

Recovery Steps:
  1. Check directory ownership:
     ls -ld /home/user/.cache/bitnet_cpp
  2. Fix ownership:
     sudo chown -R $USER /home/user/.cache/bitnet_cpp
  3. OR use custom directory:
     export BITNET_CPP_DIR=~/my-custom-bitnet
     cargo run -p xtask -- preflight --backend bitnet --repair=auto
  4. See detailed setup guide:
     docs/howto/cpp-setup.md

Exit code: 4 (repair failed: permission error)
```

#### Build Error (Exit 5)

```
❌ Backend 'bitnet.cpp' UNAVAILABLE (build error during repair)

Error Detail:
  CMake error: Could not find CMake >= 3.18
  Build tools missing

Recovery Steps:
  1. Check required dependencies:
     cmake --version      # Need >= 3.18
     gcc --version        # or clang
     git --version
  2. Install missing tools:
     # Ubuntu/Debian
     sudo apt-get install cmake build-essential git
     # CentOS/RHEL
     sudo yum install cmake gcc-c++ git
     # macOS
     brew install cmake
  3. Retry repair:
     cargo run -p xtask -- preflight --backend bitnet --repair=auto
  4. For detailed setup:
     docs/development/build-commands.md
     docs/GPU_SETUP.md (for GPU-related issues)

Exit code: 5 (repair failed: build error)
```

#### Invalid Arguments (Exit 2)

```
error: invalid value for --backend: 'gpu'

Valid backends are:
  - bitnet   (Microsoft BitNet.cpp for BitNet models)
  - llama    (ggerganov/llama.cpp for LLaMA models)

Usage:
  cargo run -p xtask -- preflight --backend <BACKEND> [OPTIONS]

Examples:
  cargo run -p xtask -- preflight --backend bitnet --verbose
  cargo run -p xtask -- preflight --backend llama --repair=never

For more information:
  cargo run -p xtask -- preflight --help

Exit code: 2 (usage error)
```

#### Backend Unavailable, Repair Disabled (Exit 1)

```
❌ Backend 'bitnet.cpp' UNAVAILABLE (repair disabled)

Error Detail:
  Libraries not detected at build time
  Auto-repair disabled via --repair=never

Quick Fix:
  cargo run -p xtask -- preflight --repair=auto

Manual Setup:
  See: docs/howto/cpp-setup.md

Exit code: 1 (backend unavailable, repair disabled)
```

### Success Messages

#### Cached Detection (Exit 0)

```
✓ bitnet.cpp AVAILABLE (cached)
  Libraries detected at build time: /home/user/.cache/bitnet_cpp/build/lib
  Last xtask build: 2025-10-26 14:32:15 UTC

Exit code: 0 (success)
```

#### Auto-Repaired (Exit 0)

```
✓ bitnet.cpp AVAILABLE (auto-repaired)
  Setup completed in 52.18s
  Libraries installed: /home/user/.cache/bitnet_cpp/build/lib

Next: Rebuild xtask to detect libraries
  cargo clean -p xtask && cargo build -p xtask --features crossval-all

Exit code: 0 (success)
```

---

## Help Text Standards (8-Section Template)

### Template Structure

```
COMMAND NAME
  Brief description (1-2 sentences)

USAGE:
  cargo run -p xtask -- command [OPTIONS]

DESCRIPTION:
  Detailed description of what the command does

  Paragraph 2: More context if needed

  Key behavior: Specific to note

OPTIONS:
  --flag              Description (example values)
  --verbose, -v       Enable verbose diagnostics
  --help, -h          Show this help message

EXAMPLES:
  cargo run -p xtask -- command --flag value
  cargo run -p xtask -- command --verbose

EXIT CODES:
  0 - Success
  1 - General failure
  2 - Usage error
  3 - <domain-specific>
  ...
  See docs/reference/exit-codes.md for complete reference

RECOVERY BY EXIT CODE:
  Exit 0: No action needed
  Exit 1: Enable verbose mode for more information
  Exit 2: Check available options with --help
  Exit 3: <specific recovery steps>
  ...

DOCUMENTATION:
  docs/howto/cpp-setup.md        - C++ reference setup
  docs/development/xtask.md      - xtask tooling reference
  See more at: https://github.com/bitnet-rs/BitNet-rs/tree/main/docs

CONTACT:
  Report issues: https://github.com/bitnet-rs/BitNet-rs/issues
  See CONTRIBUTING.md for development guidelines
```

### Complete Example: Preflight Command

```
PREFLIGHT
  Check C++ backend library detection and auto-repair if needed

USAGE:
  cargo run -p xtask -- preflight [OPTIONS]

DESCRIPTION:
  Validates that required C++ libraries were detected at BUILD TIME
  (when xtask was compiled). Automatically provisions missing backends
  via setup-cpp-auto unless explicitly disabled.

  Build-time detection constants (HAS_BITNET, HAS_LLAMA) are baked
  into the xtask binary. If libraries are installed after xtask build,
  you must rebuild xtask to refresh detection.

  RepairMode defaults:
  - Interactive mode (local dev): Auto (provision missing backends)
  - CI environment (CI=true): Never (fail fast, pre-provision backends)

OPTIONS:
  --backend <BACKEND>     Backend to check: bitnet or llama
  --repair <MODE>         Repair mode: auto (default in local), never (default in CI), always
  --verbose, -v           Enable verbose diagnostics with timestamps
  --help, -h              Show this help message

EXAMPLES:
  # Check BitNet backend with auto-repair
  cargo run -p xtask -- preflight --backend bitnet

  # Check with verbose diagnostics
  cargo run -p xtask -- preflight --backend bitnet --verbose

  # Disable auto-repair (CI-friendly)
  cargo run -p xtask -- preflight --backend bitnet --repair=never

  # Force refresh of existing backend
  cargo run -p xtask -- preflight --backend llama --repair=always

EXIT CODES:
  0 - Backend available (libraries detected at build time)
  1 - Backend unavailable (libraries not found, repair disabled or failed)
  2 - Invalid arguments (unknown backend name)
  3 - Auto-repair failed: network error (retryable)
  4 - Auto-repair failed: permission error (requires action)
  5 - Auto-repair failed: build error (requires dependencies)
  6 - Recursion detected during repair (internal error)

  See docs/reference/exit-codes.md for complete taxonomy

RECOVERY BY EXIT CODE:
  Exit 0: No action needed, proceed with cross-validation
  Exit 1: Enable auto-repair with --repair=auto OR manually provision backend
  Exit 2: Check command syntax with --help
  Exit 3: Check network ('ping github.com'), verify firewall, retry
  Exit 4: Fix directory ownership ('sudo chown -R $USER /path'), OR override with BITNET_CPP_DIR
  Exit 5: Install dependencies (cmake >= 3.18, gcc/clang, git), then retry
  Exit 6: Report issue to maintainers with verbose output

DOCUMENTATION:
  docs/howto/cpp-setup.md           - C++ backend setup (BitNet.cpp + llama.cpp)
  docs/development/xtask.md         - xtask tooling reference
  docs/reference/exit-codes.md      - Complete exit code taxonomy
  docs/explanation/dual-backend-crossval.md - Dual-backend architecture

CONTACT:
  Report issues: https://github.com/bitnet-rs/BitNet-rs/issues
  See CONTRIBUTING.md for development guidelines
```

---

## Documentation Updates (File-by-File)

### Priority 1: Critical User-Facing Documentation

#### 1. CLAUDE.md

**Location**: `/home/steven/code/Rust/BitNet-rs/CLAUDE.md`

**Changes**:

**Section: Essential Commands**
- Add preflight auto-repair workflow
- Update setup-cpp-auto to include RepairMode examples
- Add dual-backend setup patterns

**Before**:
```markdown
# C++ reference auto-bootstrap (one-command setup)
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
```

**After**:
```markdown
# C++ reference auto-bootstrap (one-command setup)
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Preflight check with auto-repair (provision missing backends)
cargo run -p xtask --features crossval-all -- preflight --backend bitnet --repair=auto
cargo run -p xtask --features crossval-all -- preflight --backend llama --repair=auto

# CI-friendly preflight (fail if backend missing, no auto-repair)
CI=true cargo run -p xtask --features crossval-all -- preflight --backend bitnet
```

**Section: Feature Flags**
- Replace "when available" with "compiled if X feature enabled"

**Before**:
```markdown
- `gpu`: CUDA acceleration (when available)
- `crossval`: Cross-validation against Microsoft BitNet C++ (when available)
```

**After**:
```markdown
- `gpu`: CUDA acceleration (compiled if gpu feature enabled, requires CUDA runtime)
- `crossval`: Cross-validation against Microsoft BitNet C++ (backend libraries detected at build time)
```

**Section: Troubleshooting**
- Add preflight auto-repair guidance
- Add RepairMode failure recovery steps
- Add exit code interpretation

**New Subsection**:
```markdown
### Preflight Auto-Repair Failures

- **Exit 3 (Network Error)**: Check internet connectivity (`ping github.com`), verify firewall, retry
- **Exit 4 (Permission Error)**: Fix ownership (`sudo chown -R $USER ~/.cache/bitnet_cpp`) OR override with `BITNET_CPP_DIR`
- **Exit 5 (Build Error)**: Install dependencies (cmake >= 3.18, gcc/clang, git), then retry with `--repair=auto`
- **Exit 6 (Recursion Detected)**: Report bug with verbose output (`--verbose`)

For manual setup, see `docs/howto/cpp-setup.md`.
```

---

#### 2. docs/howto/cpp-setup.md

**Location**: `/home/steven/code/Rust/BitNet-rs/docs/howto/cpp-setup.md`

**New Structure**:

```markdown
# C++ Backend Setup Guide

## Table of Contents
1. [Quick Start: Auto-Provisioning](#quick-start-auto-provisioning)
2. [Backend Selection](#backend-selection)
3. [Manual Setup](#manual-setup)
4. [Troubleshooting](#troubleshooting)

---

## Quick Start: Auto-Provisioning

The fastest way to set up C++ backends is via **preflight auto-repair**:

```bash
# One-command setup for BitNet.cpp
cargo run -p xtask --features crossval-all -- preflight --backend bitnet --repair=auto

# One-command setup for llama.cpp
cargo run -p xtask --features crossval-all -- preflight --backend llama --repair=auto
```

**What happens**:
1. Preflight checks if backend libraries are detected at build time
2. If missing, automatically invokes `setup-cpp-auto` to provision the backend
3. Clones the C++ repository, builds with CMake, installs libraries
4. Shows rebuild instructions to refresh detection constants

**Expected output**:
```
✓ bitnet.cpp AVAILABLE (auto-repaired)
  Setup completed in 52.18s
  Libraries installed: /home/user/.cache/bitnet_cpp/build/lib

Next: Rebuild xtask to detect libraries
  cargo clean -p xtask && cargo build -p xtask --features crossval-all
```

---

## Backend Selection

BitNet-rs supports two C++ backends for cross-validation:

| Backend | Use Case | Models | Repository |
|---------|----------|--------|------------|
| **bitnet.cpp** | BitNet models (I2_S, TL1/TL2) | microsoft/bitnet-* | microsoft/BitNet |
| **llama.cpp** | LLaMA models (FP16, Q8_0, etc.) | meta-llama/*, HuggingFace | ggerganov/llama.cpp |

**Auto-detection heuristics** (used by `crossval-per-token`):
- Path contains "bitnet" or "microsoft/bitnet" → Uses bitnet.cpp
- Path contains "llama" → Uses llama.cpp
- Default fallback → llama.cpp (conservative)

**Override detection**:
```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --cpp-backend bitnet \
  --prompt "Test" \
  --max-tokens 4
```

---

## Manual Setup

If auto-repair fails or you prefer manual setup:

### Option 1: setup-cpp-auto (Recommended)

```bash
# Bash/Zsh
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Fish
cargo run -p xtask -- setup-cpp-auto --emit=fish | source

# PowerShell
cargo run -p xtask -- setup-cpp-auto --emit=pwsh | Invoke-Expression
```

### Option 2: Manual Clone and Build

**BitNet.cpp**:
```bash
# Clone BitNet.cpp
git clone https://github.com/microsoft/BitNet.git ~/.cache/bitnet_cpp
cd ~/.cache/bitnet_cpp

# Build with CMake
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Set environment variable
export BITNET_CPP_DIR=$HOME/.cache/bitnet_cpp

# Rebuild xtask to detect libraries
cd /path/to/BitNet-rs
cargo clean -p xtask && cargo build -p xtask --features crossval-all
```

**llama.cpp** (standalone):
```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git ~/.cache/llama_cpp
cd ~/.cache/llama_cpp

# Build with CMake
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build . --config Release

# Set environment variable
export BITNET_CPP_DIR=$HOME/.cache/llama_cpp

# Rebuild xtask to detect libraries
cd /path/to/BitNet-rs
cargo clean -p xtask && cargo build -p xtask --features crossval-all
```

---

## Troubleshooting

### Problem: "Backend libraries not detected at build time"

**Cause**: Libraries were not present when xtask was compiled, OR xtask was built without crossval features.

**Solutions**:
1. **Rebuild xtask** (if libraries installed after build):
   ```bash
   cargo clean -p xtask && cargo build -p xtask --features crossval-all
   ```

2. **Override detection** (if libraries in custom location):
   ```bash
   export BITNET_CROSSVAL_LIBDIR=/custom/path/to/libs
   cargo run -p xtask -- preflight --backend bitnet
   ```

3. **Verify library presence**:
   ```bash
   ls -la ~/.cache/bitnet_cpp/build/lib*.so
   ```

### Problem: Auto-repair fails with "Network error" (Exit 3)

**Cause**: Git clone timeout, DNS resolution failure, firewall blocking GitHub.

**Solutions**:
1. Check internet connectivity:
   ```bash
   ping github.com
   curl -I https://github.com
   ```

2. Verify firewall allows git clone (check iptables, corporate proxy)

3. Retry with backoff:
   ```bash
   cargo run -p xtask -- preflight --backend bitnet --repair=auto
   ```

4. Manual setup as fallback (see Option 2 above)

### Problem: Auto-repair fails with "Permission denied" (Exit 4)

**Cause**: Directory ownership issue, insufficient permissions.

**Solutions**:
1. Check ownership:
   ```bash
   ls -ld ~/.cache/bitnet_cpp
   ```

2. Fix ownership:
   ```bash
   sudo chown -R $USER ~/.cache/bitnet_cpp
   ```

3. Use custom directory:
   ```bash
   mkdir -p ~/my-bitnet-cpp
   export BITNET_CPP_DIR=~/my-bitnet-cpp
   cargo run -p xtask -- preflight --backend bitnet --repair=auto
   ```

### Problem: Auto-repair fails with "Build error" (Exit 5)

**Cause**: Missing dependencies (cmake, gcc, git).

**Solutions**:
1. Check dependencies:
   ```bash
   cmake --version      # Need >= 3.18
   gcc --version        # or clang
   git --version
   ```

2. Install missing tools:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install cmake build-essential git

   # CentOS/RHEL
   sudo yum install cmake gcc-c++ git

   # macOS
   brew install cmake
   ```

3. Retry repair:
   ```bash
   cargo run -p xtask -- preflight --backend bitnet --repair=auto
   ```

### Problem: Preflight succeeds but cross-validation fails

**Cause**: Library detected but incompatible version, ABI mismatch.

**Solutions**:
1. Verify library ABI:
   ```bash
   ldd ~/.cache/bitnet_cpp/build/libbitnet.so
   ```

2. Rebuild with matching flags:
   ```bash
   cd ~/.cache/bitnet_cpp/build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17
   cmake --build . --config Release
   ```

3. Force refresh:
   ```bash
   cargo run -p xtask -- preflight --backend bitnet --repair=always
   ```

---

For more help, see:
- `docs/explanation/dual-backend-crossval.md` - Architecture details
- `docs/reference/exit-codes.md` - Complete exit code reference
- GitHub Issues: https://github.com/bitnet-rs/BitNet-rs/issues
```

---

#### 3. docs/reference/exit-codes.md (NEW FILE)

**Location**: `/home/steven/code/Rust/BitNet-rs/docs/reference/exit-codes.md`

**Content**: See [Exit Code Reference Table](#exit-code-reference-table) section below.

---

### Priority 2: Developer Documentation

#### 4. docs/development/xtask.md

**Changes**:
- Add preflight command documentation with RepairMode
- Add exit code interpretation for CI/CD
- Update crossval-per-token with backend selection

**New Section: Preflight Auto-Repair**:
```markdown
## Preflight Command

**Purpose**: Check C++ backend library detection and auto-repair if needed

**Usage**:
```bash
cargo run -p xtask -- preflight [OPTIONS]
```

**Options**:
- `--backend <BACKEND>`: Backend to check (bitnet, llama)
- `--repair <MODE>`: Repair mode (auto, never, always)
- `--verbose`: Enable verbose diagnostics with timestamps

**RepairMode Variants**:
- `auto` (default in local dev): Automatically provision missing backends
- `never` (default in CI): Fail fast if backend missing
- `always`: Force refresh even if backend present

**Examples**:
```bash
# Check BitNet backend with auto-repair
cargo run -p xtask -- preflight --backend bitnet --repair=auto

# CI-friendly (fail if missing, no auto-provision)
CI=true cargo run -p xtask -- preflight --backend bitnet

# Verbose diagnostics
cargo run -p xtask -- preflight --backend bitnet --verbose
```

**Exit Codes**: See [Exit Code Reference](#exit-code-reference-table)
```

---

#### 5. docs/explanation/dual-backend-crossval.md

**Changes**:
- Replace "when available" with "detected at build time"
- Add preflight auto-repair integration
- Document RepairMode behavior

**Example Change**:
```markdown
<!-- Before -->
Cross-validation is enabled when the C++ backend is available at runtime.

<!-- After -->
Cross-validation is enabled when C++ backend libraries are detected at build time
(via crossval/build.rs). If libraries are not present during xtask compilation,
use preflight auto-repair to provision them:

  cargo run -p xtask -- preflight --backend bitnet --repair=auto

After provisioning, rebuild xtask to refresh detection constants:

  cargo clean -p xtask && cargo build -p xtask --features crossval-all
```

---

### Priority 3: Feature Documentation

#### 6. docs/explanation/FEATURES.md

**Changes**:
- Replace "when available" with feature-specific terminology

**Before**:
```markdown
- `gpu`: CUDA acceleration (when available)
- `crossval`: Cross-validation (when C++ backend available)
```

**After**:
```markdown
- `gpu`: CUDA acceleration (compiled if gpu feature enabled, requires CUDA runtime)
- `crossval`: Cross-validation (requires C++ backend libraries detected at build time)
```

---

## Automated Cleanup Scripts

### Script 1: Grep-Based Verification

**File**: `scripts/verify_no_ambiguous_phrasing.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# BitNet-rs - Verify zero ambiguous "when available" phrasing

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Verifying zero ambiguous phrasing in codebase..."
echo ""

# Define ambiguous patterns
PATTERNS=(
    "when available"
    "if available"
    "runtime availability"
    "as available"
)

# Search locations
SEARCH_PATHS=(
    "xtask/src"
    "bitnet-cli/src"
    "crates/*/src"
    "tests"
    "docs"
    "CLAUDE.md"
)

FOUND_COUNT=0

for pattern in "${PATTERNS[@]}"; do
    echo "Searching for: '$pattern'"

    # Grep with context
    MATCHES=$(grep -rn "$pattern" "${SEARCH_PATHS[@]}" --include="*.rs" --include="*.md" 2>/dev/null || true)

    if [[ -n "$MATCHES" ]]; then
        echo -e "${RED}FAIL: Found '$pattern' in:${NC}"
        echo "$MATCHES"
        echo ""
        FOUND_COUNT=$((FOUND_COUNT + 1))
    else
        echo -e "${GREEN}PASS: Zero instances of '$pattern'${NC}"
        echo ""
    fi
done

if [[ $FOUND_COUNT -gt 0 ]]; then
    echo -e "${RED}FAIL: Found $FOUND_COUNT ambiguous patterns${NC}"
    echo ""
    echo "Run bulk replacement script:"
    echo "  scripts/replace_ambiguous_phrasing.sh"
    exit 1
else
    echo -e "${GREEN}SUCCESS: Zero ambiguous phrasing detected${NC}"
    exit 0
fi
```

---

### Script 2: Bulk Replacement

**File**: `scripts/replace_ambiguous_phrasing.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# BitNet-rs - Bulk replace ambiguous phrasing with context-specific terminology

echo "BitNet-rs - Automated Ambiguous Phrasing Cleanup"
echo "================================================="
echo ""
echo "This script replaces ambiguous phrasing with context-specific terminology."
echo "Backup your changes with git before proceeding!"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Context 1: Preflight/library detection (build-time)
echo "Context 1: Library detection → 'detected at build time'"
find xtask/src/crossval -name "*.rs" -type f -exec \
  sed -i 's/when available/detected at build time/g' {} \;
find crossval/src -name "*.rs" -type f -exec \
  sed -i 's/backend available/backend libraries found/g' {} \;

echo "✓ Updated preflight and crossval modules"
echo ""

# Context 2: Feature gates → 'compiled if X feature enabled'
echo "Context 2: Feature gates → 'compiled if X feature enabled'"
find bitnet-cli/src crates/*/src -name "*.rs" -type f -exec \
  sed -i 's/GPU support when available/GPU support (compiled if gpu feature enabled)/g' {} \;
find docs -name "FEATURES.md" -type f -exec \
  sed -i 's/when available/compiled if feature enabled/g' {} \;

echo "✓ Updated feature gate documentation"
echo ""

# Context 3: Status messages → 'backend libraries found'
echo "Context 3: Status messages → 'backend libraries found'"
find xtask/src/crossval -name "*.rs" -type f -exec \
  sed -i 's/backend available/backend libraries found/g' {} \;

echo "✓ Updated status messages"
echo ""

# Context 4: CLAUDE.md → multiple context-specific replacements
echo "Context 4: CLAUDE.md → context-specific replacements"
sed -i 's/GPU support when available/GPU support (compiled if gpu feature enabled)/g' CLAUDE.md
sed -i 's/Cross-validation when available/Cross-validation (backend libraries detected at build time)/g' CLAUDE.md
sed -i 's/runtime availability/runtime library resolution/g' CLAUDE.md

echo "✓ Updated CLAUDE.md"
echo ""

# Context 5: docs/howto/ → 'detected at build time'
echo "Context 5: Documentation → 'detected at build time'"
find docs/howto -name "*.md" -type f -exec \
  sed -i 's/when available/detected at build time/g' {} \;

echo "✓ Updated howto guides"
echo ""

echo "Cleanup complete! Run verification:"
echo "  scripts/verify_no_ambiguous_phrasing.sh"
echo ""
echo "Then commit changes:"
echo "  git add -A"
echo "  git commit -m 'docs: remove ambiguous \"when available\" phrasing'"
```

---

### Script 3: Exit Code Verification

**File**: `scripts/verify_exit_codes.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# BitNet-rs - Verify exit code consistency across codebase

echo "Verifying exit code consistency..."
echo ""

# Expected exit codes
EXPECTED_CODES=(
    "EXIT_SUCCESS:0"
    "EXIT_UNAVAILABLE:1"
    "EXIT_INVALID_ARGS:2"
    "EXIT_REPAIR_NETWORK:3"
    "EXIT_REPAIR_PERMISSION:4"
    "EXIT_REPAIR_BUILD:5"
    "EXIT_RECURSION:6"
)

MISSING_COUNT=0

for code in "${EXPECTED_CODES[@]}"; do
    IFS=':' read -r name value <<< "$code"

    # Check if exit code is defined
    if ! grep -rq "const $name.*= $value" xtask/src; then
        echo "FAIL: Missing exit code constant: $name = $value"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    else
        echo "PASS: Found $name = $value"
    fi
done

if [[ $MISSING_COUNT -gt 0 ]]; then
    echo ""
    echo "FAIL: $MISSING_COUNT exit codes missing or inconsistent"
    exit 1
else
    echo ""
    echo "SUCCESS: All exit codes consistent"
    exit 0
fi
```

---

## Testing and Verification

### Test Plan

#### Phase 1: Automated Verification (AC1, AC5)

```bash
# AC1: Verify zero ambiguous phrasing
scripts/verify_no_ambiguous_phrasing.sh

# AC5: Verify exit codes
scripts/verify_exit_codes.sh
```

**Expected**:
- Exit 0 for both scripts
- Zero grep matches for ambiguous patterns
- All exit code constants defined

---

#### Phase 2: Manual Review (AC2-AC4, AC6-AC10)

**AC2: Terminology Consistency**
```bash
# Sample terminology in preflight.rs
grep -A 3 "detected at build time" xtask/src/crossval/preflight.rs

# Expected: "detected at build time" in status messages
```

**AC3: Rebuild Guidance**
```bash
# Check error messages include rebuild instructions
grep -A 10 "cargo clean -p xtask" xtask/src/crossval/preflight.rs

# Expected: Exact rebuild command in error output
```

**AC4: RepairMode Documentation**
```bash
# Check help text
cargo run -p xtask -- preflight --help | grep -i "repair"

# Expected: All RepairMode variants documented
```

**AC6: Error Message Structure**
```bash
# Check error messages follow 4-part template
grep -A 15 "Error Detail:" xtask/src/crossval/preflight.rs

# Expected: Status, Error Detail, Recovery Steps, Documentation
```

**AC7: Help Text Structure**
```bash
# Check help text has 8 sections
cargo run -p xtask -- preflight --help | \
  grep -E "USAGE:|DESCRIPTION:|OPTIONS:|EXAMPLES:|EXIT CODES:|RECOVERY:|DOCUMENTATION:"

# Expected: All 8 sections present
```

**AC8: Verbose Output**
```bash
# Run with verbose flag
cargo run -p xtask -- preflight --backend bitnet --verbose 2>&1 | head -20

# Expected: Timestamps, phase markers
```

**AC9: CLAUDE.md Updates**
```bash
# Check CLAUDE.md has new preflight section
grep -A 10 "preflight.*repair" CLAUDE.md

# Expected: RepairMode examples, dual-backend patterns
```

**AC10: cpp-setup.md Updates**
```bash
# Check cpp-setup.md has preflight guidance
grep -A 5 "Quick Start.*Auto-Provisioning" docs/howto/cpp-setup.md

# Expected: One-command setup examples
```

---

#### Phase 3: Integration Tests

**Test 1: Preflight with Auto-Repair**
```bash
# Test auto-repair workflow
rm -rf ~/.cache/bitnet_cpp  # Start clean
cargo clean -p xtask
cargo build -p xtask --features crossval-all
cargo run -p xtask -- preflight --backend bitnet --repair=auto --verbose

# Expected:
# - Auto-repair triggered (backend missing)
# - Setup-cpp-auto invoked
# - Success message with rebuild instructions
# - Exit code 0
```

**Test 2: Preflight with Repair Disabled**
```bash
# Test failure when repair disabled
rm -rf ~/.cache/bitnet_cpp
cargo run -p xtask -- preflight --backend bitnet --repair=never

# Expected:
# - Error message with manual setup guidance
# - Exit code 1 (unavailable, repair disabled)
```

**Test 3: CI Environment Detection**
```bash
# Test CI-aware defaults
CI=true cargo run -p xtask -- preflight --backend bitnet

# Expected:
# - RepairMode::Never (safe default)
# - Exit code 1 if backend missing
```

**Test 4: Exit Code Verification**
```bash
# Test network error (mock)
# (Requires mock setup-cpp-auto failure)
cargo run -p xtask -- preflight --backend bitnet --repair=auto
# Expected: Exit code 3 (network failure)

# Test permission error (mock)
# (Requires directory ownership issue)
# Expected: Exit code 4 (permission denied)

# Test build error (mock)
# (Requires cmake missing)
# Expected: Exit code 5 (build failure)
```

---

## Exit Code Reference Table

### Complete Exit Code Taxonomy

```
┌──────────────────────────────────────────────────────────────┐
│                 EXIT CODE QUICK REFERENCE                    │
├──────────────────────────────────────────────────────────────┤
│ GENERAL (POSIX-compatible)                                   │
│ 0   ✓ Success                                                │
│ 1   ❌ General failure (e.g., backend unavailable)          │
│ 2   ⚠️  Usage error (invalid arguments)                     │
│                                                               │
│ REPAIR OPERATIONS (3-6)                                      │
│ 3   ⚠️  Network error during repair (retryable)             │
│ 4   ⚠️  Permission error during repair (action needed)      │
│ 5   ❌ Build error during repair (install dependencies)     │
│ 6   ❌ Recursion error (internal bug)                       │
│                                                               │
│ DOWNLOAD/VALIDATION (10-17)                                  │
│ 10  ❌ Disk space exhausted                                  │
│ 11  ❌ Authentication required (HTTP 401/403)               │
│ 12  ⚠️  Rate limited (HTTP 429) - retry recommended        │
│ 13  ❌ Hash mismatch (redownload)                           │
│ 14  ❌ Network error (generic)                              │
│ 15  ❌ Verification failed                                   │
│ 16  ❌ Inference failed                                      │
│ 17  ❌ Benchmark regression                                  │
│                                                               │
│ SIGNAL HANDLING (130+)                                       │
│ 130 ⚠️  Interrupted (Ctrl+C, SIGINT)                        │
└──────────────────────────────────────────────────────────────┘
```

### Exit Code Details by Category

#### General (0-2)

| Code | Name | Meaning | Recovery | CI Action |
|------|------|---------|----------|-----------|
| 0 | EXIT_SUCCESS | All operations succeeded | N/A | Continue |
| 1 | EXIT_UNAVAILABLE | Backend missing, repair disabled/failed | Enable `--repair=auto` OR manually provision | Fail job |
| 2 | EXIT_INVALID_ARGS | Invalid CLI arguments | Check `--help` for syntax | Fix command |

#### Repair Operations (3-6)

| Code | Name | Meaning | Recovery | CI Action |
|------|------|---------|----------|-----------|
| 3 | EXIT_REPAIR_NETWORK | Git clone timeout, DNS failure | Check network (`ping github.com`), verify firewall, retry | Retry job later |
| 4 | EXIT_REPAIR_PERMISSION | Directory ownership issue | Fix with `chown -R $USER /path` OR override `BITNET_CPP_DIR` | Fix permissions |
| 5 | EXIT_REPAIR_BUILD | CMake error, compiler missing | Install cmake >= 3.18, gcc/clang, git, then retry | Install deps |
| 6 | EXIT_RECURSION | Recursion guard triggered | Report bug with verbose output | Fail job |

#### Download/Validation (10-17)

| Code | Name | Meaning | Recovery | CI Action |
|------|------|---------|----------|-----------|
| 10 | EXIT_NO_SPACE | Disk full during download | Free disk space | Fail job |
| 11 | EXIT_AUTH | HTTP 401/403 | Provide auth token | Set credentials |
| 12 | EXIT_RATE_LIMIT | HTTP 429 | Wait and retry | Retry with backoff |
| 13 | EXIT_HASH_MISMATCH | SHA256 verification failed | Redownload model | Fail job |
| 14 | EXIT_NETWORK | Generic network error | Check connectivity | Retry job |
| 15 | EXIT_VERIFICATION_FAILED | Model validation failed | Check model integrity | Fail job |
| 16 | EXIT_INFERENCE_FAILED | Runtime execution error | Debug with `--verbose` | Fail job |
| 17 | EXIT_BENCHMARK_FAILED | Performance regression | Review performance | Fail job |

#### Signal Handling (130+)

| Code | Name | Meaning | Recovery | CI Action |
|------|------|---------|----------|-----------|
| 130 | EXIT_INTERRUPTED | Ctrl+C or SIGINT | N/A (user-initiated) | Fail job |

---

### Exit Code Usage in CI

#### GitHub Actions Example

```yaml
name: preflight-check
on: [push, pull_request]

jobs:
  preflight:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run preflight check
        id: preflight
        run: |
          cargo run -p xtask --features crossval-all -- preflight --backend bitnet --verbose
        continue-on-error: true  # Don't fail job on exit code

      - name: Interpret exit code
        run: |
          EXIT_CODE=${{ steps.preflight.outcome }}
          case $EXIT_CODE in
            0) echo "✓ Backend available"; exit 0 ;;
            1) echo "❌ Backend unavailable"; exit 1 ;;
            2) echo "⚠️  Usage error"; exit 2 ;;
            3) echo "⚠️  Network error (retryable)"; exit 3 ;;
            4) echo "⚠️  Permission error"; exit 4 ;;
            5) echo "❌ Build error"; exit 5 ;;
            6) echo "❌ Recursion error"; exit 6 ;;
            *) echo "Unknown exit code: $EXIT_CODE"; exit 1 ;;
          esac
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

**Days 1-2: Exit Code Consolidation**
- [ ] Define unified exit code constants in `xtask/src/main.rs`
- [ ] Update `classify_exit()` function with RepairError integration
- [ ] Create `docs/reference/exit-codes.md` with complete taxonomy
- [ ] Add 12 exit code tests in `xtask/tests/exit_codes.rs`

**Days 3-4: Error Message Standardization**
- [ ] Create `ErrorTemplate` struct in `xtask/src/crossval/preflight.rs`
- [ ] Update all `RepairError` variants with 4-part structure
- [ ] Add recovery step documentation to error messages
- [ ] Add 5 message format tests

**Day 5: Terminology Cleanup Automation**
- [ ] Create `scripts/verify_no_ambiguous_phrasing.sh`
- [ ] Create `scripts/replace_ambiguous_phrasing.sh`
- [ ] Create `scripts/verify_exit_codes.sh`
- [ ] Run bulk replacement and verify with grep

---

### Phase 2: Documentation (Week 2)

**Days 6-7: CLAUDE.md Updates**
- [ ] Add preflight auto-repair workflows
- [ ] Document RepairMode variants with CI detection
- [ ] Update feature flag descriptions (remove "when available")
- [ ] Add troubleshooting section for preflight failures
- [ ] Add exit code interpretation for CI/CD

**Days 8-9: docs/howto/cpp-setup.md Overhaul**
- [ ] Rewrite with Quick Start: Auto-Provisioning section
- [ ] Add Backend Selection guide (bitnet.cpp vs llama.cpp)
- [ ] Update Manual Setup with both backends
- [ ] Expand Troubleshooting with exit code recovery
- [ ] Add dual-backend examples

**Day 10: Additional Documentation**
- [ ] Update `docs/development/xtask.md` with preflight
- [ ] Update `docs/explanation/dual-backend-crossval.md`
- [ ] Update `docs/explanation/FEATURES.md` (remove "when available")
- [ ] Create `docs/reference/exit-codes.md`

---

### Phase 3: Help Text (Week 3)

**Days 11-12: Help Text Standardization**
- [ ] Update `preflight` command help with 8-section template
- [ ] Add EXIT CODES section to all xtask commands
- [ ] Add RECOVERY BY EXIT CODE section
- [ ] Add DOCUMENTATION section with links
- [ ] Test help text rendering (`cargo run -p xtask -- preflight --help`)

**Days 13-14: Verbose Mode Enhancement**
- [ ] Add timestamp formatting to verbose output
- [ ] Add phase markers (Config, Progress, Status, Result)
- [ ] Update `RepairProgress` tracker with timestamps
- [ ] Test verbose output (`--verbose` flag)

**Day 15: CLI Integration**
- [ ] Wire preflight to main command dispatcher
- [ ] Add RepairMode CLI flag mapping
- [ ] Test RepairMode behavior (Auto/Never/Always)
- [ ] Test CI environment detection

---

### Phase 4: Testing (Week 4)

**Days 16-17: Automated Tests**
- [ ] Run `scripts/verify_no_ambiguous_phrasing.sh`
- [ ] Run `scripts/verify_exit_codes.sh`
- [ ] Verify AC1-AC10 manually
- [ ] Add integration tests for preflight workflows

**Days 18-19: Manual Testing**
- [ ] Test preflight with auto-repair (backend missing)
- [ ] Test preflight with repair disabled (CI-friendly)
- [ ] Test all exit codes (3-6) with mocked failures
- [ ] Test help text rendering for all commands

**Day 20: Documentation Review**
- [ ] Review CLAUDE.md for completeness
- [ ] Review docs/howto/cpp-setup.md for accuracy
- [ ] Review docs/reference/exit-codes.md for clarity
- [ ] Get feedback from reviewers

---

### Phase 5: Finalization (Week 5)

**Days 21-22: Bug Fixes and Refinement**
- [ ] Address feedback from reviewers
- [ ] Fix failing tests
- [ ] Refine error messages based on user testing
- [ ] Update documentation based on feedback

**Days 23-24: CI/CD Integration**
- [ ] Add preflight to CI workflows
- [ ] Test exit code handling in GitHub Actions
- [ ] Add monitoring for exit code rates
- [ ] Document CI integration patterns

**Day 25: Release Preparation**
- [ ] Final verification of AC1-AC10
- [ ] Run full test suite
- [ ] Generate changelog entries
- [ ] Tag v0.2.0-rc1 for release candidate testing

---

## Summary

This specification provides a comprehensive plan to eliminate ambiguous "when available" phrasing, establish consistent message standards, and enable robust CI/CD integration through clear exit codes and recovery guidance. The implementation spans 5 weeks with clear deliverables and verification criteria for each phase.

**Key Deliverables**:
1. Zero ambiguous phrasing (70+ instances replaced)
2. 4-part error message template (Status, Error, Recovery, Docs)
3. 8-section help text standard (including exit codes)
4. Complete exit code taxonomy (0-6, 10-17, 130)
5. Updated CLAUDE.md and docs/howto/cpp-setup.md
6. Automated cleanup and verification scripts

**Verification**:
- AC1-AC10 acceptance criteria (all testable with scripts or manual review)
- Automated verification scripts (grep-based, exit 0 on success)
- Integration tests (preflight workflows, exit codes)
- CI/CD examples (GitHub Actions, GitLab CI)

**Status**: Ready for implementation after review and approval.
