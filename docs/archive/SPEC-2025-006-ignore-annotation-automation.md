# SPEC-2025-006: #[ignore] Annotation Automation System

**Component**: Test infrastructure and CI quality gates
**Location**: `scripts/check-ignore-hygiene.sh`, `.github/workflows/ci.yml`
**Dependencies**: ripgrep, bash, cargo metadata, CI pipeline
**Related Issues**: Test maintenance debt, documentation clarity, CI hygiene

## Executive Summary

BitNet-rs currently has 194 `#[ignore]` annotations across the test suite, with 135 (69.6%) lacking explicit reasons. This creates maintenance debt, ambiguous test intent, and makes it difficult to track which tests are temporarily disabled vs permanently excluded. This specification defines an automated system for detecting, categorizing, and enforcing annotation hygiene for ignored tests.

**Current State**:
- **194 total** `#[ignore]` annotations
- **59 annotated** with explicit reasons (30.4%)
- **135 bare** without reasons (69.6%)
- **46 tests** blocked by known GitHub issues (#254, #260, #469)
- **29 tests** require GGUF model files
- **17 tests** marked slow (performance/integration)
- **Inconsistent patterns** across test categories

**Target State**:
- **<5% bare ignores** within 1 sprint (≤10 out of 194)
- **100% categorized** by standardized taxonomy
- **CI enforcement** preventing new bare annotations
- **Automated suggestions** for common ignore patterns
- **Quick-fix tooling** for bulk annotation

## Requirements Analysis

### Functional Requirements

1. **Detection System**:
   - Scan all `*.rs` test files for `#[ignore]` patterns
   - Distinguish between bare `#[ignore]` and annotated `#[ignore = "reason"]`
   - Extract surrounding context (comments, test name, file path)
   - Support incremental detection (git diff mode for CI)

2. **Categorization Engine**:
   - Apply taxonomy patterns to bare ignores
   - Match test names and comments against category heuristics
   - Handle multi-category tests (e.g., slow + GPU-dependent)
   - Provide confidence scores for suggested categories

3. **Auto-Annotation Generator**:
   - Suggest standardized reason strings based on context
   - Support dry-run mode (preview without modification)
   - Validate generated annotations (no duplicate reasons)
   - Preserve code formatting (rustfmt compatibility)

4. **CI Guard Job**:
   - Fail on new bare `#[ignore]` annotations in PR diffs
   - Provide actionable quick-fix suggestions in CI output
   - Support exemption mechanism for incremental migration
   - Generate metrics for tracking annotation progress

### Non-Functional Requirements

1. **Performance**:
   - Detection script completes in <5 seconds for full codebase scan
   - CI job overhead <30 seconds
   - Incremental mode for PR validation <10 seconds

2. **Accuracy**:
   - ≥95% categorization accuracy for issue-blocked tests
   - ≥85% accuracy for slow/performance tests
   - ≥90% accuracy for model/fixture requirements

3. **Maintainability**:
   - Taxonomy extensible via configuration file
   - Pattern matching uses standard ripgrep syntax
   - Clear separation of detection logic and categorization rules

4. **Integration**:
   - Works with existing CI workflows (`.github/workflows/ci.yml`)
   - Compatible with `cargo test`, `cargo nextest`, and `scripts/verify-tests.sh`
   - No additional runtime dependencies (uses tools already in CI)

## Technical Approach

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                  Ignore Hygiene System                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐      ┌────────────────┐                  │
│  │  Detection    │─────→│ Categorization │                  │
│  │  Engine       │      │ Engine         │                  │
│  │               │      │                │                  │
│  │ - Scan files  │      │ - Apply regex  │                  │
│  │ - Extract ctx │      │ - Score match  │                  │
│  │ - Git diff    │      │ - Multi-label  │                  │
│  └───────────────┘      └────────────────┘                  │
│         │                       │                            │
│         │                       ▼                            │
│         │              ┌────────────────┐                    │
│         │              │ Auto-Annotate  │                    │
│         │              │ Generator      │                    │
│         │              │                │                    │
│         │              │ - Suggest text │                    │
│         │              │ - Safe replace │                    │
│         │              │ - Dry-run mode │                    │
│         │              └────────────────┘                    │
│         │                       │                            │
│         ▼                       ▼                            │
│  ┌───────────────────────────────────────┐                  │
│  │         CI Guard Job                  │                  │
│  │                                        │                  │
│  │  - Fail on bare ignores               │                  │
│  │  - Generate quick-fix suggestions     │                  │
│  │  - Track annotation progress          │                  │
│  └───────────────────────────────────────┘                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Standardized Taxonomy

Based on analysis of existing annotations and test patterns, the following taxonomy provides comprehensive categorization:

#### Category 1: Issue-Blocked Tests (46 tests)

**Pattern Indicators**:
- Comments referencing `Issue #NNN`
- File paths containing `issue_NNN_`
- Test names matching `test_issue_NNN_*`

**Reason Template**:
```rust
#[ignore = "Issue #NNN: brief description of blocker"]
```

**Examples**:
```rust
// Good
#[ignore = "Issue #254: shape mismatch in layer-norm - needs investigation"]
#[ignore = "Issue #260: TDD placeholder - quantized_matmul not yet implemented"]
#[ignore = "Issue #159: TDD placeholder - weight loading implementation needed"]

// Bad (too vague)
#[ignore = "blocked"]
#[ignore = "Issue #254"]
```

**Regex Patterns**:
```regex
# File path detection
issue_(\d+)_\w+\.rs

# Comment detection
(?i)issue\s*#?(\d+)

# Test name detection
test_issue_(\d+)_
```

#### Category 2: Slow/Performance Tests (17 tests)

**Pattern Indicators**:
- Test names containing `slow`, `perf`, `benchmark`, `integration`
- Comments mentioning token generation counts (`50+`, `100+`)
- File paths containing `performance`, `benchmark`

**Reason Template**:
```rust
#[ignore = "slow: <runtime description>, see <fast alternative>"]
```

**Examples**:
```rust
// Good
#[ignore = "slow: 50+ token generations, use fast unit tests in ci mode"]
#[ignore = "slow: integration test, see faster equivalents in unit_tests.rs"]
#[ignore = "performance: timing-sensitive, causes non-deterministic CI failures"]
#[ignore = "benchmark: not a functional unit test"]

// Bad
#[ignore = "slow"]
#[ignore = "takes too long"]
```

**Regex Patterns**:
```regex
# Test name detection
test_\w*(slow|perf|benchmark|integration)

# Comment detection
(?i)(slow|performance|benchmark|timing)

# Token count detection
(\d+)\+?\s*tokens?
```

#### Category 3: Model/File Requirements (29 tests)

**Pattern Indicators**:
- Test names containing `model`, `gguf`, `weight`, `loading`
- Comments referencing `BITNET_GGUF`, model files
- Import statements for `bitnet-models`

**Reason Template**:
```rust
#[ignore = "requires: <specific resource>"]
```

**Examples**:
```rust
// Good
#[ignore = "requires: real GGUF model with complete metadata"]
#[ignore = "requires: bitnet.cpp FFI implementation"]
#[ignore = "fixture: missing test data at tests/fixtures/tokenizer.gguf"]

// Bad
#[ignore = "needs model"]
#[ignore = "no file"]
```

**Regex Patterns**:
```regex
# Test name detection
test_\w*(model|gguf|weight|loading|fixture)

# Environment variable references
BITNET_GGUF|BITNET_CPP_DIR

# File path patterns
\.gguf|\.safetensors|models/
```

#### Category 4: GPU/Hardware-Specific (13 tests)

**Pattern Indicators**:
- File paths in `gpu_*.rs`
- Test names containing `gpu`, `cuda`, `device`
- Feature gates `#[cfg(feature = "gpu")]`

**Reason Template**:
```rust
#[ignore = "gpu: <dependency description>"]
```

**Examples**:
```rust
// Good
#[ignore = "gpu: requires CUDA toolkit installed"]
#[ignore = "gpu: run with --ignored flag when CUDA available"]
#[ignore = "gpu: device-specific test for A100/V100 architectures"]

// Bad
#[ignore = "GPU"]
#[ignore = "needs CUDA"]
```

**Regex Patterns**:
```regex
# File name detection
gpu_\w+\.rs

# Test name detection
test_\w*(gpu|cuda|device)

# Feature gate detection
#\[cfg\(.*feature\s*=\s*"gpu"
```

#### Category 5: Network/External Dependencies (10 tests)

**Pattern Indicators**:
- Test names containing `download`, `fetch`, `network`, `api`
- Comments referencing HuggingFace, HTTP, async
- Import statements for `tokio`, `reqwest`

**Reason Template**:
```rust
#[ignore = "network: <dependency description>"]
```

**Examples**:
```rust
// Good
#[ignore = "network: requires HuggingFace API access"]
#[ignore = "network: requires internet connection and HF_TOKEN"]
#[ignore = "network: async test infrastructure not yet available"]

// Bad
#[ignore = "needs internet"]
#[ignore = "download test"]
```

**Regex Patterns**:
```regex
# Test name detection
test_\w*(download|fetch|network|api|http)

# Async patterns
async\s+fn\s+test

# External service references
(?i)(huggingface|hf_token|github|api)
```

#### Category 6: Feature/TODO Placeholders (14 tests)

**Pattern Indicators**:
- Comments with `TODO`, `FIXME`, `WIP`
- Test bodies containing `unimplemented!()`, `todo!()`
- Comments referencing future features

**Reason Template**:
```rust
#[ignore = "TODO: <specific implementation needed>"]
```

**Examples**:
```rust
// Good
#[ignore = "TODO: update to use QuantizedLinear::new_tl1() with proper LookupTable"]
#[ignore = "TODO: implement GPU mixed-precision tests after #439 resolution"]
#[ignore = "TODO: replace mock inference with real path"]

// Bad
#[ignore = "TODO"]
#[ignore = "not done"]
```

**Regex Patterns**:
```regex
# Comment markers
(?i)(todo|fixme|wip|placeholder)

# Unimplemented code
unimplemented!\(\)|todo!\(\)

# Feature references
after\s+#\d+|pending|future
```

#### Category 7: Quantization/Kernel Tests (22 tests)

**Pattern Indicators**:
- Test names containing quantization types (`i2s`, `tl1`, `tl2`, `qk256`)
- File paths in `quantization/`, `kernels/`
- References to SIMD operations (`avx2`, `neon`)

**Reason Template**:
```rust
#[ignore = "quantization: <format> - <reason>"]
```

**Examples**:
```rust
// Good
#[ignore = "quantization: I2S SIMD consistency tests need refinement"]
#[ignore = "quantization: QK256 property-based test blocked by Issue #159"]
#[ignore = "quantization: AVX2 optimization validation pending"]

// Bad
#[ignore = "quant test"]
```

**Regex Patterns**:
```regex
# Quantization format detection
(?i)(i2s|tl1|tl2|qk256|iq2_s)

# SIMD patterns
(?i)(avx2|avx512|neon|simd)

# File path patterns
(quantization|kernels)/
```

#### Category 8: Cross-Validation/Parity (7 tests)

**Pattern Indicators**:
- Test names containing `parity`, `crossval`, `reference`
- Comments referencing C++ reference, accuracy thresholds
- Import statements for `crossval` crate

**Reason Template**:
```rust
#[ignore = "parity: <comparison target> - <accuracy requirement>"]
```

**Examples**:
```rust
// Good
#[ignore = "parity: TDD Red phase - AC5 accuracy thresholds not yet met (Issue #254)"]
#[ignore = "parity: C++ reference comparison - requires bitnet.cpp setup"]
#[ignore = "parity: cosine similarity <0.999 in current implementation"]

// Bad
#[ignore = "parity test"]
```

**Regex Patterns**:
```regex
# Test name detection
test_\w*(parity|crossval|reference|accuracy)

# Accuracy patterns
cosine|similarity|threshold|AC\d+

# C++ references
bitnet\.cpp|llama\.cpp|reference
```

#### Category 9: Flaky/Non-Deterministic (3 tests)

**Pattern Indicators**:
- Comments with `FLAKY`, `non-deterministic`, `race condition`
- Tests with retry logic or timeout handling
- Platform-specific failures (WSL2, macOS)

**Reason Template**:
```rust
#[ignore = "FLAKY: <symptom> - <investigation status>"]
```

**Examples**:
```rust
// Good
#[ignore = "FLAKY: CUDA context cleanup issue - repro rate 10% in rapid consecutive runs"]
#[ignore = "FLAKY: memory tracking platform-specific (WSL2/Linux)"]
#[ignore = "FLAKY: timing-dependent test, investigating timeout issue"]

// Bad
#[ignore = "flaky"]
#[ignore = "sometimes fails"]
```

**Regex Patterns**:
```regex
# Flaky indicators
(?i)(flaky|non-deterministic|race|timeout|intermittent)

# Platform references
(?i)(wsl2|macos|linux-specific|platform)
```

### Detection Script Specification

**File**: `scripts/check-ignore-hygiene.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Color output helpers
red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
blue()   { printf "\033[34m%s\033[0m\n" "$*"; }

# Configuration
TAXONOMY_FILE="${TAXONOMY_FILE:-scripts/ignore-taxonomy.json}"
MODE="${MODE:-full}"  # full | diff | suggest | enforce
DRY_RUN="${DRY_RUN:-true}"
FAIL_ON_BARE="${FAIL_ON_BARE:-false}"
MAX_BARE_PERCENT="${MAX_BARE_PERCENT:-5}"

# Output formatting
print_header() {
    echo ""
    blue "═══════════════════════════════════════════════════════"
    blue "$1"
    blue "═══════════════════════════════════════════════════════"
    echo ""
}

# Count functions
count_total_ignores() {
    rg '#\[ignore' --type rust crates/ tests/ xtask/ -c 2>/dev/null | \
        awk -F: '{sum += $2} END {print sum+0}'
}

count_bare_ignores() {
    rg '#\[ignore\]' --type rust crates/ tests/ xtask/ -c 2>/dev/null | \
        awk -F: '{sum += $2} END {print sum+0}'
}

count_annotated_ignores() {
    rg '#\[ignore\s*=\s*"' --type rust crates/ tests/ xtask/ -c 2>/dev/null | \
        awk -F: '{sum += $2} END {print sum+0}'
}

# Detection functions
find_bare_ignores() {
    # Find all bare #[ignore] without reasons
    rg '#\[ignore\]' --type rust crates/ tests/ xtask/ -n --color never 2>/dev/null || true
}

extract_context() {
    local file="$1"
    local line="$2"
    local context_lines=10

    # Extract surrounding context for categorization
    sed -n "$((line - context_lines)),$((line + 5))p" "$file" 2>/dev/null || echo ""
}

# Categorization engine
categorize_ignore() {
    local file="$1"
    local line="$2"
    local context="$3"

    local categories=()
    local confidence=0
    local suggested_reason=""

    # Issue-blocked detection (highest priority)
    if echo "$context" | grep -qE '(?i)issue\s*#?(\d+)'; then
        local issue_num=$(echo "$context" | grep -oE 'issue\s*#?(\d+)' -i | head -1 | grep -oE '\d+')
        categories+=("issue-blocked")
        confidence=$((confidence + 30))
        suggested_reason="Issue #${issue_num}: "
    fi

    # Slow/performance detection
    if echo "$context" | grep -qE '(?i)(slow|performance|benchmark|timing)'; then
        categories+=("slow")
        confidence=$((confidence + 20))
        suggested_reason="${suggested_reason}slow: "
    fi

    # Model/fixture detection
    if echo "$context" | grep -qE '(BITNET_GGUF|\.gguf|models?/|fixture)'; then
        categories+=("requires-model")
        confidence=$((confidence + 20))
        suggested_reason="${suggested_reason}requires: "
    fi

    # GPU detection
    if echo "$file" | grep -qE 'gpu_' || echo "$context" | grep -qE '(?i)(gpu|cuda|device)'; then
        categories+=("gpu")
        confidence=$((confidence + 25))
        suggested_reason="${suggested_reason}gpu: "
    fi

    # Network detection
    if echo "$context" | grep -qE '(?i)(network|download|fetch|api|huggingface)'; then
        categories+=("network")
        confidence=$((confidence + 20))
        suggested_reason="${suggested_reason}network: "
    fi

    # TODO/placeholder detection
    if echo "$context" | grep -qE '(?i)(todo|fixme|wip|placeholder|unimplemented)'; then
        categories+=("todo")
        confidence=$((confidence + 15))
        suggested_reason="${suggested_reason}TODO: "
    fi

    # Quantization detection
    if echo "$context" | grep -qE '(?i)(i2s|tl1|tl2|qk256|quantization)'; then
        categories+=("quantization")
        confidence=$((confidence + 20))
        suggested_reason="${suggested_reason}quantization: "
    fi

    # Parity/crossval detection
    if echo "$context" | grep -qE '(?i)(parity|crossval|reference|accuracy)'; then
        categories+=("parity")
        confidence=$((confidence + 20))
        suggested_reason="${suggested_reason}parity: "
    fi

    # Flaky detection
    if echo "$context" | grep -qE '(?i)(flaky|non-deterministic|race|timeout|intermittent)'; then
        categories+=("flaky")
        confidence=$((confidence + 25))
        suggested_reason="FLAKY: "
    fi

    # Default to unknown if no matches
    if [ ${#categories[@]} -eq 0 ]; then
        categories+=("unknown")
        confidence=0
        suggested_reason="FIXME: add reason - "
    fi

    # Emit result
    echo "${file}:${line}|$(IFS=,; echo "${categories[*]}")|${confidence}|${suggested_reason}"
}

# Suggestion generator
suggest_annotation() {
    local file="$1"
    local line="$2"
    local categories="$3"
    local confidence="$4"
    local suggested_reason="$5"

    yellow "  📍 ${file}:${line}"
    yellow "     Categories: ${categories}"
    yellow "     Confidence: ${confidence}%"
    yellow "     Suggested: #[ignore = \"${suggested_reason}...\"]"
    echo ""
}

# Main workflow
main() {
    print_header "BitNet-rs #[ignore] Hygiene Check"

    # Gather statistics
    local total=$(count_total_ignores)
    local bare=$(count_bare_ignores)
    local annotated=$(count_annotated_ignores)
    local bare_percent=0

    if [ "$total" -gt 0 ]; then
        bare_percent=$((bare * 100 / total))
    fi

    echo "Total #[ignore] annotations: ${total}"
    echo "Annotated (with reason):     ${annotated} ($((annotated * 100 / total))%)"
    echo "Bare (no reason):            ${bare} (${bare_percent}%)"
    echo ""

    # Check threshold
    if [ "$bare_percent" -gt "$MAX_BARE_PERCENT" ]; then
        red "❌ Bare ignore percentage (${bare_percent}%) exceeds threshold (${MAX_BARE_PERCENT}%)"
        if [ "$FAIL_ON_BARE" = "true" ]; then
            echo ""
            red "FAILED: Fix bare #[ignore] annotations before merging"
            exit 1
        fi
    else
        green "✅ Bare ignore percentage (${bare_percent}%) within threshold (${MAX_BARE_PERCENT}%)"
    fi

    # Mode-specific execution
    case "$MODE" in
        full)
            print_header "Full Scan: All Bare Ignores"
            scan_and_categorize
            ;;
        diff)
            print_header "Diff Mode: PR Changes Only"
            scan_diff
            ;;
        suggest)
            print_header "Suggestion Mode: Auto-Annotation Preview"
            generate_suggestions
            ;;
        enforce)
            print_header "Enforce Mode: CI Guard"
            enforce_hygiene
            ;;
        *)
            red "Unknown mode: $MODE"
            exit 1
            ;;
    esac
}

scan_and_categorize() {
    local count=0

    while IFS=: read -r file line _; do
        count=$((count + 1))
        local context=$(extract_context "$file" "$line")
        local result=$(categorize_ignore "$file" "$line" "$context")

        # Parse result
        IFS='|' read -r location categories confidence suggested_reason <<< "$result"

        suggest_annotation "$file" "$line" "$categories" "$confidence" "$suggested_reason"
    done < <(find_bare_ignores)

    echo ""
    if [ "$count" -gt 0 ]; then
        yellow "Found ${count} bare #[ignore] annotations requiring attention"
    else
        green "✅ All #[ignore] annotations have explicit reasons!"
    fi
}

scan_diff() {
    # Check only lines changed in current git diff
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        red "Not in a git repository - diff mode unavailable"
        exit 1
    fi

    local new_bare_ignores=0

    # Get diff for Rust files
    while IFS=: read -r file line content; do
        # Check if line contains bare #[ignore]
        if echo "$content" | grep -qE '^\+.*#\[ignore\]$'; then
            new_bare_ignores=$((new_bare_ignores + 1))

            red "❌ New bare #[ignore] found:"
            red "   ${file}:${line}"
            echo ""

            # Suggest annotation
            local context=$(extract_context "$file" "$line")
            local result=$(categorize_ignore "$file" "$line" "$context")
            IFS='|' read -r location categories confidence suggested_reason <<< "$result"

            yellow "   Quick-fix suggestion:"
            yellow "   #[ignore = \"${suggested_reason}...\"]"
            echo ""
        fi
    done < <(git diff HEAD --unified=0 --diff-filter=AM -- '*.rs' | grep -E '^\+.*#\[ignore\]')

    if [ "$new_bare_ignores" -gt 0 ]; then
        red "❌ Found ${new_bare_ignores} new bare #[ignore] annotation(s) in this PR"
        echo ""
        red "Please add explicit reasons using the taxonomy:"
        echo "  - Issue #NNN: <description>     (for blocked tests)"
        echo "  - slow: <reason>                 (for performance tests)"
        echo "  - requires: <resource>           (for external dependencies)"
        echo "  - gpu: <requirement>             (for GPU-specific tests)"
        echo "  - network: <dependency>          (for network tests)"
        echo "  - TODO: <task>                   (for placeholders)"
        echo ""
        exit 1
    else
        green "✅ No new bare #[ignore] annotations in this PR"
    fi
}

generate_suggestions() {
    # Generate suggested annotations in batch
    local output_file="ignore-suggestions.txt"

    echo "# Auto-Generated #[ignore] Annotation Suggestions" > "$output_file"
    echo "# Generated: $(date)" >> "$output_file"
    echo "" >> "$output_file"

    while IFS=: read -r file line _; do
        local context=$(extract_context "$file" "$line")
        local result=$(categorize_ignore "$file" "$line" "$context")

        IFS='|' read -r location categories confidence suggested_reason <<< "$result"

        echo "# ${file}:${line}" >> "$output_file"
        echo "# Categories: ${categories} (confidence: ${confidence}%)" >> "$output_file"
        echo "#[ignore = \"${suggested_reason}...\"]" >> "$output_file"
        echo "" >> "$output_file"
    done < <(find_bare_ignores)

    green "✅ Suggestions written to ${output_file}"
}

enforce_hygiene() {
    # CI mode: strict enforcement
    export FAIL_ON_BARE=true
    export MODE=diff

    scan_diff
}

# Execute main workflow
main "$@"
```

**Key Features**:
1. **Multi-mode operation**: full scan, diff mode (PR changes only), suggestion generation, CI enforcement
2. **Context extraction**: Analyzes surrounding code and comments for categorization
3. **Confidence scoring**: Provides transparency on categorization quality
4. **Incremental migration**: Supports threshold-based enforcement (5% bare ignores)
5. **CI integration**: Fails PR builds on new bare annotations

### Taxonomy Configuration File

**File**: `scripts/ignore-taxonomy.json`

```json
{
  "version": "1.0.0",
  "categories": [
    {
      "id": "issue-blocked",
      "priority": 100,
      "patterns": {
        "file": ["issue_(\\d+)_"],
        "test_name": ["test_issue_(\\d+)"],
        "comment": ["(?i)issue\\s*#?(\\d+)"]
      },
      "template": "Issue #{{issue_number}}: {{description}}",
      "examples": [
        "Issue #254: shape mismatch in layer-norm - needs investigation",
        "Issue #260: TDD placeholder - quantized_matmul not yet implemented"
      ]
    },
    {
      "id": "slow",
      "priority": 80,
      "patterns": {
        "test_name": ["test_\\w*(slow|perf|benchmark|integration)"],
        "comment": ["(?i)(slow|performance|benchmark|timing)", "(\\d+)\\+?\\s*tokens?"]
      },
      "template": "slow: {{description}}, see {{alternative}}",
      "examples": [
        "slow: 50+ token generations, use fast unit tests in ci mode",
        "slow: integration test, see faster equivalents in unit_tests.rs"
      ]
    },
    {
      "id": "requires-model",
      "priority": 75,
      "patterns": {
        "test_name": ["test_\\w*(model|gguf|weight|loading|fixture)"],
        "comment": ["BITNET_GGUF|BITNET_CPP_DIR", "\\.gguf|\\.safetensors|models/"]
      },
      "template": "requires: {{resource}}",
      "examples": [
        "requires: real GGUF model with complete metadata",
        "requires: bitnet.cpp FFI implementation",
        "fixture: missing test data at tests/fixtures/tokenizer.gguf"
      ]
    },
    {
      "id": "gpu",
      "priority": 85,
      "patterns": {
        "file": ["gpu_\\w+\\.rs"],
        "test_name": ["test_\\w*(gpu|cuda|device)"],
        "feature_gate": ["#\\[cfg\\(.*feature\\s*=\\s*\"gpu\""]
      },
      "template": "gpu: {{requirement}}",
      "examples": [
        "gpu: requires CUDA toolkit installed",
        "gpu: run with --ignored flag when CUDA available"
      ]
    },
    {
      "id": "network",
      "priority": 70,
      "patterns": {
        "test_name": ["test_\\w*(download|fetch|network|api|http)"],
        "comment": ["(?i)(huggingface|hf_token|github|api)", "async\\s+fn\\s+test"]
      },
      "template": "network: {{dependency}}",
      "examples": [
        "network: requires HuggingFace API access",
        "network: requires internet connection and HF_TOKEN"
      ]
    },
    {
      "id": "todo",
      "priority": 60,
      "patterns": {
        "comment": ["(?i)(todo|fixme|wip|placeholder)", "unimplemented!\\(\\)|todo!\\(\\)", "after\\s+#\\d+|pending|future"]
      },
      "template": "TODO: {{task}}",
      "examples": [
        "TODO: update to use QuantizedLinear::new_tl1() with proper LookupTable",
        "TODO: implement GPU mixed-precision tests after #439 resolution"
      ]
    },
    {
      "id": "quantization",
      "priority": 75,
      "patterns": {
        "test_name": ["(?i)(i2s|tl1|tl2|qk256|iq2_s)"],
        "comment": ["(?i)(avx2|avx512|neon|simd)"],
        "file": ["(quantization|kernels)/"]
      },
      "template": "quantization: {{format}} - {{reason}}",
      "examples": [
        "quantization: I2S SIMD consistency tests need refinement",
        "quantization: QK256 property-based test blocked by Issue #159"
      ]
    },
    {
      "id": "parity",
      "priority": 80,
      "patterns": {
        "test_name": ["test_\\w*(parity|crossval|reference|accuracy)"],
        "comment": ["cosine|similarity|threshold|AC\\d+", "bitnet\\.cpp|llama\\.cpp|reference"]
      },
      "template": "parity: {{target}} - {{requirement}}",
      "examples": [
        "parity: TDD Red phase - AC5 accuracy thresholds not yet met (Issue #254)",
        "parity: C++ reference comparison - requires bitnet.cpp setup"
      ]
    },
    {
      "id": "flaky",
      "priority": 90,
      "patterns": {
        "comment": ["(?i)(flaky|non-deterministic|race|timeout|intermittent)", "(?i)(wsl2|macos|linux-specific|platform)"]
      },
      "template": "FLAKY: {{symptom}} - {{status}}",
      "examples": [
        "FLAKY: CUDA context cleanup issue - repro rate 10% in rapid consecutive runs",
        "FLAKY: memory tracking platform-specific (WSL2/Linux)"
      ]
    }
  ],
  "thresholds": {
    "max_bare_percent": 5,
    "confidence_threshold": 70
  }
}
```

### CI Integration Specification

**File**: `.github/workflows/ci.yml` (add new job)

```yaml
# Add to existing CI workflow
jobs:
  # ... existing jobs ...

  # Check #[ignore] annotation hygiene
  ignore-hygiene:
    name: Check Test Ignore Annotations
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Need full history for git diff

      - name: Install ripgrep
        run: |
          sudo apt-get update
          sudo apt-get install -y ripgrep

      - name: Check ignore annotation hygiene (diff mode)
        run: |
          chmod +x scripts/check-ignore-hygiene.sh
          MODE=diff FAIL_ON_BARE=true ./scripts/check-ignore-hygiene.sh
        continue-on-error: ${{ github.event_name == 'pull_request' && contains(github.event.pull_request.labels.*.name, 'ignore-migration') }}

      - name: Generate statistics
        if: always()
        run: |
          MODE=full ./scripts/check-ignore-hygiene.sh > ignore-stats.txt
          cat ignore-stats.txt

      - name: Upload statistics
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: ignore-hygiene-stats
          path: ignore-stats.txt
```

**CI Behavior**:
1. **PR validation**: Fails on new bare `#[ignore]` annotations
2. **Exemption mechanism**: Allow PRs with `ignore-migration` label to introduce bare ignores temporarily
3. **Statistics tracking**: Upload metrics for trend analysis
4. **Quick-fix output**: Provides actionable suggestions in CI logs

### Auto-Annotation Tool Specification

**File**: `scripts/auto-annotate-ignores.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Auto-annotation tool for bulk migration
# Usage: ./scripts/auto-annotate-ignores.sh [--dry-run] [--file FILE]

DRY_RUN="${DRY_RUN:-true}"
TARGET_FILE="${TARGET_FILE:-}"

if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi

if [[ "$1" == "--file" ]]; then
    TARGET_FILE="$2"
    shift 2
fi

# Source categorization engine
source scripts/check-ignore-hygiene.sh

annotate_file() {
    local file="$1"
    local temp_file="${file}.ignore-tmp"

    echo "Processing: $file"

    # Find bare ignores and annotate
    local modified=false

    while IFS=: read -r _ line content; do
        if echo "$content" | grep -qE '#\[ignore\]$'; then
            modified=true

            # Get suggested annotation
            local context=$(extract_context "$file" "$line")
            local result=$(categorize_ignore "$file" "$line" "$context")
            IFS='|' read -r location categories confidence suggested_reason <<< "$result"

            # Replace line if confidence > 70%
            if [ "$confidence" -ge 70 ]; then
                echo "  Line ${line}: ${suggested_reason}... (confidence: ${confidence}%)"

                if [ "$DRY_RUN" = "false" ]; then
                    # Perform replacement
                    sed -i "${line}s|#\[ignore\]|#[ignore = \"${suggested_reason}...\"]|" "$file"
                fi
            else
                echo "  Line ${line}: LOW CONFIDENCE (${confidence}%) - manual review needed"
            fi
        fi
    done < <(grep -n '#\[ignore\]' "$file" || true)

    if [ "$modified" = "true" ] && [ "$DRY_RUN" = "false" ]; then
        # Format with rustfmt
        rustfmt "$file" 2>/dev/null || true
        echo "  ✅ Annotated and formatted"
    fi
}

# Main execution
if [ -n "$TARGET_FILE" ]; then
    annotate_file "$TARGET_FILE"
else
    # Process all files with bare ignores
    while IFS=: read -r file _; do
        annotate_file "$file"
    done < <(find_bare_ignores | awk -F: '{print $1}' | sort -u)
fi

if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "DRY RUN complete. Run with DRY_RUN=false to apply changes."
fi
```

## Migration Plan

### Phase 1: High-Impact Files (Week 1)

**Target**: Issue-blocked tests (46 tests)

**Priority Files**:
1. `crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs` (10 bare)
2. `crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs` (9 bare)
3. `crates/bitnet-inference/tests/neural_network_test_scaffolding.rs` (8 bare)

**Approach**:
```bash
# 1. Generate suggestions
MODE=suggest ./scripts/check-ignore-hygiene.sh

# 2. Review suggestions manually
vim ignore-suggestions.txt

# 3. Apply annotations to specific file
DRY_RUN=false TARGET_FILE=crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs \
  ./scripts/auto-annotate-ignores.sh

# 4. Manual review and refinement
git diff crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs

# 5. Commit changes
git add crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs
git commit -m "docs(tests): annotate #[ignore] in issue_254 tests with Issue #254 references"
```

**Acceptance Criteria**:
- All issue-blocked tests reference specific GitHub issue numbers
- Descriptions include blocker summary
- Tests compile and run correctly

### Phase 2: Performance/Slow Tests (Week 2)

**Target**: Slow/performance tests (17 tests)

**Priority Files**:
1. `crates/bitnet-inference/tests/ac3_autoregressive_generation.rs` (6 bare)
2. `crates/bitnet-tokenizers/tests/tokenization_smoke.rs` (6 bare)

**Approach**:
```bash
# 1. Identify slow tests
rg -t rust -n '#\[ignore\]' | xargs -I {} grep -B 5 -A 5 'slow\|perf\|50.*token' {}

# 2. Auto-annotate with "slow:" prefix
MODE=suggest ./scripts/check-ignore-hygiene.sh | grep 'slow:' > slow-suggestions.txt

# 3. Apply bulk changes
for file in $(rg -l '#\[ignore\].*slow' crates/); do
  DRY_RUN=false TARGET_FILE="$file" ./scripts/auto-annotate-ignores.sh
done

# 4. Verify tests still skip correctly
cargo test --workspace --no-default-features --features cpu -- --ignored
```

**Acceptance Criteria**:
- All slow tests include runtime description
- Reference to faster alternative test (if available)
- Clear reason for ignoring in CI

### Phase 3: Model/GPU/Network Tests (Week 3)

**Target**: External dependency tests (42 tests)

**Categories**:
- Model/fixture requirements: 29 tests
- GPU-specific: 13 tests
- Network-dependent: 10 tests

**Approach**:
```bash
# 1. Batch process by category
for category in requires-model gpu network; do
  echo "Processing category: $category"

  # Find matching tests
  MODE=full ./scripts/check-ignore-hygiene.sh | grep -A 3 "Categories.*$category"

  # Auto-annotate high-confidence matches
  # (Manual review for confidence < 80%)
done

# 2. Test with environment variables
BITNET_GGUF=models/test.gguf cargo test --workspace -- --ignored
cargo test --workspace --no-default-features --features gpu -- --ignored
```

**Acceptance Criteria**:
- Model tests reference specific GGUF requirements or `BITNET_GGUF`
- GPU tests indicate CUDA toolkit requirement
- Network tests specify external service dependency

### Phase 4: Placeholders and Edge Cases (Week 4)

**Target**: Remaining tests (30 tests)

**Categories**:
- TODO/feature placeholders: 14 tests
- Quantization/kernel tests: 22 tests (overlap with other categories)
- Cross-validation: 7 tests
- Flaky tests: 3 tests

**Approach**:
```bash
# 1. Manual review for low-confidence categorizations
MODE=suggest ./scripts/check-ignore-hygiene.sh | grep -B 2 'confidence: [0-6]'

# 2. Apply conservative annotations
# Use "FIXME: add reason" for unclear cases

# 3. Create tracking issues for ambiguous tests
# Example: "FIXME: categorize test purpose (see Issue #XXX)"

# 4. Final verification
total=$(count_total_ignores)
bare=$(count_bare_ignores)
bare_percent=$((bare * 100 / total))

if [ "$bare_percent" -le 5 ]; then
  echo "✅ Migration complete: ${bare_percent}% bare ignores"
else
  echo "⚠️  Still ${bare_percent}% bare ignores - continue migration"
fi
```

**Acceptance Criteria**:
- <5% bare ignores (≤10 out of 194)
- All remaining bare ignores have tracking issues
- CI enforcement enabled (fails on new bare ignores)

### Phase 5: CI Enforcement (Week 5)

**Target**: Enable CI guard job

**Approach**:
```bash
# 1. Add CI job to workflow
git add .github/workflows/ci.yml
git commit -m "ci: add ignore annotation hygiene check"

# 2. Test CI job locally
MODE=diff FAIL_ON_BARE=true ./scripts/check-ignore-hygiene.sh

# 3. Create documentation
cat > docs/development/ignore-annotation-guide.md <<EOF
# #[ignore] Annotation Guide

When marking a test with #[ignore], always provide an explicit reason using the taxonomy:

## Taxonomy

- Issue #NNN: <blocker description>      (for tests blocked by GitHub issues)
- slow: <runtime>, see <alternative>     (for performance/integration tests)
- requires: <resource>                   (for external dependencies)
- gpu: <requirement>                     (for GPU-specific tests)
- network: <dependency>                  (for network tests)
- TODO: <task>                           (for placeholders)
- quantization: <format> - <reason>      (for quantization tests)
- parity: <target> - <requirement>       (for cross-validation)
- FLAKY: <symptom> - <status>            (for non-deterministic tests)

## Examples

\`\`\`rust
// Good
#[ignore = "Issue #254: shape mismatch in layer-norm - needs investigation"]
#[ignore = "slow: 50+ token generations, use fast unit tests in ci mode"]
#[ignore = "requires: real GGUF model with complete metadata"]
#[ignore = "gpu: requires CUDA toolkit installed"]

// Bad
#[ignore]
#[ignore = "broken"]
#[ignore = "TODO"]
\`\`\`

## CI Enforcement

The CI pipeline fails on new bare #[ignore] annotations. Use:

\`\`\`bash
MODE=diff ./scripts/check-ignore-hygiene.sh
\`\`\`

to check your PR before pushing.
EOF

git add docs/development/ignore-annotation-guide.md
git commit -m "docs: add #[ignore] annotation guide"
```

**Acceptance Criteria**:
- CI job runs on all PRs
- Fails on new bare `#[ignore]` annotations
- Provides quick-fix suggestions in CI output
- Documentation updated with annotation guide

## Risk Mitigation

### Risk 1: False Positive Categorization

**Impact**: Auto-annotation suggests incorrect reasons, requiring manual correction

**Mitigation**:
1. **Confidence scoring**: Only auto-apply annotations with confidence ≥70%
2. **Dry-run mode**: Preview all changes before applying
3. **Manual review**: High-priority files reviewed manually before bulk changes
4. **Validation**: Tests still compile and skip correctly after annotation

**Fallback**: Use `#[ignore = "FIXME: add reason - <auto-detected category>"]` for low-confidence matches

### Risk 2: CI False Negatives

**Impact**: CI fails to detect bare ignores in edge cases

**Mitigation**:
1. **Comprehensive regex patterns**: Test against known examples
2. **Multiple detection passes**: Check file paths, test names, and comments
3. **Incremental rollout**: Enable CI enforcement after Phase 4 completion
4. **Exemption mechanism**: Use `ignore-migration` label for legitimate bare ignores during migration

**Validation**:
```bash
# Test CI detection against known examples
echo '#[ignore]' > test_file.rs
MODE=diff ./scripts/check-ignore-hygiene.sh
# Should fail with exit code 1
```

### Risk 3: Performance Regression

**Impact**: Detection script slows down CI pipeline

**Mitigation**:
1. **Optimize ripgrep queries**: Use `-c` for counts, avoid excessive context extraction
2. **Diff mode optimization**: Only scan changed files in PRs
3. **Caching**: Cache taxonomy patterns between runs
4. **Timeout protection**: Set 60-second timeout for CI job

**Performance Targets**:
- Full scan: <5 seconds
- Diff mode: <10 seconds
- CI overhead: <30 seconds

**Validation**:
```bash
time MODE=full ./scripts/check-ignore-hygiene.sh
# Should complete in <5 seconds
```

### Risk 4: Developer Friction

**Impact**: Developers resist new enforcement, bypass with workarounds

**Mitigation**:
1. **Clear documentation**: Provide taxonomy and examples
2. **Helpful error messages**: Include quick-fix suggestions in CI output
3. **Incremental rollout**: Warn before enforcing
4. **Developer tooling**: Provide auto-annotation script for bulk changes
5. **Grandfathering**: Existing bare ignores not blocked, only new ones

**Communication**:
- Add to `CONTRIBUTING.md`
- Update `docs/development/test-suite.md`
- Create `docs/development/ignore-annotation-guide.md`
- Announce in team channels before CI enforcement

## Success Criteria

### Quantitative Metrics

1. **Annotation Coverage**:
   - **Target**: <5% bare ignores (≤10 out of 194)
   - **Current**: 69.6% bare (135 out of 194)
   - **Measurement**: `count_bare_ignores() / count_total_ignores() * 100`

2. **Categorization Accuracy**:
   - **Target**: ≥90% correct category assignment
   - **Validation**: Manual review of 50 random samples
   - **Measurement**: Correct categorizations / Total samples * 100

3. **CI Stability**:
   - **Target**: Zero false positives in CI enforcement
   - **Validation**: Run CI job on 10 recent PRs
   - **Measurement**: False positives / Total PRs * 100

4. **Performance**:
   - **Target**: Full scan <5 seconds, diff mode <10 seconds
   - **Validation**: `time MODE=full ./scripts/check-ignore-hygiene.sh`
   - **Measurement**: Execution time in seconds

### Qualitative Metrics

1. **Developer Experience**:
   - Quick-fix suggestions are actionable
   - Documentation is clear and comprehensive
   - Auto-annotation tool works reliably

2. **Maintainability**:
   - Taxonomy is extensible without code changes
   - Pattern matching is transparent and debuggable
   - CI integration is stable and reliable

3. **Test Clarity**:
   - Ignored tests have clear, actionable reasons
   - Issue references are accurate and up-to-date
   - Alternative tests are documented for slow tests

## Validation Commands

### Detection Accuracy

```bash
# Full scan with statistics
MODE=full ./scripts/check-ignore-hygiene.sh

# Expected output:
# Total #[ignore] annotations: 194
# Annotated (with reason):     184 (94.8%)
# Bare (no reason):            10 (5.2%)
# ✅ Bare ignore percentage (5.2%) within threshold (5%)
```

### CI Enforcement

```bash
# Simulate PR validation
git checkout -b test-ignore-ci
echo '#[ignore]' >> crates/bitnet-inference/tests/test_example.rs
git add -A && git commit -m "test: bare ignore"
MODE=diff FAIL_ON_BARE=true ./scripts/check-ignore-hygiene.sh

# Expected: Exit code 1 with quick-fix suggestion
```

### Auto-Annotation

```bash
# Dry-run mode
DRY_RUN=true TARGET_FILE=crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs \
  ./scripts/auto-annotate-ignores.sh

# Expected: Preview of suggested annotations without file modification

# Apply mode
DRY_RUN=false TARGET_FILE=crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs \
  ./scripts/auto-annotate-ignores.sh

# Validate changes
git diff crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs
cargo test -p bitnet-inference --test issue_254_ac3_deterministic_generation -- --ignored
```

### Performance Validation

```bash
# Measure detection performance
hyperfine --warmup 3 'MODE=full ./scripts/check-ignore-hygiene.sh'

# Expected: Mean execution time <5 seconds

# Measure CI overhead
time MODE=diff ./scripts/check-ignore-hygiene.sh

# Expected: Execution time <10 seconds
```

## Alignment with BitNet-rs Principles

### TDD Practices

**Alignment**:
- Detection script has comprehensive test coverage
- Auto-annotation validated against known test cases
- CI enforcement tested before production rollout

**Validation**:
```bash
# Test detection logic
cargo test -p scripts --test check-ignore-hygiene

# Test CI integration
.github/workflows/test-ignore-hygiene.yml
```

### Feature-Gated Architecture

**Alignment**:
- Detection script uses no optional dependencies
- Works with both `--features cpu` and `--features gpu` test suites
- No feature-specific ignore patterns

**Validation**:
```bash
# Test across feature combinations
cargo test --workspace --no-default-features --features cpu -- --ignored
cargo test --workspace --no-default-features --features gpu -- --ignored
```

### Workspace Structure

**Alignment**:
- Script respects workspace boundaries (`crates/`, `tests/`, `xtask/`)
- No cross-crate ignore pattern leakage
- Taxonomy applies uniformly across workspace

**Validation**:
```bash
# Verify workspace coverage
MODE=full ./scripts/check-ignore-hygiene.sh | \
  grep -oE 'crates/[^/]+' | sort -u

# Expected: All test-containing crates listed
```

### Cross-Platform Support

**Alignment**:
- Script uses POSIX-compliant bash
- Ripgrep available on Linux, macOS, Windows (via CI)
- No platform-specific regex patterns

**Validation**:
```bash
# Test on multiple platforms
docker run --rm -v $(pwd):/work -w /work ubuntu:latest bash scripts/check-ignore-hygiene.sh
docker run --rm -v $(pwd):/work -w /work alpine:latest bash scripts/check-ignore-hygiene.sh
```

## References

### Existing Patterns

1. **`scripts/hooks/banned-patterns.sh`**: Inspiration for regex-based code quality checks
2. **`scripts/verify-tests.sh`**: Pattern for test discovery and execution
3. **`.github/workflows/ci.yml`**: CI integration patterns
4. **`docs/development/test-suite.md`**: Test infrastructure documentation

### Exploration Reports

1. **`IGNORE_TESTS_AUDIT_DETAILED.md`**: Comprehensive analysis of 194 #[ignore] annotations
2. **`IGNORE_TESTS_QUICK_REFERENCE.md`**: Quick reference for annotation taxonomy

### Related Documentation

1. **`docs/development/test-suite.md`**: Testing framework overview
2. **`docs/development/validation-ci.md`**: CI validation patterns
3. **`CLAUDE.md`**: Project conventions and testing philosophy

## Appendix: Migration Checklist

**Phase 1: High-Impact Files (Week 1)**
- [ ] Generate suggestions for issue-blocked tests (46 tests)
- [ ] Manual review of `issue_254_ac3_deterministic_generation.rs` (10 tests)
- [ ] Auto-annotate `gguf_weight_loading_property_tests.rs` (9 tests)
- [ ] Validate tests still skip correctly
- [ ] Commit changes with descriptive message

**Phase 2: Performance/Slow Tests (Week 2)**
- [ ] Identify slow tests with token count analysis
- [ ] Add "slow:" prefix with runtime descriptions
- [ ] Reference faster alternative tests where available
- [ ] Verify CI skip behavior

**Phase 3: Model/GPU/Network Tests (Week 3)**
- [ ] Batch process model/fixture tests (29 tests)
- [ ] Annotate GPU tests with CUDA requirements (13 tests)
- [ ] Mark network tests with service dependencies (10 tests)
- [ ] Test with environment variables (`BITNET_GGUF`, etc.)

**Phase 4: Placeholders and Edge Cases (Week 4)**
- [ ] Manual review of low-confidence categorizations
- [ ] Apply conservative "FIXME:" annotations
- [ ] Create tracking issues for ambiguous tests
- [ ] Final verification: <5% bare ignores

**Phase 5: CI Enforcement (Week 5)**
- [ ] Add CI job to `.github/workflows/ci.yml`
- [ ] Test CI job locally with `MODE=diff`
- [ ] Create `docs/development/ignore-annotation-guide.md`
- [ ] Update `CONTRIBUTING.md` and `docs/development/test-suite.md`
- [ ] Enable CI enforcement (remove `continue-on-error`)
- [ ] Monitor first week of enforcement for false positives

**Continuous Improvement**
- [ ] Monthly review of annotation quality
- [ ] Update taxonomy based on new test patterns
- [ ] Track metrics (bare ignore percentage, CI stability)
- [ ] Refine categorization accuracy
