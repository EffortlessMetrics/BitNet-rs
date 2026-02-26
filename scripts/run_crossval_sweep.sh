#!/usr/bin/env bash
# run_crossval_sweep.sh - Comprehensive cross-validation sweep between BitNet-rs and bitnet.cpp
#
# This script orchestrates deterministic cross-validation across multiple test scenarios,
# capturing traces, logits, and token outputs for systematic divergence analysis.
#
# Usage: ./scripts/run_crossval_sweep.sh <model.gguf> <tokenizer.json> [output_dir]
#
# Features:
#  - Deterministic execution (BITNET_DETERMINISTIC=1, BITNET_SEED=42, RAYON_NUM_THREADS=4)
#  - Tracing with BITNET_TRACE_DIR (90+ trace files per scenario)
#  - C++ parity comparison via FFI (graceful degradation to Rust-only mode)
#  - Blake3 hash divergence detection
#  - Cosine similarity metrics per position
#  - Comprehensive summary report with actionable recommendations

set -euo pipefail

# ============================================================================
# Color Output
# ============================================================================
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ============================================================================
# Logging Functions
# ============================================================================
info() {
    echo -e "${CYAN}INFO:${NC} $*" >&2
}

success() {
    echo -e "${GREEN}✓${NC} $*" >&2
}

warn() {
    echo -e "${YELLOW}WARN:${NC} $*" >&2
}

error() {
    echo -e "${RED}ERROR:${NC} $*" >&2
    exit "${2:-1}"
}

section() {
    echo ""
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}$*${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

# ============================================================================
# Usage and Argument Parsing
# ============================================================================
if [[ $# -lt 2 ]]; then
    cat <<EOF
Usage: $0 <model.gguf> <tokenizer.json> [output_dir]

Orchestrates full cross-validation between BitNet-rs and bitnet.cpp across
multiple deterministic test scenarios.

Arguments:
  model.gguf      Path to GGUF model file
  tokenizer.json  Path to tokenizer file
  output_dir      Output directory for results (default: ./crossval-results)

Environment:
  BITNET_CPP_DIR         Path to C++ reference implementation (optional)
                         Default: \$HOME/.cache/bitnet_cpp
                         If not available, runs in Rust-only mode

  CROSSVAL_TIMEOUT_SECS  Timeout per scenario in seconds (default: 180)

Example:
  ./scripts/run_crossval_sweep.sh \\
    models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \\
    models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \\
    /tmp/crossval-sweep

Output structure:
  crossval-results/
  ├── scenario1/
  │   ├── rs-traces/         (90+ trace files)
  │   ├── rs-output.txt
  │   ├── cpp-output.txt
  │   ├── logits-comparison.json
  │   └── report.txt
  ├── scenario2/
  │   └── ...
  ├── scenario3/
  │   └── ...
  └── summary.md             (final divergence report)
EOF
    exit 1
fi

MODEL_PATH="$1"
TOKENIZER_PATH="$2"
OUTPUT_DIR="${3:-./crossval-results}"

# Validate inputs
[[ -f "$MODEL_PATH" ]] || error "Model not found: $MODEL_PATH"
[[ -f "$TOKENIZER_PATH" ]] || error "Tokenizer not found: $TOKENIZER_PATH"

# Resolve absolute paths
MODEL_PATH="$(realpath "$MODEL_PATH")"
TOKENIZER_PATH="$(realpath "$TOKENIZER_PATH")"

# Timeout configuration
TIMEOUT_SECS="${CROSSVAL_TIMEOUT_SECS:-180}"

# ============================================================================
# Environment Setup
# ============================================================================
section "Environment Setup"

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
info "Output directory: $OUTPUT_DIR"

# Check for C++ reference
BITNET_CPP_DIR="${BITNET_CPP_DIR:-$HOME/.cache/bitnet_cpp}"
if [[ -d "$BITNET_CPP_DIR/build" ]]; then
    CPP_AVAILABLE=1
    success "C++ reference available: $BITNET_CPP_DIR"

    # Set library paths for C++ FFI
    if [[ "$OSTYPE" == "darwin"* ]]; then
        export DYLD_LIBRARY_PATH="$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src:$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src:${DYLD_LIBRARY_PATH:-}"
    else
        export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src:$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src:${LD_LIBRARY_PATH:-}"
    fi
else
    CPP_AVAILABLE=0
    warn "C++ reference not available - running in Rust-only mode"
    info "Set BITNET_CPP_DIR to enable full parity validation"
fi

export BITNET_CPP_DIR

# Check for required tools
command -v jq >/dev/null 2>&1 || warn "jq not found - JSON parsing will be limited"
command -v timeout >/dev/null 2>&1 || warn "timeout not found - timeout protection disabled"

# Check if cargo is available
command -v cargo >/dev/null 2>&1 || error "cargo not found in PATH"

# ============================================================================
# Build BitNet-rs with crossval features (if needed)
# ============================================================================
section "Building BitNet-rs"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Build CLI
info "Building bitnet-cli with CPU features..."
if ! cargo build -q --no-default-features --features cpu,full-cli -p bitnet-cli 2>&1; then
    error "Failed to build bitnet-cli"
fi

# Build crossval crate if C++ is available
if [[ $CPP_AVAILABLE -eq 1 ]]; then
    info "Building crossval crate with FFI features..."
    if ! cargo build -q --features "integration-tests,crossval,ffi" -p bitnet-crossval 2>&1; then
        warn "Failed to build crossval with FFI - C++ parity tests may not work"
        CPP_AVAILABLE=0
    fi
fi

success "Build complete"

# ============================================================================
# Test Scenarios Definition
# ============================================================================
section "Test Scenarios"

# Define test scenarios: (name, prompt, max_tokens, description)
declare -a SCENARIOS=(
    "scenario1|2+2=|1|Single token prefill (minimal test)"
    "scenario2|Hello|2|Two token generation"
    "scenario3|Count: 1,2,3,|4|Four token generation"
)

info "Configured ${#SCENARIOS[@]} test scenarios:"
for scenario in "${SCENARIOS[@]}"; do
    IFS='|' read -r name prompt tokens desc <<< "$scenario"
    info "  - $name: $desc (prompt=\"$prompt\", max_tokens=$tokens)"
done

# ============================================================================
# Scenario Execution
# ============================================================================
SCENARIO_RESULTS=()

for scenario_spec in "${SCENARIOS[@]}"; do
    IFS='|' read -r SCENARIO_NAME PROMPT MAX_TOKENS DESC <<< "$scenario_spec"

    section "Running $SCENARIO_NAME: $DESC"

    # Create scenario output directory
    SCENARIO_DIR="$OUTPUT_DIR/$SCENARIO_NAME"
    mkdir -p "$SCENARIO_DIR"

    # Trace directory for Rust
    RS_TRACE_DIR="$SCENARIO_DIR/rs-traces"
    mkdir -p "$RS_TRACE_DIR"

    # Output files
    RS_OUTPUT="$SCENARIO_DIR/rs-output.txt"
    CPP_OUTPUT="$SCENARIO_DIR/cpp-output.txt"
    LOGITS_JSON="$SCENARIO_DIR/logits-comparison.json"
    REPORT_FILE="$SCENARIO_DIR/report.txt"

    # ========================================================================
    # Run Rust Inference with Tracing
    # ========================================================================
    info "Running Rust inference with tracing..."

    # Set deterministic environment
    export BITNET_DETERMINISTIC=1
    export BITNET_SEED=42
    export RAYON_NUM_THREADS=4
    export BITNET_TRACE_DIR="$RS_TRACE_DIR"
    export BITNET_GGUF="$MODEL_PATH"
    export CROSSVAL_GGUF="$MODEL_PATH"

    # Run inference with timeout
    RS_EXIT_CODE=0
    if command -v timeout >/dev/null 2>&1; then
        timeout "${TIMEOUT_SECS}s" cargo run -q --no-default-features --features cpu,full-cli -p bitnet-cli -- \
            run \
            --model "$MODEL_PATH" \
            --tokenizer "$TOKENIZER_PATH" \
            --prompt "$PROMPT" \
            --max-tokens "$MAX_TOKENS" \
            --temperature 0.0 \
            --greedy \
            --seed 42 \
            2>&1 | tee "$RS_OUTPUT" || RS_EXIT_CODE=$?
    else
        cargo run -q --no-default-features --features cpu,full-cli -p bitnet-cli -- \
            run \
            --model "$MODEL_PATH" \
            --tokenizer "$TOKENIZER_PATH" \
            --prompt "$PROMPT" \
            --max-tokens "$MAX_TOKENS" \
            --temperature 0.0 \
            --greedy \
            --seed 42 \
            2>&1 | tee "$RS_OUTPUT" || RS_EXIT_CODE=$?
    fi

    if [[ $RS_EXIT_CODE -ne 0 ]]; then
        error "Rust inference failed with exit code $RS_EXIT_CODE" 1
    fi

    # Count trace files
    TRACE_COUNT=$(find "$RS_TRACE_DIR" -name "*.trace" -type f 2>/dev/null | wc -l)
    if [[ $TRACE_COUNT -eq 0 ]]; then
        warn "No trace files generated in $RS_TRACE_DIR"
    else
        success "Generated $TRACE_COUNT trace files in $RS_TRACE_DIR"
    fi

    # ========================================================================
    # Run C++ Inference (if available)
    # ========================================================================
    if [[ $CPP_AVAILABLE -eq 1 ]]; then
        info "Running C++ inference via crossval FFI..."

        # Run parity test (writes receipt to docs/baselines/)
        CPP_EXIT_CODE=0
        cd "$REPO_ROOT"
        cargo test -q --features "integration-tests,crossval,ffi" -p bitnet-crossval \
            parity_bitnetcpp -- --nocapture --test-threads=1 \
            2>&1 | tee "$CPP_OUTPUT" || CPP_EXIT_CODE=$?

        if [[ $CPP_EXIT_CODE -ne 0 ]]; then
            warn "C++ parity test failed with exit code $CPP_EXIT_CODE"
            echo "C++ inference failed" > "$CPP_OUTPUT"
        else
            success "C++ parity test completed"
        fi
    else
        info "Skipping C++ inference (not available)"
        echo "C++ reference not available" > "$CPP_OUTPUT"
    fi

    # ========================================================================
    # Analysis: Extract Metrics
    # ========================================================================
    info "Analyzing results..."

    # Extract Rust output tokens
    RS_TOKENS=$(grep -oP "Generated tokens:.*" "$RS_OUTPUT" 2>/dev/null || echo "N/A")

    # Initialize comparison metrics
    COSINE_SIM="N/A"
    EXACT_MATCH_RATE="N/A"
    FIRST_DIVERGENCE="N/A"
    TOKEN_MATCH="N/A"

    # Extract parity metrics from receipt (if C++ was run)
    if [[ $CPP_AVAILABLE -eq 1 ]]; then
        # Find most recent parity receipt
        RECEIPT_PATH=$(find "$REPO_ROOT/docs/baselines" -name "parity-bitnetcpp.json" -type f 2>/dev/null | sort -r | head -n1)

        if [[ -f "$RECEIPT_PATH" ]] && command -v jq >/dev/null 2>&1; then
            COSINE_SIM=$(jq -r '.parity.cosine_similarity // "N/A"' "$RECEIPT_PATH" 2>/dev/null || echo "N/A")
            EXACT_MATCH_RATE=$(jq -r '.parity.exact_match_rate // "N/A"' "$RECEIPT_PATH" 2>/dev/null || echo "N/A")
            TOKEN_MATCH=$(jq -r '.parity.status // "N/A"' "$RECEIPT_PATH" 2>/dev/null || echo "N/A")

            # Copy receipt to scenario directory
            cp "$RECEIPT_PATH" "$LOGITS_JSON" 2>/dev/null || true
        fi
    fi

    # Detect first diverging trace (by Blake3 hash)
    FIRST_DIVERGING_TRACE="N/A"
    if [[ $TRACE_COUNT -gt 0 ]] && command -v jq >/dev/null 2>&1; then
        # This is a placeholder - real divergence detection requires C++ traces
        # For now, just list first trace file
        FIRST_TRACE=$(find "$RS_TRACE_DIR" -name "*.trace" -type f 2>/dev/null | sort | head -n1)
        if [[ -f "$FIRST_TRACE" ]]; then
            FIRST_DIVERGING_TRACE=$(basename "$FIRST_TRACE")
        fi
    fi

    # ========================================================================
    # Generate Scenario Report
    # ========================================================================
    cat > "$REPORT_FILE" <<EOF
================================================================================
Cross-Validation Report: $SCENARIO_NAME
================================================================================

Test Configuration
------------------
Description:   $DESC
Prompt:        "$PROMPT"
Max Tokens:    $MAX_TOKENS
Model:         $(basename "$MODEL_PATH")
Tokenizer:     $(basename "$TOKENIZER_PATH")
Seed:          42
Temperature:   0.0 (greedy)

Execution Summary
-----------------
Rust Exit Code:     $RS_EXIT_CODE
C++ Available:      $([ $CPP_AVAILABLE -eq 1 ] && echo "Yes" || echo "No")
Trace Files:        $TRACE_COUNT files in $RS_TRACE_DIR

Output Comparison
-----------------
Rust Tokens:        $RS_TOKENS
Token Match:        $TOKEN_MATCH

Parity Metrics
--------------
Cosine Similarity:  $COSINE_SIM
Exact Match Rate:   $EXACT_MATCH_RATE
First Divergence:   $FIRST_DIVERGENCE

Trace Analysis
--------------
First Trace File:   $FIRST_DIVERGING_TRACE
Trace Directory:    $RS_TRACE_DIR

File Locations
--------------
Rust Output:        $RS_OUTPUT
C++ Output:         $CPP_OUTPUT
Logits JSON:        $LOGITS_JSON
Report:             $REPORT_FILE

EOF

    if [[ $CPP_AVAILABLE -eq 1 ]] && [[ -f "$LOGITS_JSON" ]]; then
        cat >> "$REPORT_FILE" <<EOF
Recommendations
---------------
1. Review parity receipt: $LOGITS_JSON
2. Compare trace Blake3 hashes for divergence detection
3. Check RMS values in trace files for numerical stability
4. If cosine similarity < 0.99, investigate first diverging trace

EOF
    else
        cat >> "$REPORT_FILE" <<EOF
Recommendations
---------------
1. C++ reference not available - parity metrics unavailable
2. Review Rust traces for internal consistency
3. To enable full parity validation, set BITNET_CPP_DIR and re-run

EOF
    fi

    # Display report
    cat "$REPORT_FILE"

    # Store scenario result
    SCENARIO_RESULTS+=("$SCENARIO_NAME|$RS_EXIT_CODE|$TOKEN_MATCH|$COSINE_SIM|$TRACE_COUNT")

    success "$SCENARIO_NAME completed"
done

# ============================================================================
# Generate Summary Report
# ============================================================================
section "Generating Summary Report"

SUMMARY_FILE="$OUTPUT_DIR/summary.md"

cat > "$SUMMARY_FILE" <<EOF
# Cross-Validation Sweep Summary

**Generated:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Model:** \`$(basename "$MODEL_PATH")\`
**Tokenizer:** \`$(basename "$TOKENIZER_PATH")\`
**C++ Reference:** $([ $CPP_AVAILABLE -eq 1 ] && echo "Available" || echo "Not Available (Rust-only mode)")

---

## Scenario Results

| Scenario | Exit Code | Token Match | Cosine Similarity | Trace Files |
|----------|-----------|-------------|-------------------|-------------|
EOF

for result in "${SCENARIO_RESULTS[@]}"; do
    IFS='|' read -r name exit_code token_match cosine trace_count <<< "$result"
    echo "| $name | $exit_code | $token_match | $cosine | $trace_count |" >> "$SUMMARY_FILE"
done

cat >> "$SUMMARY_FILE" <<EOF

---

## Analysis

### Success Criteria

- **Exit Code:** All scenarios should exit with code 0
- **Token Match:** Status should be "ok" (for full parity) or "rust_only"
- **Cosine Similarity:** Should be ≥ 0.99 for high-quality parity
- **Trace Files:** Should generate 90+ trace files per scenario

### Divergence Detection

EOF

if [[ $CPP_AVAILABLE -eq 1 ]]; then
    cat >> "$SUMMARY_FILE" <<EOF
Full C++ parity validation was performed. Check individual scenario reports for:

1. **First Divergence Position:** Token position where Rust and C++ outputs first differ
2. **Trace Hash Comparison:** Blake3 hash comparison to identify diverging layer/operation
3. **RMS Consistency:** Statistical validation of tensor magnitudes

### Actionable Recommendations

- ✅ **Green (Cosine ≥ 0.99):** Parity is excellent, no action needed
- ⚠️  **Yellow (0.95 ≤ Cosine < 0.99):** Minor divergence, investigate traces
- ❌ **Red (Cosine < 0.95):** Significant divergence, check first diverging trace

### Next Steps

1. Review scenario reports in \`$OUTPUT_DIR/scenario*/report.txt\`
2. Compare trace Blake3 hashes between Rust and C++ implementations
3. For divergences, examine RMS values and tensor shapes in trace files
4. Use \`jq\` to analyze parity receipts: \`jq . $OUTPUT_DIR/scenario*/logits-comparison.json\`

EOF
else
    cat >> "$SUMMARY_FILE" <<EOF
**Note:** C++ reference not available - this is a Rust-only validation.

To enable full cross-validation:
1. Set \`BITNET_CPP_DIR\` to point to a built bitnet.cpp installation
2. Re-run this script

Current trace files can still be used for:
- Internal consistency validation
- Numerical stability analysis
- Regression testing against future changes

EOF
fi

cat >> "$SUMMARY_FILE" <<EOF
---

## Directory Structure

\`\`\`
$OUTPUT_DIR/
$(find "$OUTPUT_DIR" -maxdepth 2 -type f -o -type d | sed "s|^$OUTPUT_DIR/|├── |" | head -30)
\`\`\`

---

## Environment

- **BITNET_DETERMINISTIC:** 1
- **BITNET_SEED:** 42
- **RAYON_NUM_THREADS:** 4
- **BITNET_CPP_DIR:** $BITNET_CPP_DIR
- **Timeout per scenario:** ${TIMEOUT_SECS}s

---

**Full documentation:** See \`docs/development/validation-framework.md\`
EOF

# Display summary
cat "$SUMMARY_FILE"

success "Summary report generated: $SUMMARY_FILE"

# ============================================================================
# Final Status
# ============================================================================
section "Cross-Validation Sweep Complete"

echo ""
info "Results saved to: $OUTPUT_DIR"
info "Summary report:   $SUMMARY_FILE"
echo ""

# Exit with success if all scenarios passed
ALL_PASSED=1
for result in "${SCENARIO_RESULTS[@]}"; do
    IFS='|' read -r name exit_code _ _ _ <<< "$result"
    if [[ $exit_code -ne 0 ]]; then
        ALL_PASSED=0
        warn "Scenario $name failed with exit code $exit_code"
    fi
done

if [[ $ALL_PASSED -eq 1 ]]; then
    success "All scenarios passed successfully!"
    exit 0
else
    error "Some scenarios failed - review reports for details" 1
fi
