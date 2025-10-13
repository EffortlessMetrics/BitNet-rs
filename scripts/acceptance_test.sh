#!/usr/bin/env bash
# 10-minute acceptance test for dual-format support
# This script validates both SafeTensors and GGUF are production-ready

set -euo pipefail

# Safe defaults for set -u
FAILED=${FAILED:-0}
RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
BITNET_DETERMINISTIC=${BITNET_DETERMINISTIC:-1}
BITNET_BIN=${BITNET_BIN:-$(command -v bitnet || echo "target/release/bitnet")}

# JSON-safe boolean (no "false: unbound variable")
bool_json() { if eval "$1" >/dev/null 2>&1; then echo true; else echo false; fi; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Setup deterministic environment immediately
setup_deterministic_env
print_platform_banner

# Configuration
# Prefer a GGUF in models/, else env, else bail with a clear message
MODEL_BASE="${MODEL_BASE:-}"
if [[ -z "${MODEL_BASE}" ]]; then
    # any gguf under models/
    MODEL_BASE="$(find models -type f -name '*.gguf' | head -n1 || true)"
    # allow explicit override via env
    [[ -z "${MODEL_BASE}" && -n "${BITNET_GGUF:-}" ]] && MODEL_BASE="$BITNET_GGUF"
fi

if [[ -z "${MODEL_BASE}" || ! -f "${MODEL_BASE}" ]]; then
    echo "No model found. Run: scripts/prepare_test_model.sh"
    exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-test-results/acceptance-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTPUT_DIR"

# Find BitNet binary
BITNET_BIN=$(find_bitnet_binary)
echo "Using BitNet binary: $BITNET_BIN"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}✅${NC} $1"
}

log_fail() {
    echo -e "${RED}❌${NC} $1"
}

log_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Track overall status
FAILED=0

# 1. Model introspection (both formats)
run_introspection() {
    log_step "1. Model Introspection (format, tokenizer, scoring policy)"

    # SafeTensors
    SAFETENSORS_MODEL=""
    SAFETENSORS_TOKENIZER=""
    if [[ -f "$MODEL_BASE/safetensors/model.safetensors" ]]; then
        SAFETENSORS_MODEL="$MODEL_BASE/safetensors/model.safetensors"
        SAFETENSORS_TOKENIZER="$MODEL_BASE/safetensors/tokenizer.json"
    elif [[ "$MODEL_BASE" == *.safetensors && -f "$MODEL_BASE" ]]; then
        SAFETENSORS_MODEL="$MODEL_BASE"
        SAFETENSORS_TOKENIZER="${MODEL_BASE%/*}/tokenizer.json"
    fi

    if [[ -n "$SAFETENSORS_MODEL" && -f "$SAFETENSORS_MODEL" ]]; then
        echo "  Checking SafeTensors format..."
        env RUST_LOG=error $BITNET_BIN inspect \
            --model "$SAFETENSORS_MODEL" \
            --json > "$OUTPUT_DIR/info-safetensors.json" 2>&1

        if [ $? -eq 0 ]; then
            log_pass "SafeTensors introspection"
            jq -r '.format, .tokenizer.source, .scoring_policy' "$OUTPUT_DIR/info-safetensors.json" | head -3
        else
            log_fail "SafeTensors introspection failed"
            FAILED=$((FAILED + 1))
        fi
    fi

    # GGUF
    GGUF_MODEL=""
    if [[ -f "$MODEL_BASE/gguf/model.gguf" ]]; then
        GGUF_MODEL="$MODEL_BASE/gguf/model.gguf"
    elif [[ "$MODEL_BASE" == *.gguf && -f "$MODEL_BASE" ]]; then
        GGUF_MODEL="$MODEL_BASE"
    fi

    if [[ -n "$GGUF_MODEL" && -f "$GGUF_MODEL" ]]; then
        echo "  Checking GGUF format..."
        env RUST_LOG=error $BITNET_BIN inspect \
            --model "$GGUF_MODEL" \
            --json > "$OUTPUT_DIR/info-gguf.json" 2>&1

        if [ $? -eq 0 ]; then
            log_pass "GGUF introspection"
            jq -r '.format, .tokenizer.source, .scoring_policy' "$OUTPUT_DIR/info-gguf.json" | head -3
        else
            log_fail "GGUF introspection failed"
            FAILED=$((FAILED + 1))
        fi
    fi
    echo ""
}

# 2. Format parity validation
run_parity() {
    log_step "2. Format Parity (tokenizer + τ-b + TF NLL)"

    if "$SCRIPT_DIR/validate_format_parity.sh" > "$OUTPUT_DIR/parity.log" 2>&1; then
        # Check the parity results
        if [ -f "/tmp/parity_results.json" ]; then
            cp /tmp/parity_results.json "$OUTPUT_DIR/parity_results.json"

            # Extract key metrics
            TAU=$(jq -r '.tau_b.median // 0' "$OUTPUT_DIR/parity_results.json")
            NLL_DELTA=$(jq -r '.nll.delta_mean // 999' "$OUTPUT_DIR/parity_results.json")
            PASS=$(jq -r '.pass // false' "$OUTPUT_DIR/parity_results.json")

            echo "  Median τ-b: $TAU"
            echo "  |Δ mean_nll|: $NLL_DELTA"

            if [ "$PASS" = "true" ]; then
                log_pass "Format parity validation"
            else
                log_fail "Format parity validation (see $OUTPUT_DIR/parity_results.json)"
                FAILED=$((FAILED + 1))
            fi
        else
            log_fail "No parity results generated"
            FAILED=$((FAILED + 1))
        fi
    else
        log_fail "Format parity validation failed"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# 3. Performance measurements
run_performance() {
    log_step "3. Performance Measurements → JSON → Markdown"

    if "$SCRIPT_DIR/measure_perf_json.sh" > "$OUTPUT_DIR/perf.log" 2>&1; then
        # Check for generated JSONs
        PERF_JSONS=$(ls bench/results/*-{safetensors,gguf}.json 2>/dev/null | wc -l)

        if [ "$PERF_JSONS" -ge 2 ]; then
            log_pass "Performance JSONs generated ($PERF_JSONS files)"

            # Render markdown
            python3 "$SCRIPT_DIR/render_perf_md.py" \
                bench/results/*-safetensors.json \
                bench/results/*-gguf.json \
                > "$OUTPUT_DIR/PERF_COMPARISON.md" 2>&1

            if [ $? -eq 0 ]; then
                log_pass "Performance markdown rendered"
                echo "  Report: $OUTPUT_DIR/PERF_COMPARISON.md"
            else
                log_fail "Markdown rendering failed"
                FAILED=$((FAILED + 1))
            fi
        else
            log_fail "Insufficient performance JSONs (found $PERF_JSONS, need 2+)"
            FAILED=$((FAILED + 1))
        fi
    else
        log_fail "Performance measurement failed"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# 4. CI validation checks
run_ci_checks() {
    log_step "4. CI Validation Checks"

    # Check for mock features (ignore CI guards)
    echo -n "  Checking for mock features... "
    MOCK_HITS=$(
      git grep -n -E \
        '(^|[[:space:]])cargo[[:space:]]+(build|test|run)([[:space:][:graph:]]*?)--features([[:space:]=]+)[^#\n]*\bmocks\b' \
        -- scripts/ crates/ Cargo.* Makefile* 2>/dev/null \
      | grep -v -E '\.github/|grep[[:space:]].*--features[[:space:]=]+mocks' \
      || true
    )
    if [ -n "$MOCK_HITS" ]; then
        log_fail "Mock features found in codebase"
        echo "$MOCK_HITS"
        FAILED=$((FAILED + 1))
    else
        log_pass "No mock features"
    fi

    # Check GGUF changes
    echo -n "  Checking GGUF validation requirement... "
    GGUF_CHANGED=$(git diff --name-only HEAD~1..HEAD 2>/dev/null | grep -E '\.gguf$' || true)
    if [ -n "$GGUF_CHANGED" ]; then
        if [ -f "$OUTPUT_DIR/parity_results.json" ]; then
            log_pass "GGUF changes have parity validation"
        else
            log_fail "GGUF changes without parity validation"
            FAILED=$((FAILED + 1))
        fi
    else
        log_info "No GGUF changes to validate"
    fi

    # Check perf docs vs JSON
    echo -n "  Checking perf docs match JSON... "
    PERF_MD_CHANGED=$(git diff --name-only HEAD~1..HEAD 2>/dev/null | grep -E '^docs/PERF_.*\.md$' || true)
    if [ -n "$PERF_MD_CHANGED" ]; then
        PERF_JSON_CHANGED=$(git diff --name-only HEAD~1..HEAD 2>/dev/null | grep -E '^bench/results/.*\.json$' || true)
        if [ -n "$PERF_JSON_CHANGED" ]; then
            log_pass "Performance docs have matching JSON"
        else
            log_fail "Performance docs changed without JSON"
            FAILED=$((FAILED + 1))
        fi
    else
        log_info "No performance doc changes"
    fi
    echo ""
}

# Generate summary
generate_summary() {
    log_step "5. Generating Summary"

    # Build summary via jq -n to avoid heredoc boolean issues under set -u
    jq -n \
      --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --arg platform "$(uname -s)-$(uname -m)$(grep -qi microsoft /proc/version && echo -n '-WSL2' || true)" \
      --arg bin "$BITNET_BIN" \
      --argjson wsl2 "$(bool_json 'grep -qi microsoft /proc/version')" \
      --argjson det "${BITNET_DETERMINISTIC:-1}" \
      --argjson thr "${RAYON_NUM_THREADS:-1}" \
      --argjson introspect  "$( ([ "${FAILED:-0}" -eq 0 ] && echo true) || echo false)" \
      --argjson parity       "$( ([ -f "$OUTPUT_DIR/parity_results.json" ] && jq -r '.pass' "$OUTPUT_DIR/parity_results.json") || echo false)" \
      --argjson perf         "$( ([ -f "$OUTPUT_DIR/PERF_COMPARISON.md" ] && echo true) || echo false)" \
      --argjson cicheck      "$( ([ "${FAILED:-0}" -eq 0 ] && echo true) || echo false)" \
      --argjson failed       "${FAILED:-0}" \
      --argjson pass         "$( ([ "${FAILED:-0}" -eq 0 ] && echo true) || echo false)" \
    '{
      timestamp: $ts,
      platform: $platform,
      wsl2: $wsl2,
      deterministic: $det,
      threads: $thr,
      binary: $bin,
      tests: {
        introspection: $introspect,
        parity: $parity,
        performance: $perf,
        ci_checks: $cicheck
      },
      artifacts: {
        info_safetensors: "'"$OUTPUT_DIR"'/info-safetensors.json",
        info_gguf: "'"$OUTPUT_DIR"'/info-gguf.json",
        parity_results: "'"$OUTPUT_DIR"'/parity_results.json",
        performance_report: "'"$OUTPUT_DIR"'/PERF_COMPARISON.md"
      },
      failed_checks: $failed,
      pass: $pass
    }' > "$OUTPUT_DIR/acceptance_summary.json"

    log_pass "Summary generated: $OUTPUT_DIR/acceptance_summary.json"
}

# Main execution
main() {
    echo "====================================================="
    echo "    10-Minute Acceptance Test (Both Formats)"
    echo "====================================================="
    echo ""
    log_info "Output directory: $OUTPUT_DIR"
    log_info "Models: $MODEL_BASE"
    echo ""

    # Run all tests
    run_introspection
    run_parity
    run_performance
    run_ci_checks
    generate_summary

    # Final status
    echo "====================================================="
    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}✅ ALL ACCEPTANCE TESTS PASSED!${NC}"
        echo ""
        echo "Both SafeTensors and GGUF formats are production-ready."
        echo ""
        echo "Pass criteria met:"
        echo "  • Format parity: τ-b ≥ 0.60, |Δ NLL| ≤ 2e-2"
        echo "  • Performance measured and documented"
        echo "  • CI gates properly configured"
        echo "  • No mock features in codebase"
        echo ""
        echo "Artifacts: $OUTPUT_DIR/"
        return 0
    else
        echo -e "${RED}❌ ACCEPTANCE TESTS FAILED${NC}"
        echo ""
        echo "$FAILED checks failed. Review artifacts in:"
        echo "  $OUTPUT_DIR/"
        return 1
    fi
}

# Run if executed directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
