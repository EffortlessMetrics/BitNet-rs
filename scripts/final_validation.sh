#!/usr/bin/env bash
# Final validation script - The last line of defense before release
# This is your go/no-go decision point
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}               BitNet.rs Final Release Validation                  ${NC}"
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo

# Track validation results
VALIDATION_RESULTS=()
FAILED=false

# Function to record result
record_result() {
    local test_name="$1"
    local status="$2"
    local details="${3:-}"

    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✅ ${test_name}${NC}"
        VALIDATION_RESULTS+=("✅ ${test_name}")
    else
        echo -e "${RED}❌ ${test_name}${NC}"
        [ -n "$details" ] && echo -e "   ${YELLOW}${details}${NC}"
        VALIDATION_RESULTS+=("❌ ${test_name}: ${details}")
        FAILED=true
    fi
}

# 0. Platform and Environment
echo -e "${BOLD}1. Platform & Environment${NC}"
echo "──────────────────────────"
setup_deterministic_env
print_platform_banner
record_result "Environment setup" "PASS"
echo

# 1. Run acceptance tests
echo -e "${BOLD}2. Acceptance Tests (Both Formats)${NC}"
echo "────────────────────────────────────"
if "${SCRIPT_DIR}/acceptance_test.sh" >/tmp/acceptance.log 2>&1; then
    record_result "Acceptance tests" "PASS"
    echo -e "   ${GREEN}Both SafeTensors and GGUF formats validated${NC}"
else
    record_result "Acceptance tests" "FAIL" "Check /tmp/acceptance.log"
fi
echo

# 2. Format Parity Validation
echo -e "${BOLD}3. Format Parity Validation${NC}"
echo "─────────────────────────────"
if "${SCRIPT_DIR}/validate_format_parity.sh" >/tmp/parity.log 2>&1; then
    # Check thresholds from JSON
    PARITY_JSON=$(ls -t artifacts/*format_parity*.json 2>/dev/null | head -1)
    if [ -f "$PARITY_JSON" ]; then
        FP32_TAU=$(jq -r '.results.fp32_vs_fp32.kendall_tau_b.median // 0' "$PARITY_JSON")
        FP32_NLL=$(jq -r '.results.fp32_vs_fp32.nll_diff.mean // 999' "$PARITY_JSON")
        QUANT_TAU=$(jq -r '.results.quant_vs_fp32.kendall_tau_b.median // 0' "$PARITY_JSON")
        QUANT_NLL=$(jq -r '.results.quant_vs_fp32.nll_diff.mean // 999' "$PARITY_JSON")

        # Check FP32 thresholds
        if awk "BEGIN {exit !($FP32_TAU >= 0.95)}" && \
           awk "BEGIN {exit !(($FP32_NLL < 0 ? -$FP32_NLL : $FP32_NLL) <= 0.01)}"; then
            record_result "FP32↔FP32 parity" "PASS" "τ-b=${FP32_TAU}, ΔNLL=${FP32_NLL}"
        else
            record_result "FP32↔FP32 parity" "FAIL" "τ-b=${FP32_TAU} (need ≥0.95), ΔNLL=${FP32_NLL} (need ≤0.01)"
        fi

        # Check Quant thresholds
        if awk "BEGIN {exit !($QUANT_TAU >= 0.60)}" && \
           awk "BEGIN {exit !(($QUANT_NLL < 0 ? -$QUANT_NLL : $QUANT_NLL) <= 0.02)}"; then
            record_result "Quant↔FP32 parity" "PASS" "τ-b=${QUANT_TAU}, ΔNLL=${QUANT_NLL}"
        else
            record_result "Quant↔FP32 parity" "FAIL" "τ-b=${QUANT_TAU} (need ≥0.60), ΔNLL=${QUANT_NLL} (need ≤0.02)"
        fi
    else
        record_result "Format parity" "FAIL" "No parity JSON found"
    fi
else
    record_result "Format parity" "FAIL" "Validation failed"
fi
echo

# 3. Performance Measurement
echo -e "${BOLD}4. Performance Measurement${NC}"
echo "───────────────────────────"
# Check for recent perf JSONs
ST_PERF=$(ls -t bench/results/*-safetensors.json 2>/dev/null | head -1)
GGUF_PERF=$(ls -t bench/results/*-gguf.json 2>/dev/null | head -1)

if [ -f "$ST_PERF" ] && [ -f "$GGUF_PERF" ]; then
    record_result "SafeTensors perf JSON" "PASS" "$(basename "$ST_PERF")"
    record_result "GGUF perf JSON" "PASS" "$(basename "$GGUF_PERF")"

    # Verify provenance fields
    for json in "$ST_PERF" "$GGUF_PERF"; do
        if jq -e '.git_commit and .model_hash and .schema_version' "$json" >/dev/null; then
            record_result "Provenance in $(basename "$json")" "PASS"
        else
            record_result "Provenance in $(basename "$json")" "FAIL" "Missing required fields"
        fi
    done
else
    record_result "Performance JSONs" "FAIL" "Missing SafeTensors or GGUF perf JSON"
fi
echo

# 4. Documentation Sync
echo -e "${BOLD}5. Documentation Sync${NC}"
echo "──────────────────────"
if [ -f "$ST_PERF" ] && [ -f "$GGUF_PERF" ]; then
    # Regenerate and compare
    python3 "${SCRIPT_DIR}/render_perf_md.py" "$ST_PERF" "$GGUF_PERF" > /tmp/perf_check.md 2>/dev/null

    if [ -f docs/PERF_COMPARISON.md ]; then
        if diff -q <(grep -v "Generated from" docs/PERF_COMPARISON.md) \
                   <(grep -v "Generated from" /tmp/perf_check.md) >/dev/null; then
            record_result "PERF_COMPARISON.md sync" "PASS"
        else
            record_result "PERF_COMPARISON.md sync" "FAIL" "Out of sync with JSON sources"
        fi
    else
        record_result "PERF_COMPARISON.md" "FAIL" "File not found"
    fi
else
    record_result "Documentation sync" "SKIP" "No perf JSONs to validate against"
fi
echo

# 5. Release Sign-off
echo -e "${BOLD}6. Release Sign-off Checks${NC}"
echo "────────────────────────────"
if "${SCRIPT_DIR}/release_signoff.sh" >/tmp/signoff.log 2>&1; then
    record_result "Release sign-off" "PASS"
else
    record_result "Release sign-off" "FAIL" "Check /tmp/signoff.log"
fi
echo

# 6. JSON Schema Validation
echo -e "${BOLD}7. JSON Schema Validation${NC}"
echo "──────────────────────────"
if command -v python3 >/dev/null && python3 -c "import jsonschema" 2>/dev/null; then
    # Validate perf JSONs
    if python3 - <<'EOF' 2>/dev/null
import json, glob, sys
from jsonschema import Draft7Validator
schema = json.load(open('bench/schema/perf.schema.json'))
for p in glob.glob('bench/results/*.json'):
    Draft7Validator(schema).validate(json.load(open(p)))
sys.exit(0)
EOF
    then
        record_result "Performance JSON schemas" "PASS"
    else
        record_result "Performance JSON schemas" "FAIL" "Schema validation failed"
    fi
else
    record_result "JSON schema validation" "SKIP" "jsonschema not installed"
fi
echo

# Final Summary
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}                        VALIDATION SUMMARY                         ${NC}"
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo

for result in "${VALIDATION_RESULTS[@]}"; do
    echo -e "  $result"
done
echo

if [ "$FAILED" = true ]; then
    echo -e "${BOLD}${RED}❌ VALIDATION FAILED${NC}"
    echo -e "${YELLOW}Fix the issues above before proceeding with release${NC}"
    echo
    echo -e "${BOLD}Debug logs available at:${NC}"
    echo "  - /tmp/acceptance.log"
    echo "  - /tmp/parity.log"
    echo "  - /tmp/signoff.log"
    exit 1
else
    echo -e "${BOLD}${GREEN}✅ ALL VALIDATIONS PASSED!${NC}"
    echo -e "${GREEN}Ready for release${NC}"
    echo
    echo -e "${BOLD}Next steps:${NC}"
    echo "  1. Run: ./scripts/prepare_release.sh --version X.Y.Z"
    echo "  2. Review: RELEASE_CHECKLIST_vX.Y.Z.md"
    echo "  3. Tag: git tag -a vX.Y.Z -m 'Dual-format validation: audited & gated'"
    echo "  4. Push: git push --tags"
fi

echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
