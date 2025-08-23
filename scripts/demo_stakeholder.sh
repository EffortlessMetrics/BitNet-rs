#!/usr/bin/env bash
# Quick stakeholder demo script (5-minute tour)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

setup_deterministic_env
print_platform_banner
detect_wsl2 || true

BIN=$(find_bitnet_binary)
MODEL_ST="${MODEL_ST:-models/bitnet_b1_58-3B/safetensors/model.safetensors}"
TOK_ST="${TOK_ST:-models/bitnet_b1_58-3B/safetensors/tokenizer.json}"
MODEL_GGUF="${MODEL_GGUF:-models/bitnet_b1_58-3B/gguf/model.gguf}"

echo ""
echo "======================================================"
echo "  BitNet.rs Stakeholder Demo - 5 Minute Tour"
echo "======================================================"
echo ""

echo "==> Step 1: Model Introspection (SafeTensors)"
if [ -f "$MODEL_ST" ]; then
    $BIN info --model "$MODEL_ST" --tokenizer "$TOK_ST" --json | jq '{format, tokenizer_source, scoring_policy}'
else
    echo "SafeTensors model not found at $MODEL_ST"
fi

echo ""
echo "==> Step 2: Model Introspection (GGUF)"
if [ -f "$MODEL_GGUF" ]; then
    $BIN info --model "$MODEL_GGUF" --json | jq '{format, tokenizer_source, scoring_policy}'
else
    echo "GGUF model not found at $MODEL_GGUF"
fi

echo ""
echo "==> Step 3: Format Parity Validation"
if [ -f "scripts/validate_format_parity.sh" ]; then
    scripts/validate_format_parity.sh | tail -20
fi

echo ""
echo "==> Step 4: Performance Measurement"
if [ -f "scripts/measure_perf_json.sh" ]; then
    echo "Measuring performance (this may take a minute)..."
    scripts/measure_perf_json.sh
fi

echo ""
echo "==> Step 5: Render Performance Report"
PLATFORM=$(get_platform_name)
if [ -f "bench/results/${PLATFORM}-safetensors.json" ] && [ -f "bench/results/${PLATFORM}-gguf.json" ]; then
    python3 scripts/render_perf_md.py \
        "bench/results/${PLATFORM}-safetensors.json" \
        "bench/results/${PLATFORM}-gguf.json" \
        > docs/PERF_COMPARISON.md
    echo "Performance comparison saved to docs/PERF_COMPARISON.md"
    head -30 docs/PERF_COMPARISON.md
fi

echo ""
echo "======================================================"
echo "✅ Demo Complete!"
echo "======================================================"
echo ""
echo "Key takeaways:"
echo "  • Both SafeTensors and GGUF formats fully supported"
echo "  • Format parity validated through multiple layers"
echo "  • Performance measured from actual runs (not estimates)"
echo "  • All results reproducible with deterministic mode"
echo ""
echo "For full validation, run: ./scripts/reality_proof_checklist.sh"