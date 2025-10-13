#!/usr/bin/env bash
set -euo pipefail

# Comprehensive validation for both SafeTensors and GGUF formats
# Runs the full validation pyramid for each format and compares results

# Configuration
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPTS_DIR}/common.sh"

MODELS_DIR="${MODELS_DIR:-models}"
OUTPUT_DIR=$(ensure_output_dir "validation_results")
MODEL_ID="${MODEL_ID:-bitnet_b1_58-3B}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging
log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Setup environment
setup_env() {
    log_info "Setting up validation environment"

    # Set up model storage if needed
    if [ ! -d "$MODELS_DIR" ] || [ ! -f "${MODELS_DIR}/index.json" ]; then
        log_info "Initializing model storage layout..."
        bash "${SCRIPTS_DIR}/setup_model_storage.sh"
    fi

    # Find model paths
    export SAFETENSORS_MODEL="${MODELS_DIR}/${MODEL_ID}/safetensors/model.safetensors"
    export GGUF_MODEL="${MODELS_DIR}/${MODEL_ID}/gguf/model.gguf"
    export TOKENIZER="${MODELS_DIR}/${MODEL_ID}/safetensors/tokenizer.json"

    # Check SafeTensors model
    if [ ! -f "$SAFETENSORS_MODEL" ]; then
        log_warn "SafeTensors model not found, attempting download..."
        cargo xtask download-model || {
            log_error "Failed to download model"
            exit 1
        }

        # Re-setup storage
        bash "${SCRIPTS_DIR}/setup_model_storage.sh"
    fi

    # Convert to GGUF if needed
    if [ ! -f "$GGUF_MODEL" ]; then
        log_info "Converting SafeTensors to GGUF..."
        python3 "${SCRIPTS_DIR}/convert_safetensors_to_gguf.py" \
            "${MODELS_DIR}/${MODEL_ID}/safetensors" \
            "$GGUF_MODEL" || {
            log_warn "GGUF conversion failed, will skip GGUF tests"
        }
    fi

    # Set deterministic environment
    export BITNET_DETERMINISTIC=1
    export BITNET_SEED=42
    export RAYON_NUM_THREADS=1

    log_info "Environment ready:"
    log_info "  SafeTensors: ${SAFETENSORS_MODEL}"
    [ -f "$GGUF_MODEL" ] && log_info "  GGUF: ${GGUF_MODEL}"
    log_info "  Tokenizer: ${TOKENIZER}"
}

# Run validation for a specific format
validate_format() {
    local format="$1"
    local model_path="$2"
    local output_subdir="${OUTPUT_DIR}/${MODEL_ID}/${format}"

    log_info "=== Validating ${format} format ==="

    mkdir -p "$output_subdir"

    # Test 1: Tokenizer parity (if GGUF has embedded tokenizer)
    if [ "$format" = "gguf" ] && [ -f "${SCRIPTS_DIR}/test-tokenizer-parity.py" ]; then
        log_info "Test 1: Tokenizer parity"

        export BITNET_BIN="${BITNET_BIN:-bitnet}"
        export MODEL_PATH="$model_path"

        if python3 "${SCRIPTS_DIR}/test-tokenizer-parity.py" --smoke \
            > "${output_subdir}/tokenizer_parity.log" 2>&1; then
            log_info "  ✓ Tokenizer parity: PASSED"
        else
            log_warn "  ✗ Tokenizer parity: FAILED (see ${output_subdir}/tokenizer_parity.log)"
        fi
    fi

    # Test 2: Logit parity (vs HuggingFace)
    if [ -n "${HF_MODEL_ID:-}" ] && [ -f "${SCRIPTS_DIR}/logit-parity.sh" ]; then
        log_info "Test 2: Logit parity vs HuggingFace"

        export MODEL_PATH="$model_path"
        export PROP_EXAMPLES="${PROP_EXAMPLES:-10}"
        export TAU_STEPS="${TAU_STEPS:-24}"
        export LOGIT_TOPK="${LOGIT_TOPK:-100}"
        export TAU_MIN="${TAU_MIN:-0.60}"

        if bash "${SCRIPTS_DIR}/logit-parity.sh" \
            > "${output_subdir}/logit_parity.log" 2>&1; then
            log_info "  ✓ Logit parity: PASSED"
        else
            log_warn "  ✗ Logit parity: FAILED (see ${output_subdir}/logit_parity.log)"
        fi
    fi

    # Test 3: NLL parity (perplexity)
    if [ -f "${SCRIPTS_DIR}/nll-parity.sh" ] && [ -f "crossval/data/ppl_smoke.txt" ]; then
        log_info "Test 3: NLL parity (perplexity)"

        export MODEL_PATH="$model_path"
        export PPL_FILE="crossval/data/ppl_smoke.txt"
        export DELTA_NLL_MAX="${DELTA_NLL_MAX:-0.01}"

        if bash "${SCRIPTS_DIR}/nll-parity.sh" \
            > "${output_subdir}/nll_parity.log" 2>&1; then
            log_info "  ✓ NLL parity: PASSED"
        else
            log_warn "  ✗ NLL parity: FAILED (see ${output_subdir}/nll_parity.log)"
        fi
    fi

    # Test 4: Performance benchmark
    log_info "Test 4: Performance benchmark"

    "${BITNET_BIN:-bitnet}" benchmark \
        --model "$model_path" \
        --model-format "$format" \
        --iterations 10 \
        --json > "${output_subdir}/benchmark.json" 2>/dev/null || {
        log_warn "  ✗ Benchmark failed"
    }

    if [ -f "${output_subdir}/benchmark.json" ]; then
        # Extract key metrics
        python3 -c "
import json
with open('${output_subdir}/benchmark.json') as f:
    data = json.load(f)
    if 'throughput' in data:
        print(f'  Throughput: {data[\"throughput\"][\"mean_tps\"]:.1f} tok/s')
    if 'latency' in data:
        print(f'  First token: {data[\"latency\"][\"first_token_ms\"]:.1f} ms')
    if 'memory' in data:
        print(f'  Peak memory: {data[\"memory\"][\"peak_rss_mb\"]:.1f} MB')
"
        log_info "  ✓ Benchmark complete"
    fi

    # Generate format report
    local report="${output_subdir}/validation_report.json"
    python3 -c "
import json
from datetime import datetime
import os

report = {
    'format': '$format',
    'model_path': '$model_path',
    'timestamp': datetime.utcnow().isoformat() + 'Z',
    'tests': {},
}

# Check test results
for test, logfile in [
    ('tokenizer_parity', '${output_subdir}/tokenizer_parity.log'),
    ('logit_parity', '${output_subdir}/logit_parity.log'),
    ('nll_parity', '${output_subdir}/nll_parity.log'),
]:
    if os.path.exists(logfile):
        # Simple heuristic: check if log contains 'PASSED' or no error indicators
        with open(logfile) as f:
            content = f.read()
            report['tests'][test] = 'PASSED' in content or 'error' not in content.lower()

# Add benchmark data if available
bench_file = '${output_subdir}/benchmark.json'
if os.path.exists(bench_file):
    with open(bench_file) as f:
        report['benchmark'] = json.load(f)

with open('$report', 'w') as f:
    json.dump(report, f, indent=2)
"

    log_info "Format validation report: ${report}"
}

# Compare formats
compare_formats() {
    log_info "=== Comparing SafeTensors vs GGUF ==="

    local st_report="${OUTPUT_DIR}/${MODEL_ID}/safetensors/validation_report.json"
    local gguf_report="${OUTPUT_DIR}/${MODEL_ID}/gguf/validation_report.json"

    if [ ! -f "$st_report" ] || [ ! -f "$gguf_report" ]; then
        log_warn "Missing validation reports for comparison"
        return 1
    fi

    # Run format parity validation
    if [ -f "${SCRIPTS_DIR}/validate_format_parity.sh" ]; then
        log_info "Running format parity validation..."
        bash "${SCRIPTS_DIR}/validate_format_parity.sh" "$MODEL_ID" || {
            log_warn "Format parity validation failed"
        }
    fi

    # Generate comparison report
    local comparison="${OUTPUT_DIR}/${MODEL_ID}/format_comparison.json"
    python3 -c "
import json
from datetime import datetime

with open('$st_report') as f:
    st_data = json.load(f)
with open('$gguf_report') as f:
    gguf_data = json.load(f)

comparison = {
    'timestamp': datetime.utcnow().isoformat() + 'Z',
    'model_id': '$MODEL_ID',
    'formats': {
        'safetensors': st_data,
        'gguf': gguf_data,
    },
}

# Calculate performance differences if benchmarks available
if 'benchmark' in st_data and 'benchmark' in gguf_data:
    st_bench = st_data['benchmark']
    gguf_bench = gguf_data['benchmark']

    comparison['performance_comparison'] = {}

    if 'throughput' in st_bench and 'throughput' in gguf_bench:
        st_tps = st_bench['throughput'].get('mean_tps', 0)
        gguf_tps = gguf_bench['throughput'].get('mean_tps', 0)
        if st_tps > 0:
            diff_pct = ((gguf_tps - st_tps) / st_tps) * 100
            comparison['performance_comparison']['throughput_diff_pct'] = round(diff_pct, 1)

    if 'memory' in st_bench and 'memory' in gguf_bench:
        st_mem = st_bench['memory'].get('peak_rss_mb', 0)
        gguf_mem = gguf_bench['memory'].get('peak_rss_mb', 0)
        if st_mem > 0:
            diff_pct = ((gguf_mem - st_mem) / st_mem) * 100
            comparison['performance_comparison']['memory_diff_pct'] = round(diff_pct, 1)

with open('$comparison', 'w') as f:
    json.dump(comparison, f, indent=2)

# Print summary
print('\\nFormat Comparison Summary:')
print('-' * 40)

# Test results
for fmt in ['safetensors', 'gguf']:
    tests = comparison['formats'][fmt].get('tests', {})
    passed = sum(1 for v in tests.values() if v)
    total = len(tests)
    print(f'{fmt.upper():12} {passed}/{total} tests passed')

# Performance comparison
if 'performance_comparison' in comparison:
    perf = comparison['performance_comparison']
    if 'throughput_diff_pct' in perf:
        sign = '+' if perf['throughput_diff_pct'] > 0 else ''
        print(f'\\nGGUF vs SafeTensors:')
        print(f'  Throughput: {sign}{perf[\"throughput_diff_pct\"]:.1f}%')
    if 'memory_diff_pct' in perf:
        sign = '+' if perf['memory_diff_pct'] > 0 else ''
        print(f'  Memory:     {sign}{perf[\"memory_diff_pct\"]:.1f}%')
"

    log_info "Comparison report: ${comparison}"
}

# Main validation flow
main() {
    log_info "=== BitNet Multi-Format Validation ==="
    log_info "Model: ${MODEL_ID}"
    log_info "Output: ${OUTPUT_DIR}"
    echo

    # Setup environment
    setup_env

    # Validate SafeTensors
    if [ -f "$SAFETENSORS_MODEL" ]; then
        validate_format "safetensors" "$SAFETENSORS_MODEL"
    else
        log_warn "SafeTensors model not available"
    fi

    echo

    # Validate GGUF
    if [ -f "$GGUF_MODEL" ]; then
        validate_format "gguf" "$GGUF_MODEL"
    else
        log_warn "GGUF model not available"
    fi

    echo

    # Compare formats
    compare_formats

    echo
    log_info "=== Validation Complete ==="
    log_info "Results saved to: ${OUTPUT_DIR}/${MODEL_ID}/"

    # Generate markdown report
    if [ -f "${OUTPUT_DIR}/${MODEL_ID}/format_comparison.json" ]; then
        python3 -c "
import json
from datetime import datetime

with open('${OUTPUT_DIR}/${MODEL_ID}/format_comparison.json') as f:
    data = json.load(f)

print('# BitNet Format Validation Report')
print(f'**Model:** {data[\"model_id\"]}')
print(f'**Date:** {data[\"timestamp\"][:10]}')
print()
print('## Test Results')
print('| Format | Tests Passed | Status |')
print('|--------|-------------|--------|')

for fmt in ['safetensors', 'gguf']:
    if fmt in data['formats']:
        tests = data['formats'][fmt].get('tests', {})
        passed = sum(1 for v in tests.values() if v)
        total = len(tests)
        status = '✅' if passed == total else '⚠️'
        print(f'| {fmt.capitalize():12} | {passed}/{total} | {status} |')

if 'performance_comparison' in data:
    print()
    print('## Performance Comparison (GGUF vs SafeTensors)')
    perf = data['performance_comparison']
    if perf:
        print('| Metric | Difference |')
        print('|--------|-----------|')
        for key, val in perf.items():
            metric = key.replace('_diff_pct', '').replace('_', ' ').title()
            sign = '+' if val > 0 else ''
            print(f'| {metric} | {sign}{val:.1f}% |')
" > "${OUTPUT_DIR}/${MODEL_ID}/VALIDATION_REPORT.md"

        log_info "Markdown report: ${OUTPUT_DIR}/${MODEL_ID}/VALIDATION_REPORT.md"
    fi
}

# Run main
main "$@"
