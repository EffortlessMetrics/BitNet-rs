#!/usr/bin/env bash
set -euo pipefail

# Validate equivalence between SafeTensors and GGUF formats
# This ensures both formats produce identical results within tolerance

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Configuration
BITNET_BIN=$(find_bitnet_binary)
MODELS_DIR="${MODELS_DIR:-models}"
OUTPUT_DIR=$(ensure_output_dir "validation_results")
TOLERANCE_NLL="${TOLERANCE_NLL:-0.01}"  # 1e-2 for FP32↔FP32
TOLERANCE_LOGIT="${TOLERANCE_LOGIT:-0.60}"  # τ-b correlation threshold

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Check dependencies
check_deps() {
    local missing=()
    
    [ ! -x "$BITNET_BIN" ] && missing+=("bitnet CLI")
    command -v python3 >/dev/null 2>&1 || missing+=("python3")
    command -v jq >/dev/null 2>&1 || missing+=("jq")
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        log_info "Binary path checked: $BITNET_BIN"
        exit 1
    fi
}

# Find model paths
find_model_paths() {
    local model_id="$1"
    local safetensors_path="${MODELS_DIR}/${model_id}/safetensors/model.safetensors"
    local gguf_path="${MODELS_DIR}/${model_id}/gguf/model.gguf"
    local tokenizer_path="${MODELS_DIR}/${model_id}/safetensors/tokenizer.json"
    
    if [ ! -f "$safetensors_path" ]; then
        log_error "SafeTensors model not found: $safetensors_path"
        return 1
    fi
    
    if [ ! -f "$gguf_path" ]; then
        log_warn "GGUF model not found: $gguf_path"
        log_info "Attempting conversion..."
        
        python3 scripts/convert_safetensors_to_gguf.py \
            "${MODELS_DIR}/${model_id}/safetensors" \
            "$gguf_path" || {
            log_error "Conversion failed"
            return 1
        }
    fi
    
    echo "$safetensors_path:$gguf_path:$tokenizer_path"
}

# Run inference and capture output
run_inference() {
    local model_path="$1"
    local tokenizer_path="$2"
    local prompt="$3"
    local output_file="$4"
    local format="${5:-auto}"
    
    # Set deterministic mode
    export BITNET_DETERMINISTIC=1
    export BITNET_SEED=42
    export RAYON_NUM_THREADS=1
    
    # Run inference with JSON output
    "$BITNET_BIN" run \
        --model "$model_path" \
        --tokenizer "$tokenizer_path" \
        --model-format "$format" \
        --prompt "$prompt" \
        --max-new-tokens 20 \
        --greedy \
        --deterministic \
        --threads 1 \
        --dump-logit-steps 20 \
        --logits-topk 100 \
        --json-out "$output_file" 2>/dev/null || {
        log_error "Inference failed for $model_path"
        return 1
    }
}

# Evaluate perplexity
eval_perplexity() {
    local model_path="$1"
    local tokenizer_path="$2"
    local text_file="$3"
    local output_file="$4"
    local format="${5:-auto}"
    
    # Set deterministic mode
    export BITNET_DETERMINISTIC=1
    export BITNET_SEED=42
    export RAYON_NUM_THREADS=1
    
    "$BITNET_BIN" eval \
        --model "$model_path" \
        --tokenizer "$tokenizer_path" \
        --model-format "$format" \
        --text-file "$text_file" \
        --json-out "$output_file" 2>/dev/null || {
        log_error "Evaluation failed for $model_path"
        return 1
    }
}

# Compare JSON outputs
compare_outputs() {
    local file1="$1"
    local file2="$2"
    local metric="$3"
    
    python3 -c "
import json
import sys
import numpy as np
from scipy import stats

with open('$file1') as f:
    data1 = json.load(f)
with open('$file2') as f:
    data2 = json.load(f)

metric = '$metric'

if metric == 'nll':
    # Compare mean NLL
    nll1 = data1.get('mean_nll', 0)
    nll2 = data2.get('mean_nll', 0)
    diff = abs(nll1 - nll2)
    threshold = $TOLERANCE_NLL
    
    passed = diff <= threshold
    print(f'NLL1: {nll1:.6f}')
    print(f'NLL2: {nll2:.6f}')
    print(f'Diff: {diff:.6f} (threshold: {threshold})')
    sys.exit(0 if passed else 1)
    
elif metric == 'logits':
    # Compare logit distributions using Kendall's tau-b
    logits1 = data1.get('logits_dump', [])
    logits2 = data2.get('logits_dump', [])
    
    if not logits1 or not logits2:
        print('No logits to compare')
        sys.exit(1)
    
    correlations = []
    for step1, step2 in zip(logits1, logits2):
        # Extract top-k logits
        top1 = {e['token_id']: e['logit'] for e in step1.get('top_logits', [])}
        top2 = {e['token_id']: e['logit'] for e in step2.get('top_logits', [])}
        
        # Get common tokens
        common_ids = set(top1.keys()) & set(top2.keys())
        if len(common_ids) < 2:
            continue
            
        # Compute correlation
        ids = sorted(common_ids)
        vals1 = [top1[i] for i in ids]
        vals2 = [top2[i] for i in ids]
        
        tau, _ = stats.kendalltau(vals1, vals2)
        if not np.isnan(tau):
            correlations.append(tau)
    
    if correlations:
        median_tau = np.median(correlations)
        threshold = $TOLERANCE_LOGIT
        passed = median_tau >= threshold
        
        print(f'Median τ-b: {median_tau:.3f} (threshold: {threshold})')
        print(f'Min τ-b: {min(correlations):.3f}')
        print(f'Max τ-b: {max(correlations):.3f}')
        sys.exit(0 if passed else 1)
    else:
        print('No valid correlations computed')
        sys.exit(1)
        
elif metric == 'tokens':
    # Compare generated token sequences
    tokens1 = data1.get('generated_ids', [])
    tokens2 = data2.get('generated_ids', [])
    
    matches = sum(1 for t1, t2 in zip(tokens1, tokens2) if t1 == t2)
    total = min(len(tokens1), len(tokens2))
    
    if total > 0:
        accuracy = matches / total
        print(f'Token match: {matches}/{total} ({accuracy:.1%})')
        sys.exit(0 if accuracy >= 0.95 else 1)
    else:
        print('No tokens to compare')
        sys.exit(1)
"
}

# Validate a model
validate_model() {
    local model_id="$1"
    
    log_info "Validating format parity for: $model_id"
    
    # Get paths
    local paths
    paths=$(find_model_paths "$model_id") || return 1
    
    IFS=':' read -r safetensors_path gguf_path tokenizer_path <<< "$paths"
    
    # Create output directory
    local result_dir="${OUTPUT_DIR}/${model_id}"
    mkdir -p "$result_dir"
    
    # Test prompt
    local test_prompt="The capital of France is"
    local test_text="${result_dir}/test_text.txt"
    echo "The quick brown fox jumps over the lazy dog." > "$test_text"
    echo "Machine learning is a subset of artificial intelligence." >> "$test_text"
    
    local all_passed=true
    
    # Test 1: Token generation
    log_info "Test 1: Token generation consistency"
    
    run_inference "$safetensors_path" "$tokenizer_path" "$test_prompt" \
        "${result_dir}/gen_safetensors.json" "safetensors"
    
    run_inference "$gguf_path" "$tokenizer_path" "$test_prompt" \
        "${result_dir}/gen_gguf.json" "gguf"
    
    if compare_outputs "${result_dir}/gen_safetensors.json" \
                      "${result_dir}/gen_gguf.json" "tokens"; then
        log_info "  ✓ Token generation: PASSED"
    else
        log_warn "  ✗ Token generation: FAILED"
        all_passed=false
    fi
    
    # Test 2: Logit correlation
    log_info "Test 2: Logit distribution correlation"
    
    if compare_outputs "${result_dir}/gen_safetensors.json" \
                      "${result_dir}/gen_gguf.json" "logits"; then
        log_info "  ✓ Logit correlation: PASSED"
    else
        log_warn "  ✗ Logit correlation: FAILED"
        all_passed=false
    fi
    
    # Test 3: Perplexity (NLL)
    log_info "Test 3: Perplexity (mean NLL) consistency"
    
    eval_perplexity "$safetensors_path" "$tokenizer_path" "$test_text" \
        "${result_dir}/ppl_safetensors.json" "safetensors"
    
    eval_perplexity "$gguf_path" "$tokenizer_path" "$test_text" \
        "${result_dir}/ppl_gguf.json" "gguf"
    
    if compare_outputs "${result_dir}/ppl_safetensors.json" \
                      "${result_dir}/ppl_gguf.json" "nll"; then
        log_info "  ✓ Perplexity: PASSED"
    else
        log_warn "  ✗ Perplexity: FAILED"
        all_passed=false
    fi
    
    # Generate parity report
    local report="${result_dir}/parity_report.json"
    python3 -c "
import json
from datetime import datetime

report = {
    'model_id': '$model_id',
    'timestamp': datetime.utcnow().isoformat() + 'Z',
    'safetensors_path': '$safetensors_path',
    'gguf_path': '$gguf_path',
    'tests': {
        'token_generation': $([ "$all_passed" = true ] && echo 'true' || echo 'false'),
        'logit_correlation': $([ "$all_passed" = true ] && echo 'true' || echo 'false'),
        'perplexity': $([ "$all_passed" = true ] && echo 'true' || echo 'false'),
    },
    'passed': $([ "$all_passed" = true ] && echo 'true' || echo 'false'),
    'tolerances': {
        'nll': $TOLERANCE_NLL,
        'logit_tau_b': $TOLERANCE_LOGIT,
    }
}

with open('$report', 'w') as f:
    json.dump(report, f, indent=2)
"
    
    log_info "Parity report saved to: $report"
    
    if [ "$all_passed" = true ]; then
        log_info "✓ All parity tests PASSED for $model_id"
        return 0
    else
        log_warn "✗ Some parity tests FAILED for $model_id"
        return 1
    fi
}

# Main validation
main() {
    # Set up deterministic environment
    setup_deterministic_env
    
    # Print platform information
    print_platform_banner
    
    log_info "=== BitNet Format Parity Validation ==="
    
    # Check dependencies
    check_deps
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Get model to validate
    local model_id="${1:-bitnet_b1_58-3B}"
    
    # Run validation
    if validate_model "$model_id"; then
        log_info "=== Validation PASSED ==="
        exit 0
    else
        log_error "=== Validation FAILED ==="
        exit 1
    fi
}

# Run if not sourced
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi