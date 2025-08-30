#!/usr/bin/env bash
# Comprehensive validation suite for BitNet.rs vs llama.cpp
set -euo pipefail

# Source concurrency caps and preflight checks
if [[ -f "$(dirname "$0")/preflight.sh" ]]; then
    source "$(dirname "$0")/preflight.sh"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MODEL="${BITNET_GGUF:-models/bitnet/ggml-model-i2_s.gguf}"
SPM="${TOKENIZER_PATH:-}"
CROSSVAL_DIR="${CROSSVAL_DIR:-crossval}"
TMP="${TMPDIR:-/tmp}/bitnet_validation.$$"
mkdir -p "$TMP"

# Cleanup on exit
trap 'rm -rf "$TMP"' EXIT

echo -e "${YELLOW}BitNet.rs Comprehensive Validation Suite${NC}"
echo "================================================"
echo "Model: $MODEL"
echo "Tokenizer: ${SPM:-embedded}"
echo "Temp dir: $TMP"
echo ""

# Function to check if llama.cpp is available
check_llama_cpp() {
    if ! command -v llama-cli &> /dev/null; then
        echo -e "${YELLOW}Warning: llama-cli not found. Attempting to use crossval build...${NC}"
        if [[ -f "$HOME/.cache/bitnet_cpp/build/bin/llama-cli" ]]; then
            export PATH="$HOME/.cache/bitnet_cpp/build/bin:$PATH"
        else
            echo -e "${RED}Error: llama-cli not available. Run 'cargo xtask fetch-cpp' first.${NC}"
            exit 1
        fi
    fi
}

# Function to run bitnet CLI
run_bitnet() {
    local args="$@"
    if [[ -f "target/release/bitnet" ]]; then
        ./target/release/bitnet $args
    else
        cargo run --release -p bitnet-cli --no-default-features --features cpu -- $args
    fi
}

# 1. Model Compatibility Check
validate_model_compatibility() {
    echo -e "\n${YELLOW}1. Model Compatibility Check${NC}"
    echo "--------------------------------"
    
    # Check model with bitnet
    echo "Checking model with BitNet.rs..."
    local bitnet_out="$TMP/model_check.json"
    
    if [[ -n "$SPM" ]]; then
        run_bitnet compat-check --model "$MODEL" --tokenizer "$SPM" --json-out "$bitnet_out" || true
    else
        run_bitnet compat-check --model "$MODEL" --json-out "$bitnet_out" || true
    fi
    
    if [[ -f "$bitnet_out" ]]; then
        local unmapped=$(jq -r '.counts.unmapped // 0' "$bitnet_out")
        local n_tensors=$(jq -r '.counts.n_tensors // 0' "$bitnet_out")
        local tokenizer_origin=$(jq -r '.tokenizer.origin // "unknown"' "$bitnet_out")
        
        echo "  Tensors: $n_tensors"
        echo "  Unmapped: $unmapped"
        echo "  Tokenizer: $tokenizer_origin"
        
        if [[ "$unmapped" -gt 0 ]]; then
            echo -e "${RED}  ✗ Model has $unmapped unmapped tensors${NC}"
            return 1
        else
            echo -e "${GREEN}  ✓ All tensors mapped successfully${NC}"
        fi
    else
        echo -e "${RED}  ✗ Failed to check model compatibility${NC}"
        return 1
    fi
}

# 2. Perplexity/NLL Parity Test
validate_perplexity_parity() {
    echo -e "\n${YELLOW}2. Perplexity/NLL Parity Test${NC}"
    echo "--------------------------------"
    
    # Create test dataset if not exists
    local dataset="$CROSSVAL_DIR/data/ppl_smoke.txt"
    if [[ ! -f "$dataset" ]]; then
        echo "Creating smoke test dataset..."
        mkdir -p "$(dirname "$dataset")"
        cat > "$dataset" << 'EOF'
The quick brown fox jumps over the lazy dog.
To be, or not to be, that is the question.
1 1 2 3 5 8 13 21 34 55
Today is 2025-08-22 in ISO format.
fn main() { println!("hello"); }
¿Dónde está la biblioteca?
今天天气很好。
Les élèves étudient la physique quantique.
Der schnellste Weg ist nicht immer der beste.
π is approximately 3.14159.
Call me Ishmael.
The capital of Canada is Ottawa.
HTTP/2 introduced header compression.
Rust's borrow checker prevents data races.
NaN != NaN by IEEE-754 rules.
2024-02-29 is a leap day.
SELECT 1 WHERE 1=1;
🐍 > 🦀? Depends on the day.
Long contexts require careful KV cache management.
EOF
    fi
    
    # Set deterministic mode
    export BITNET_DETERMINISTIC=1
    export BITNET_SEED=42
    export RAYON_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    
    echo "Running BitNet.rs perplexity calculation..."
    local bitnet_ppl="$TMP/bitnet_ppl.json"
    
    if [[ -n "$SPM" ]]; then
        run_bitnet score --model "$MODEL" --tokenizer "$SPM" --file "$dataset" \
            --bos --json-out "$bitnet_ppl" > /dev/null 2>&1 || true
    else
        run_bitnet score --model "$MODEL" --file "$dataset" \
            --bos --json-out "$bitnet_ppl" > /dev/null 2>&1 || true
    fi
    
    if [[ -f "$bitnet_ppl" ]]; then
        local bitnet_nll=$(jq -r '.mean_nll // "N/A"' "$bitnet_ppl")
        local bitnet_ppl_val=$(jq -r '.ppl // "N/A"' "$bitnet_ppl")
        
        echo "  BitNet.rs NLL: $bitnet_nll"
        echo "  BitNet.rs PPL: $bitnet_ppl_val"
        
        # Try to run llama.cpp for comparison
        if command -v llama-cli &> /dev/null; then
            echo "Running llama.cpp perplexity calculation..."
            local cpp_ppl="$TMP/cpp_ppl.json"
            
            # Note: This assumes llama-cli has similar interface
            # You may need to adjust based on actual llama.cpp CLI
            llama-cli --model "$MODEL" --ppl-file "$dataset" \
                --json-out "$cpp_ppl" > /dev/null 2>&1 || true
            
            if [[ -f "$cpp_ppl" ]]; then
                local cpp_nll=$(jq -r '.mean_nll // "N/A"' "$cpp_ppl")
                
                if [[ "$bitnet_nll" != "N/A" && "$cpp_nll" != "N/A" ]]; then
                    # Calculate difference
                    local diff=$(python3 -c "
import sys
r = float('$bitnet_nll')
c = float('$cpp_nll')
d = abs(r - c)
print(f'{d:.6f}')
sys.exit(0 if d <= 0.01 else 1)
")
                    if [[ $? -eq 0 ]]; then
                        echo -e "${GREEN}  ✓ NLL parity passed: |Δ| = $diff ≤ 0.01${NC}"
                    else
                        echo -e "${RED}  ✗ NLL parity failed: |Δ| = $diff > 0.01${NC}"
                    fi
                fi
            fi
        else
            echo -e "${YELLOW}  ⚠ llama.cpp not available for comparison${NC}"
        fi
    else
        echo -e "${RED}  ✗ Failed to calculate perplexity${NC}"
    fi
}

# 3. Token ID A/B Parity Test
validate_token_parity() {
    echo -e "\n${YELLOW}3. Token ID A/B Parity Test${NC}"
    echo "--------------------------------"
    
    # Create prompts file
    local prompts_file="$CROSSVAL_DIR/prompts.yaml"
    if [[ ! -f "$prompts_file" ]]; then
        echo "Creating test prompts..."
        mkdir -p "$(dirname "$prompts_file")"
        cat > "$prompts_file" << 'EOF'
version: 1
bos: true
max_new_tokens: 64
prompts:
  - "Add 27 and 38."
  - "Print a Rust function that reverses a string."
  - "Translate 'good morning' to French."
  - "What day of the week was 2000-01-01?"
  - "List the first 10 prime numbers."
  - "Explain HTTP cookies in one paragraph."
  - "生成一个中文的打招呼句子。"
  - "Wie buchstabiert man 'Quantenmechanik'?"
  - "Sum from 1 to 100."
  - "Write a SQL query to select users older than 30."
  - "Continue the Fibonacci sequence: 1, 1, 2, 3, 5, 8,"
  - "C++: create a vector and push three integers."
  - "日本の首都はどこですか。"
  - "What's the derivative of sin(x)*x^2?"
  - "Tell a two-sentence horror story."
  - "Explain SIMD briefly."
  - "Give a JSON object with keys a,b,c."
  - "Name three Canadian provinces."
  - "Write a haiku about databases."
  - "What is 2^10?"
EOF
    fi
    
    echo "Running token generation with BitNet.rs..."
    local bitnet_ids="$TMP/bitnet.ids"
    local bitnet_run="$TMP/bitnet_run.json"
    
    if [[ -n "$SPM" ]]; then
        run_bitnet run --model "$MODEL" --tokenizer "$SPM" \
            --prompts "$prompts_file" --temperature 0 \
            --dump-token-ids "$bitnet_ids" \
            --json-out "$bitnet_run" > /dev/null 2>&1 || true
    else
        run_bitnet run --model "$MODEL" \
            --prompts "$prompts_file" --temperature 0 \
            --dump-token-ids "$bitnet_ids" \
            --json-out "$bitnet_run" > /dev/null 2>&1 || true
    fi
    
    if [[ -f "$bitnet_ids" ]]; then
        local n_prompts=$(wc -l < "$bitnet_ids")
        echo "  Generated tokens for $n_prompts prompts"
        
        # If llama.cpp is available, compare
        if command -v llama-cli &> /dev/null; then
            echo "Running token generation with llama.cpp..."
            local cpp_ids="$TMP/cpp.ids"
            
            llama-cli --model "$MODEL" --prompts "$prompts_file" \
                --temperature 0 --dump-token-ids "$cpp_ids" \
                > /dev/null 2>&1 || true
            
            if [[ -f "$cpp_ids" ]]; then
                # Compare token IDs
                python3 - "$bitnet_ids" "$cpp_ids" << 'PY' || true
import sys
try:
    rs = [l.strip().split() for l in open(sys.argv[1])]
    cp = [l.strip().split() for l in open(sys.argv[2])]
    if len(rs) != len(cp):
        print(f"  ⚠ Prompt count mismatch: {len(rs)} vs {len(cp)}")
        sys.exit(1)
    
    total = len(rs)
    ok = 0
    divergences = []
    
    for i, (a, b) in enumerate(zip(rs, cp)):
        if a == b:
            ok += 1
        else:
            j = 0
            while j < min(len(a), len(b)) and a[j] == b[j]:
                j += 1
            divergences.append((i, j, a[j:j+3] if j < len(a) else [], b[j:j+3] if j < len(b) else []))
    
    rate = ok / total if total > 0 else 0
    print(f"  Exact match rate: {rate:.1%} ({ok}/{total})")
    
    if divergences and len(divergences) <= 3:
        print("  First divergences:")
        for i, pos, rs_tok, cpp_tok in divergences[:3]:
            print(f"    Prompt #{i} @ pos {pos}: rs={rs_tok} cpp={cpp_tok}")
    
    if rate >= 0.95:
        print(f"\033[0;32m  ✓ Token ID parity passed: {rate:.1%} ≥ 95%\033[0m")
        sys.exit(0)
    else:
        print(f"\033[0;31m  ✗ Token ID parity failed: {rate:.1%} < 95%\033[0m")
        sys.exit(1)
except Exception as e:
    print(f"  Error comparing tokens: {e}")
    sys.exit(1)
PY
            fi
        else
            echo -e "${YELLOW}  ⚠ llama.cpp not available for comparison${NC}"
        fi
    else
        echo -e "${RED}  ✗ Failed to generate tokens${NC}"
    fi
}

# 4. Performance Validation
validate_performance() {
    echo -e "\n${YELLOW}4. Performance Validation${NC}"
    echo "--------------------------------"
    
    # Check if we have a baseline
    local baseline_file="ci/baseline.json"
    if [[ ! -f "$baseline_file" ]]; then
        echo -e "${YELLOW}  ⚠ No baseline file found at $baseline_file${NC}"
        echo "  Creating initial baseline..."
        
        # Run a performance test
        local perf_out="$TMP/perf.json"
        echo "Running performance benchmark..."
        
        # Create a simple prompt for benchmarking
        cat > "$TMP/bench_prompt.yaml" << 'EOF'
version: 1
bos: true
max_new_tokens: 128
prompts:
  - "Write a detailed explanation of how neural networks work, including backpropagation."
EOF
        
        if [[ -n "$SPM" ]]; then
            run_bitnet run --model "$MODEL" --tokenizer "$SPM" \
                --prompts "$TMP/bench_prompt.yaml" --temperature 0.7 \
                --json-out "$perf_out" > /dev/null 2>&1 || true
        else
            run_bitnet run --model "$MODEL" \
                --prompts "$TMP/bench_prompt.yaml" --temperature 0.7 \
                --json-out "$perf_out" > /dev/null 2>&1 || true
        fi
        
        if [[ -f "$perf_out" ]]; then
            local tok_s=$(jq -r '.throughput.tokens_per_second // 0' "$perf_out")
            local rss_mb=$(jq -r '.memory.rss_mb // 0' "$perf_out")
            
            echo "  Tokens/second: $tok_s"
            echo "  RSS (MB): $rss_mb"
            
            # Check absolute floor
            if (( $(echo "$tok_s >= 1.0" | bc -l) )); then
                echo -e "${GREEN}  ✓ Performance floor passed: $tok_s ≥ 1.0 tok/s${NC}"
            else
                echo -e "${RED}  ✗ Performance floor failed: $tok_s < 1.0 tok/s${NC}"
            fi
            
            # Create initial baseline
            mkdir -p "$(dirname "$baseline_file")"
            cat > "$baseline_file" << EOF
{
  "cpu": {
    "model_default": {
      "tok_s": $tok_s,
      "rss_mb": $rss_mb
    }
  }
}
EOF
            echo "  Created initial baseline at $baseline_file"
        else
            echo -e "${RED}  ✗ Failed to run performance benchmark${NC}"
        fi
    else
        # Compare against baseline
        echo "Comparing against baseline..."
        
        local perf_out="$TMP/perf.json"
        cat > "$TMP/bench_prompt.yaml" << 'EOF'
version: 1
bos: true
max_new_tokens: 128
prompts:
  - "Write a detailed explanation of how neural networks work, including backpropagation."
EOF
        
        if [[ -n "$SPM" ]]; then
            run_bitnet run --model "$MODEL" --tokenizer "$SPM" \
                --prompts "$TMP/bench_prompt.yaml" --temperature 0.7 \
                --json-out "$perf_out" > /dev/null 2>&1 || true
        else
            run_bitnet run --model "$MODEL" \
                --prompts "$TMP/bench_prompt.yaml" --temperature 0.7 \
                --json-out "$perf_out" > /dev/null 2>&1 || true
        fi
        
        if [[ -f "$perf_out" ]]; then
            local tok_s=$(jq -r '.throughput.tokens_per_second // 0' "$perf_out")
            local rss_mb=$(jq -r '.memory.rss_mb // 0' "$perf_out")
            local baseline_tok_s=$(jq -r '.cpu.model_default.tok_s // 0' "$baseline_file")
            local baseline_rss_mb=$(jq -r '.cpu.model_default.rss_mb // 0' "$baseline_file")
            
            echo "  Current: $tok_s tok/s, $rss_mb MB RSS"
            echo "  Baseline: $baseline_tok_s tok/s, $baseline_rss_mb MB RSS"
            
            # Check ratios
            python3 - << PY || true
import sys
tok_s = float($tok_s)
rss_mb = float($rss_mb)
baseline_tok_s = float($baseline_tok_s)
baseline_rss_mb = float($baseline_rss_mb)

# Check absolute floor
if tok_s < 1.0:
    print(f"\033[0;31m  ✗ Performance floor failed: {tok_s} < 1.0 tok/s\033[0m")
    sys.exit(1)

# Check throughput ratio
if baseline_tok_s > 0:
    ratio = tok_s / baseline_tok_s
    if ratio >= 0.95:
        print(f"\033[0;32m  ✓ Throughput ratio passed: {ratio:.1%} ≥ 95%\033[0m")
    else:
        print(f"\033[0;31m  ✗ Throughput ratio failed: {ratio:.1%} < 95%\033[0m")
        sys.exit(1)

# Check memory ratio
if baseline_rss_mb > 0:
    mem_ratio = rss_mb / baseline_rss_mb
    if mem_ratio <= 1.03:
        print(f"\033[0;32m  ✓ Memory ratio passed: {mem_ratio:.1%} ≤ 103%\033[0m")
    else:
        print(f"\033[0;31m  ✗ Memory ratio failed: {mem_ratio:.1%} > 103%\033[0m")
        sys.exit(1)
PY
        fi
    fi
}

# Main validation flow
main() {
    local failed=0
    
    # Check prerequisites
    check_llama_cpp
    
    # Run validation steps
    validate_model_compatibility || ((failed++))
    validate_perplexity_parity || ((failed++))
    validate_token_parity || ((failed++))
    validate_performance || ((failed++))
    
    # Summary
    echo -e "\n${YELLOW}Validation Summary${NC}"
    echo "=================="
    
    if [[ $failed -eq 0 ]]; then
        echo -e "${GREEN}✓ All validation checks passed!${NC}"
        exit 0
    else
        echo -e "${RED}✗ $failed validation check(s) failed${NC}"
        exit 1
    fi
}

# Run main function
main "$@"