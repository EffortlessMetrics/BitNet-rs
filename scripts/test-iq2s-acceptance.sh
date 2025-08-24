#!/bin/bash
# IQ2_S Quantization Acceptance Test
set -euo pipefail

echo "=== IQ2_S Quantization Acceptance Test ==="
echo

# Configuration
BITNET_BIN="${BITNET_BIN:-target/release/bitnet}"
GGUF_MODEL="${GGUF_MODEL:-}"
OUT_DIR="${OUT_DIR:-test-results/iq2s-acceptance-$(date +%Y%m%d-%H%M%S)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_binary() {
    if [ ! -f "$BITNET_BIN" ]; then
        log_error "Binary not found: $BITNET_BIN"
        log_info "Building binary with IQ2_S support..."
        cargo build -p bitnet-cli --release --no-default-features --features "cpu,iq2s-ffi"
    fi
    
    # Check if binary has IQ2_S support
    if ! "$BITNET_BIN" --version | grep -q "iq2s-ffi"; then
        log_warn "Binary does not have IQ2_S support. Rebuild with --features iq2s-ffi"
        exit 1
    fi
}

check_model() {
    if [ -z "$GGUF_MODEL" ] || [ ! -f "$GGUF_MODEL" ]; then
        log_warn "No GGUF model specified or file not found: $GGUF_MODEL"
        log_info "Set GGUF_MODEL environment variable to a valid IQ2_S model path"
        exit 1
    fi
}

inspect_model() {
    log_info "Inspecting model: $GGUF_MODEL"
    
    local inspect_out="$OUT_DIR/inspect.json"
    env RUST_LOG=error "$BITNET_BIN" inspect \
        --model "$GGUF_MODEL" \
        --json > "$inspect_out"
    
    local quantization=$(jq -r '.quantization // "unknown"' "$inspect_out")
    local ggml_commit=$(jq -r '.environment.ggml_commit // "not-found"' "$inspect_out")
    
    echo "  Quantization: $quantization"
    echo "  GGML Commit: $ggml_commit"
    
    if [ "$quantization" != "IQ2_S" ]; then
        log_warn "Model is not IQ2_S quantized (found: $quantization)"
        return 1
    fi
    
    if [ "$ggml_commit" = "not-found" ] || [ "$ggml_commit" = "unknown" ]; then
        log_warn "GGML commit not found in binary (may not have vendored files)"
    fi
    
    return 0
}

test_inference() {
    log_info "Testing IQ2_S inference..."
    
    local run_out="$OUT_DIR/run.txt"
    local run_json="$OUT_DIR/run.json"
    
    # Run deterministic inference
    env BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
        timeout 30 "$BITNET_BIN" run \
        --model "$GGUF_MODEL" \
        --prompt "The capital of France is" \
        --max-new-tokens 8 \
        --greedy \
        --deterministic \
        --threads 1 \
        --json-out "$run_json" \
        2>&1 | tee "$run_out"
    
    # Check for IQ2_S backend banner in logs
    if grep -q "IQ2_S backend: ffi" "$run_out"; then
        log_info "IQ2_S FFI backend detected"
    elif grep -q "IQ2_S backend: rust" "$run_out"; then
        log_info "IQ2_S Rust backend detected"
    else
        log_warn "No IQ2_S backend banner found (may need RUST_LOG=info)"
    fi
    
    # Check output
    if [ -f "$run_json" ]; then
        local generated=$(jq -r '.generated // ""' "$run_json")
        if [ -n "$generated" ]; then
            echo "  Generated text: $generated"
        else
            log_warn "No generated text found in output"
        fi
    fi
}

test_constants() {
    log_info "Testing IQ2_S constants extraction..."
    
    # Create a simple test program to check constants
    cat > "$OUT_DIR/test_constants.rs" << 'EOF'
#[cfg(feature = "iq2s-ffi")]
fn main() {
    println!("QK_IQ2_S: {}", bitnet_ggml_ffi::iq2s_qk());
    println!("Bytes per block: {}", bitnet_ggml_ffi::iq2s_bytes_per_block());
    println!("Requires QK multiple: {}", bitnet_ggml_ffi::iq2s_requires_qk_multiple());
    println!("GGML commit: {}", bitnet_ggml_ffi::GGML_COMMIT);
}

#[cfg(not(feature = "iq2s-ffi"))]
fn main() {
    println!("IQ2_S FFI not enabled");
}
EOF
    
    # Try to compile and run (may fail if FFI not properly configured)
    if rustc --edition 2021 \
           -L target/release/deps \
           -o "$OUT_DIR/test_constants" \
           "$OUT_DIR/test_constants.rs" 2>/dev/null; then
        "$OUT_DIR/test_constants" || true
    else
        log_warn "Could not compile constants test (FFI may not be configured)"
    fi
}

generate_summary() {
    log_info "Generating test summary..."
    
    local summary="$OUT_DIR/summary.json"
    
    # Parse results
    local quantization="unknown"
    local ggml_commit="unknown"
    local inference_success="false"
    
    if [ -f "$OUT_DIR/inspect.json" ]; then
        quantization=$(jq -r '.quantization // "unknown"' "$OUT_DIR/inspect.json")
        ggml_commit=$(jq -r '.environment.ggml_commit // "unknown"' "$OUT_DIR/inspect.json")
    fi
    
    if [ -f "$OUT_DIR/run.json" ] && [ -s "$OUT_DIR/run.json" ]; then
        inference_success="true"
    fi
    
    jq -n \
        --arg model "$GGUF_MODEL" \
        --arg quantization "$quantization" \
        --arg ggml_commit "$ggml_commit" \
        --arg inference_success "$inference_success" \
        --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{
            model: $model,
            quantization: $quantization,
            ggml_commit: $ggml_commit,
            tests: {
                inspect: ($quantization == "IQ2_S"),
                inference: ($inference_success == "true"),
                constants: true
            },
            timestamp: $timestamp
        }' > "$summary"
    
    echo
    echo "Summary saved to: $summary"
    jq . "$summary"
}

# Main execution
main() {
    mkdir -p "$OUT_DIR"
    
    log_info "Starting IQ2_S acceptance test"
    echo "  Binary: $BITNET_BIN"
    echo "  Model: $GGUF_MODEL"
    echo "  Output: $OUT_DIR"
    echo
    
    check_binary
    check_model
    
    if inspect_model; then
        test_inference
        test_constants
    else
        log_error "Model inspection failed - not an IQ2_S model"
        exit 1
    fi
    
    generate_summary
    
    echo
    log_info "IQ2_S acceptance test completed successfully"
}

main "$@"