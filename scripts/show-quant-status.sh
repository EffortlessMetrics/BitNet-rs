#!/bin/bash
# Show current quantization support status in BitNet-rs

set -euo pipefail

echo "=== BitNet-rs Quantization Support Status ==="
echo

# Check if binaries exist
CLI_BIN="target/release/bitnet"
if [ ! -f "$CLI_BIN" ]; then
    echo "⚠️  Release binary not found. Building..."
    cargo build -p bitnet-cli --release --no-default-features --features "cpu,iq2s-ffi"
fi

echo "📊 Quantization Format Support:"
echo "================================"
echo

echo "✅ I2_S (BitNet Native 2-bit signed)"
echo "   - Implementation: Pure Rust"
echo "   - Dependencies: None"
echo "   - Feature flag: Always available with 'cpu'"
echo "   - Block size: 256 elements, 66 bytes"
echo "   - Status: FULLY IMPLEMENTED"
echo

echo "✅ IQ2_S (GGML/llama.cpp compatible)"
echo "   - Implementation: GGML FFI (C bridge)"
echo "   - Dependencies: Vendored GGML files"
echo "   - Feature flag: iq2s-ffi"
echo "   - Block size: Determined at runtime from GGML"
echo "   - Status: IMPLEMENTED (needs real GGML vendor)"
echo

echo "🔄 Other Formats (Planned):"
echo "   - Q4_0, Q4_1: 4-bit quantization"
echo "   - Q5_0, Q5_1: 5-bit quantization"
echo "   - Q8_0: 8-bit quantization"
echo "   - K-quants: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K"
echo

echo "📦 Build Configurations:"
echo "========================"
echo

echo "1. CPU with I2_S only (no external deps):"
echo "   cargo build --release --no-default-features --features cpu"
echo

echo "2. CPU with both I2_S and IQ2_S:"
echo "   cargo build --release --no-default-features --features 'cpu,iq2s-ffi'"
echo

echo "3. GPU/CUDA support:"
echo "   cargo build --release --no-default-features --features cuda"
echo

echo "🧪 Testing Quantization:"
echo "======================="
echo

echo "# Run I2_S tests (native):"
echo "cargo test -p bitnet-models --tests -- i2s"
echo

echo "# Run IQ2_S tests (FFI):"
echo "cargo test -p bitnet-models --tests --features iq2s-ffi -- iq2s"
echo

echo "# Test with a model:"
echo "BITNET_DETERMINISTIC=1 BITNET_SEED=42 \\"
echo "  $CLI_BIN run --model <path/to/model.gguf> \\"
echo "  --prompt 'Hello' --max-new-tokens 8"
echo

echo "⚠️  Important Notes:"
echo "==================="
echo "- IQ2_S currently uses stub GGML implementation"
echo "- Run 'cargo xtask vendor-ggml --commit <sha>' to get real GGML"
echo "- Both I2_S and IQ2_S dequantize to f32 at load time (correctness-first)"
echo "- Performance optimizations (on-the-fly dequant) planned post-alpha"
echo

echo "✨ Next Steps:"
echo "============="
echo "1. Vendor real GGML files from llama.cpp"
echo "2. Add pure-Rust IQ2_S implementation"
echo "3. Run parity tests between FFI and Rust paths"
echo "4. Enable on-the-fly dequantization for memory efficiency"