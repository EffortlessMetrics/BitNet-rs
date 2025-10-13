#!/usr/bin/env bash
# Generate correction policy file for a GGUF model
set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf> [output.yml]}"
OUTPUT="${2:-correction-policy.yml}"

if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model file not found: $MODEL" >&2
    exit 1
fi

echo "Computing fingerprint for: $MODEL"
FP=$(sha256sum "$MODEL" | awk '{print "sha256-"$1}')
echo "Fingerprint: $FP"

cat > "$OUTPUT" <<EOF
version: 1
models:
  - fingerprint: "$FP"
    notes: "BitNet model with inverted I2_S scales in Q/K/V projections"
    corrections:
      # Override I2_S dequantization for attention projections
      # Use inv=true to invert the scales (1/scale instead of scale)
      - type: I2S_DEQUANT_OVERRIDE
        tensors:
          # LLaMA/HF-style names
          - "q_proj.weight"
          - "k_proj.weight"
          - "v_proj.weight"
          # Microsoft BitNet-style names
          - "wq.weight"
          - "wk.weight"
          - "wv.weight"
          # Alternative naming patterns
          - "attn_q.weight"
          - "attn_k.weight"
          - "attn_v.weight"
        inv: true
        k: 1.0
EOF

echo "Policy file written to: $OUTPUT"
echo ""
echo "To use this policy:"
echo "  export BITNET_CORRECTION_POLICY=$OUTPUT"
echo "  export BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1"
echo '  RUST_LOG=info,bitnet_models=debug ./scripts/debug_inference.sh \'
echo "    \"$MODEL\" \\"
echo '    models/llama3-tokenizer/tokenizer.json \'
echo '    "Answer in one short sentence: Why is the sky blue?"'
