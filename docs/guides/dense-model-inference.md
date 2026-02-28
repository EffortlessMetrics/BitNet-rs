# Dense Model Inference in BitNet-rs

BitNet-rs was originally designed for 1.58-bit BitNet models but now supports dense
FP16/BF16 transformer models from the broader SLM ecosystem (Phi-4, LLaMA, Qwen, Gemma, etc.).

## How Dense Models Flow Through the Pipeline

### 1. Model Loading

Dense models are loaded via two paths:

- **SafeTensors**: The HuggingFace loader (`bitnet-models/src/formats/huggingface.rs`) reads
  `.safetensors` files directly. BF16 weights are automatically converted to F32 during loading
  via `half::bf16::from_bits(h).to_f32()`.

- **GGUF**: Models converted to GGUF format (e.g., via `bitnet-st2gguf`) are loaded by the
  GGUF reader. The format supports multiple quantization types including F16 and F32.

### 2. Architecture Detection

During loading, the architecture string (from GGUF `general.architecture` or HuggingFace
`config.json`) is passed to `ModelConfig::apply_architecture_defaults()`. This queries the
`ArchitectureRegistry` to set:

- **Normalization type**: RMSNorm (Phi-4, LLaMA, Qwen, Mistral) or LayerNorm (GPT-2, Falcon)
- **Activation function**: SiLU (most modern models), GeLU (GPT-2, Gemma), or ReLU² (BitNet)
- **Context length**: Model-specific defaults (e.g., 16384 for Phi-4, 131072 for Qwen2.5)

### 3. Transformer Execution

The `TransformerBlock` in `bitnet-transformer` dispatches based on config:

```
Input → Norm(config.norm_type) → Attention(GQA) → Residual
      → Norm(config.norm_type) → FFN(config.activation_type) → Residual
```

- **Normalization**: `build_norm_layer()` creates either `candle_nn::LayerNorm` or
  `candle_nn::RmsNorm` based on `NormType`.
- **Activation**: `apply_activation()` dispatches to SiLU (`x * sigmoid(x)`), GeLU, or ReLU².
- **Linear layers**: Dense models use standard `candle_nn::Linear` — no quantization dispatch.
  The QK256/BitNet quantization path is only activated for I2_S quantized tensors.

### 4. Grouped-Query Attention (GQA)

All supported architectures use GQA with configurable group sizes:

| Model     | Heads | KV Heads | Group Size |
|-----------|-------|----------|------------|
| Phi-4     | 40    | 10       | 4          |
| LLaMA-3   | 32    | 8        | 4          |
| Qwen2     | 28    | 4        | 7          |
| Gemma-2   | 16    | 8        | 2          |
| Mistral   | 32    | 8        | 4          |

The `num_key_value_heads` config field controls GQA behavior. When set to 0, it defaults
to `num_heads` (standard multi-head attention).

## Supported Architectures

BitNet-rs recognizes 36+ architecture strings across 19+ model families. Use the CLI to
list them all:

```bash
cargo run -p bitnet-cli --no-default-features --features cpu -- list-architectures
cargo run -p bitnet-cli --no-default-features --features cpu -- list-architectures --json
```

## Performance Considerations

- **Memory**: Dense FP16 models require ~2 bytes per parameter. A 14B model needs ~28 GB.
  Memory-mapped I/O is used for large models.
- **KV Cache**: For 16K context with 40 layers and 128-dim heads, the KV cache is ~3 GB (FP16).
- **Throughput**: Dense models use standard matrix multiplication (no specialized bit-packing
  kernels), so throughput depends on hardware BLAS performance.

## Quantization Options

For faster inference on dense models, consider:

1. **F16**: Default for most models. Good balance of speed and accuracy.
2. **BF16→F32**: Automatic conversion during loading. Full precision but 2× memory.
3. **Future**: int8/int4 quantization paths are planned for v0.3+.
