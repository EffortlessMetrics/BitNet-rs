# Model Loading Quick Reference

## Key Files

| File | Purpose |
|------|---------|
| `crates/bitnet-models/src/loader.rs` | Main model loader dispatcher with format detection |
| `crates/bitnet-models/src/formats/gguf/reader.rs` | GGUF file parser (v2/v3) |
| `crates/bitnet-models/src/formats/gguf/loader.rs` | GGUF format loader implementation |
| `crates/bitnet-models/src/bitnet.rs` | BitNet model construction from loaded tensors |
| `crates/bitnet-models/src/weight_mapper.rs` | Tensor name remapping and normalization |
| `crates/bitnet-models/src/production_loader.rs` | Production-grade loader with validation |
| `crates/bitnet-tokenizers/src/gguf_loader.rs` | Tokenizer extraction from GGUF metadata |

## Model Loading Flow (Simplified)

```
File → Format Detection → GgufReader → extract_config() → load_tensors() 
→ Validation → build_transformer() → BitNetModel → Ready
```

## Critical Metadata Keys

### Model Configuration
- `llama.vocab_size` / `bitnet-b1.58.vocab_size`
- `llama.embedding_length` / `bitnet-b1.58.embedding_length`
- `llama.block_count` / `bitnet-b1.58.block_count`
- `llama.attention.head_count` / `bitnet-b1.58.attention.head_count`
- `llama.attention.head_count_kv` / `bitnet-b1.58.attention.head_count_kv`

### RoPE Configuration
- `llama.rope.freq_base` / `bitnet-b1.58.rope.freq_base`
- `llama.rope.dimension_count`

### Tokenizer Special Tokens
- `tokenizer.ggml.bos_token_id`
- `tokenizer.ggml.eos_token_id`
- `tokenizer.ggml.eot_token_id` (LLaMA-3)
- `tokenizer.ggml.model` ("llama" for SPM, "gpt2"/"bpe" for BPE)

## Validation Stages

1. **GGUF Structure**: Magic bytes, version, tensor/metadata counts
2. **LayerNorm Stats**: RMS in [0.5, 2.0] range (strict mode exits on failure)
3. **Tensor Presence**: Required embeddings and output tensors
4. **Offset Bounds**: All tensor offsets within file bounds

## Memory Management

- **Default**: Memory-mapped (`use_mmap: true`)
- **Fallback**: Full buffer load if mmap unavailable
- **Benefits**: Zero-copy, OS page cache, multi-process safe

## Quantization Format Detection

### I2_S BitNet32-F16 (Production)
- 32-element blocks
- Inline F16 scales
- Production performance

### I2_S QK256 (MVP)
- 256-element blocks
- Separate scales
- MVP performance (scalar kernels ~0.1 tok/s for 2B models)

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `BITNET_STRICT_MODE=1` | Fail on LayerNorm/projection warnings (exit 8) |
| `BITNET_CORRECTION_POLICY=/path/to/policy.yml` | Apply model-specific corrections |
| `BITNET_ALLOW_RUNTIME_CORRECTIONS=1` | Enable correction policy (CI blocks) |
| `BITNET_CPP_DIR=/path/to/bitnet.cpp` | Route QK256 to C++ FFI for validation |

## Error Handling Patterns

### Fail-Fast (Required)
- Missing embeddings tensor
- Malformed GGUF header
- Tensor count/offset overflow

### Graceful Degradation (Inferred)
- Missing vocab_size → infer from embedding tensor
- Missing hidden_size → infer from embedding shape
- Missing num_layers → infer from tensor name patterns
- Missing num_kv_heads → infer from k_proj shape

### Warnings (Conditional)
- Suspicious LayerNorm RMS → warn (unless `BITNET_STRICT_MODE`)
- Quantized LayerNorm weights → warn, continue with policy

## Integration Points

### CLI Inference
```
bitnet run --model model.gguf --prompt "Question?"
  ↓
ModelLoader::load() → Auto-tokenizer detection → Auto-template detection
```

### Validation
```
./scripts/validate_gguf.sh model.gguf tokenizer.json
  ↓
3-stage validation: LayerNorm, projection, linguistic sanity
```

### Cross-Validation
```
cargo run -p xtask -- crossval-per-token --model model.gguf
  ↓
Load model → Load tokenizer → Token parity check → Logits comparison
```

## Common Issues

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| "LayerNorm gamma suspicious" | Quantized LN weights | Use F16/F32 LayerNorm, or apply correction policy |
| Token mismatch in crossval | Template/BOS difference | Use `--prompt-template raw` or `--no-bos` |
| Missing metadata keys | Unusual model format | Use tensor shape inference |
| Slow QK256 inference | MVP scalar kernels | Use `--max-tokens 4-16` for validation only |
| Model shape mismatch | Embedding transposition | Auto-corrected for [hidden, vocab] → [vocab, hidden] |

## Performance Characteristics

- **Model loading**: 100ms-1s (includes mmap, metadata extraction, validation)
- **Tensor loading**: Device-dependent (CPU: 1-5s for 2B model, GPU: 0.5-2s)
- **Memory overhead**: ~15-20% above model size (metadata, caches, alignment)
- **QK256 scalar inference**: ~0.1 tok/s for 2B models (MVP)
- **QK256 AVX2 inference**: ~0.12 tok/s for 2B models (early optimization)

## Extension Points

1. **New Format Support**: Implement `FormatLoader` trait
2. **Validation Rules**: `correction_policy.rs` for model-specific fixes
3. **Tensor Remapping**: Add patterns to `weight_mapper.rs`
4. **Metadata Extraction**: Add keys to GGUF loader metadata chains

