# ADR-001: Production Model for CPU Baseline

**Status**: ACCEPTED
**Date**: 2025-10-15
**Context**: Issue #465 (v0.1.0-mvp Release Polish)
**Related**: AC3 (Generate Pinned CPU Baseline Receipt)

---

## Context

Issue #465 requires pinned CPU baseline receipt with realistic performance metrics for v0.1.0-mvp release. The baseline serves as a known-good reference for:
- Reproducibility verification across builds
- Regression detection for performance changes
- Honest compute evidence with real neural network execution
- User expectations for production inference performance

Two model options exist for baseline generation:

### Option 1: Test Model (`tests/models/mini.gguf`)
- **Size**: 224 bytes (minimal test model)
- **Purpose**: Smoke testing, CI verification
- **Pros**: Fast baseline generation (<1 second), no download required, minimal disk space
- **Cons**: Not representative of production, minimal kernel coverage, trivial computation

### Option 2: Production Model (`microsoft/bitnet-b1.58-2B-4T-gguf`)
- **Size**: ~2GB (2B parameter model)
- **Purpose**: Production inference, realistic benchmarking
- **Pros**: Representative performance (10-20 tok/s CPU), comprehensive kernel coverage, honest compute evidence
- **Cons**: Requires download (~5-10 minutes), slower baseline generation (~2 minutes), ~2GB disk space

---

## Decision

**Use production model (Option 2) for CPU baseline.**

---

## Rationale

### 1. Realistic Performance Metrics
- **MVP Baseline Requirement**: v0.1.0-mvp baseline should represent real-world production use cases
- **User Expectations**: 10-20 tok/s CPU is achievable and representative of 2B model inference
- **Test Model Inadequacy**: 224-byte model provides trivial computation, not neural network inference

### 2. Comprehensive Kernel Coverage
- **Transformer Pipeline**: Production model exercises full pipeline (attention, FFN, LayerNorm)
- **Quantization Validation**: I2_S 2-bit quantization with real GEMM operations
- **Device-Aware Kernels**: TL1/TL2 lookup table dequantization, KV-cache management

**Kernel Evidence** (production model):
```json
{
  "kernels": [
    "i2s_cpu_quantized_matmul",      // GEMM operations
    "tl1_lut_dequant_forward",       // Lookup table dequantization
    "attention_kv_cache_update",     // Attention mechanism
    "layernorm_forward",             // LayerNorm forward pass
    "ffn_silu_activation"            // FFN activation
  ]
}
```

**Kernel Evidence** (test model):
```json
{
  "kernels": [
    "minimal_forward"                // Trivial computation
  ]
}
```

### 3. Honest Compute Evidence
- **Receipt Verification**: Production model proves real transformer forward pass execution
- **Mock Detection**: Test model may trigger mock fallbacks (insufficient for honest compute)
- **Cross-Validation**: Production model aligns with Microsoft BitNet C++ reference (2B standard)

### 4. Alignment with BitNet-rs Standards
- **CLAUDE.md Guidance**: Production models recommended for realistic inference
- **Documentation Examples**: README, quickstart guide use production models
- **User Workflows**: Download → inference → verification flow uses production models

### 5. One-Time Cost vs. Ongoing Value
- **Download Cost**: ~2GB one-time download (5-10 minutes), cached for future use
- **Baseline Generation**: ~2 minutes vs <1 second (negligible for one-time operation)
- **Disk Space**: ~2GB acceptable for production development environment
- **Reproducibility**: Model download failures mitigated with retry logic and caching

---

## Consequences

### Positive
- ✅ **Realistic Baseline**: 10-20 tok/s CPU represents production performance
- ✅ **Comprehensive Coverage**: Full transformer pipeline exercised
- ✅ **Honest Compute**: Receipt proves real neural network execution
- ✅ **Cross-Validation**: Aligns with C++ reference baseline
- ✅ **User Expectations**: Baseline matches production use cases

### Negative
- ⚠️ **One-Time Download**: ~2GB model download required (5-10 minutes)
- ⚠️ **Slower Baseline Generation**: ~2 minutes vs <1 second for test model
- ⚠️ **Disk Space**: ~2GB storage required for model cache
- ⚠️ **Network Dependency**: Model download may fail (mitigated with retry)

### Mitigation Strategies
1. **Download Retry Logic**: Exponential backoff for network failures
2. **Model Caching**: Reuse downloaded models across baseline regenerations
3. **Offline Fallback**: Document manual model download for air-gapped environments
4. **Baseline Validation**: Verify model size >1GB to detect test model usage

---

## Alternatives Considered

### Alternative 1: Test Model Only
**Rejected**: Does not provide realistic performance metrics or comprehensive kernel coverage. Not suitable for MVP baseline.

### Alternative 2: Multiple Baselines (Test + Production)
**Deferred**: Adds complexity without immediate value. Test model baseline can be added post-MVP if needed for CI smoke tests.

### Alternative 3: Medium-Sized Model (1B parameters)
**Rejected**: No official BitNet 1B model available. Production 2B model is standard reference.

---

## Implementation Details

### Baseline Generation Command
```bash
# Download production model (one-time)
cargo run -p xtask -- download-model \
  --id microsoft/bitnet-b1.58-2B-4T-gguf \
  --file ggml-model-i2_s.gguf

# Generate deterministic baseline
export BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 BITNET_STRICT_MODE=1
cargo run -p xtask -- benchmark \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokens 128 \
  --prompt "The capital of France is"

# Copy to pinned baseline
DATE=$(date +%Y%m%d)
cp ci/inference.json docs/baselines/${DATE}-cpu.json
```

### Validation
```bash
# Verify model size (should be ~2GB)
MODEL_SIZE=$(stat -c%s models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf)
echo "Model size: $(( MODEL_SIZE / 1024 / 1024 )) MB"

# Verify baseline receipt
cargo run -p xtask -- verify-receipt --path docs/baselines/${DATE}-cpu.json
```

---

## References

- **Issue #465**: CPU Path Followup (v0.1.0-mvp Release Polish)
- **AC3**: Generate Pinned CPU Baseline Receipt
- **Microsoft BitNet**: https://github.com/microsoft/BitNet
- **CLAUDE.md**: BitNet-rs development guidance (production model recommendations)

---

## Changelog

- **2025-10-15**: Initial decision for v0.1.0-mvp baseline
