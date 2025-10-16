#### Receipt Gate Evidence (CPU)

**Receipt Validation Summary**
- ✅ `compute_path`: **real** (honest compute, no mocks)
- ✅ `backend`: **cpu** (SIMD-optimized CPU inference)
- ✅ `strict`: **enabled** (quantized-only enforcement)
- ✅ `deterministic`: **enabled** (reproducible inference)
- ✅ `kernels`: **present** (quantized kernel IDs verified)
- ✅ `tokens_per_second`: **measured** (actual throughput)
- ✅ `rust_version`: **populated** (toolchain fingerprint)

**Kernel Inventory (abbreviated)**
```
i2s_gemv           # I2S quantized GEMV
tl1_lookup         # TL1 table lookup
tl2_lookup         # TL2 table lookup
rope_apply         # RoPE position encoding
logits_projection  # Final logits projection
... (see full receipt for complete list)
```

**Verification Command**
```bash
export BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 RAYON_NUM_THREADS=1
cargo run -p xtask --no-default-features --features inference -- \
  verify-receipt --path ci/inference.json
```

**Gate Policy**

✅ **CPU Receipts**
- Require quantized kernels: `i2s_*` / `tl1_*` / `tl2_*` (prefix match)
- Tokens per second: >0 (positive throughput)
- Kernel count: 1-10000 (reasonable bounds)
- Kernel ID hygiene: max 128 chars, no empty strings

❌ **Rejected Patterns**
- `compute_path = "mock"` → **FAIL** (mocked inference)
- `kernels = []` → **FAIL** (no kernel evidence)
- Empty kernel IDs → **FAIL** (missing provenance)

✅ **GPU Receipts (optional, when GPU lane active)**
- Require GPU kernels: `gemm_*` / `wmma_*` / `cuda_*`
- Auto-enforcement: `backend = "cuda"` implies GPU kernel requirement

---

📋 See full receipt: `ci/inference.json`
📘 Validation policy: `docs/baselines/20251015-cpu.json`
