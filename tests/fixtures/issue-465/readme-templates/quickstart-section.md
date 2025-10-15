## 10-Line CPU Quickstart

Get started with BitNet.rs CPU inference in under 10 lines:

```bash
# 1. Build with explicit CPU features
cargo build --no-default-features --features cpu

# 2. Download a BitNet model
cargo run -p xtask -- download-model

# 3. Run deterministic inference (128 tokens)
export BITNET_DETERMINISTIC=1 RAYON_NUM_THREADS=1 BITNET_SEED=42
cargo run -p xtask -- benchmark --model models/*.gguf --tokens 128

# 4. Verify honest compute receipt
cargo run -p xtask -- verify-receipt ci/inference.json
```

**Expected Performance:** 10-20 tok/s on CPU for 2B I2_S models (see [baselines/](docs/baselines/) for measured results).

**Receipt Verification:** All inference runs generate receipts (`ci/inference.json`) with kernel IDs proving real computation. CI blocks PRs with mocked receipts.
