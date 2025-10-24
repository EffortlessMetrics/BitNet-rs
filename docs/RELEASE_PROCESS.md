# BitNet.rs Release Process

## 🎯 Final Go/No-Go Checklist

Run these commands exactly in order:

```bash
# 1. Setup deterministic environment
source scripts/common.sh
setup_deterministic_env
print_platform_banner

# 2. Run final validation (comprehensive)
./scripts/final_validation.sh

# 3. If all green, prepare release
./scripts/prepare_release.sh --version X.Y.Z
```

## ✅ Ship Criteria

You're ready to ship when:

1. **Acceptance Tests**: `ALL ACCEPTANCE TESTS PASSED`
2. **Format Parity**:
   - FP32↔FP32: τ-b ≥ 0.95, |ΔNLL| ≤ 1e-2
   - Quant↔FP32: τ-b ≥ 0.60, |ΔNLL| ≤ 2e-2
3. **Performance JSONs**: Both SafeTensors and GGUF exist with provenance
4. **Documentation**: PERF_COMPARISON.md rendered from JSONs
5. **Sign-off**: All integrity checks pass

## 🚀 Release Commands

```bash
# Quick validation (10 minutes)
./scripts/final_validation.sh

# Full release preparation
./scripts/prepare_release.sh --version 0.2.0

# Review checklist
cat RELEASE_CHECKLIST_v0.2.0.md

# Tag and push
git tag -a v0.2.0 -m "Dual-format validation: audited & gated"
git push --tags
```

## 📦 Release Artifacts

Attach to GitHub release:
- `bitnet-rs-v0.2.0-artifacts.tar.gz`
- `bench/results/*.json` (performance data)
- `docs/PERF_COMPARISON.md` (generated from JSONs)
- Link to `docs/PRODUCTION_READINESS.md`

## 🔒 CI Gates

Automatic protection on every PR:
- No mock features allowed
- Format parity required for model changes
- Perf docs must match JSON sources
- Schema validation for all artifacts

## 🛠️ Day-2 Operations

### Adding new GGUF models
```bash
# 1. Convert or fetch model
# 2. Validate parity
scripts/validate_format_parity.sh
# 3. Measure performance
scripts/measure_perf_json.sh
# 4. Update docs from JSON
python3 scripts/render_perf_md.py bench/results/*-*.json > docs/PERF_COMPARISON.md
```

### Debugging failures
```bash
# Replay parity failures
python3 scripts/replay_parity.py artifacts/parity_failures.jsonl --row N

# Check tokenizer configuration
target/release/bitnet info --model model.gguf

# Validate determinism
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 ./your_test.sh
```

## 📊 Provenance Tracking

Every JSON artifact includes:
- `schema_version`: 1
- `git_commit`: Short SHA
- `git_dirty`: Clean/dirty flag
- `model_hash`: First 16 chars of SHA256
- `cli_args`: Exact command used
- `wsl2`: Platform awareness flag

## ✨ One-Command Release

For the confident:
```bash
./scripts/final_validation.sh && ./scripts/prepare_release.sh --version X.Y.Z
```

---

**You've built a measured, auditable, and self-healing dual-format validation system. Ship it! 🚀**
