# Cross-Validation Quick Reference Card

**Quick access guide for dual-backend cross-validation (BitNet.cpp + llama.cpp)**

## 🚀 Quick Start (3 Commands)

```bash
# 1. One-command setup (auto-builds both backends)
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# 2. Verify backends available
cargo run -p xtask --features crossval-all -- preflight --verbose

# 3. Run first validation
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4
```

## 📋 Essential Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `--model` | GGUF file path | `models/model.gguf` |
| `--tokenizer` | Tokenizer file | `tokenizer.json` |
| `--prompt` | Input text | `"What is 2+2?"` |
| `--cpp-backend` | Force backend | `bitnet` or `llama` |
| `--verbose` | Show diagnostics | (flag, no value) |
| `--dump-ids` | Debug Rust tokens | (flag, no value) |
| `--dump-cpp-ids` | Debug C++ tokens | (flag, no value) |

## 🔍 Auto-Detection Rules

```
Path contains "bitnet" → bitnet.cpp
Path contains "llama"  → llama.cpp
Default (no match)     → llama.cpp
```

Override: `--cpp-backend bitnet` or `--cpp-backend llama`

## 🛠️ Common Commands

### Validate BitNet Model (Auto-Detect)
```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --format json
```

### Validate LLaMA Model (Auto-Detect)
```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/llama-3-8b-instruct.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Capital of France?" \
  --max-tokens 8
```

### Debug Token Mismatch
```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test" \
  --max-tokens 2 \
  --dump-ids \
  --dump-cpp-ids \
  --verbose 2>&1 | grep -E "TOKENIZE|IDs|Parity"
```

### Force Specific Backend
```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --cpp-backend llama \
  --prompt "Test" \
  --max-tokens 4 \
  --verbose
```

### Check Backend Availability
```bash
# All backends
cargo run -p xtask --features crossval-all -- preflight

# Specific backend
cargo run -p xtask --features crossval-all -- preflight --backend bitnet --verbose
```

## 🩹 Quick Troubleshooting

| Error | Fix |
|-------|-----|
| "libbitnet.so: cannot open" | `export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/bin:$LD_LIBRARY_PATH"` |
| "Backend 'bitnet' requires libbitnet" | Run `cargo run -p xtask -- fetch-cpp --force` |
| Token IDs don't match | Use `--dump-ids --dump-cpp-ids --verbose` to compare |
| "feature 'inference' not enabled" | Add `--features crossval-all` to cargo run |
| Wrong backend auto-detected | Use `--cpp-backend bitnet` or `--cpp-backend llama` |

## 📚 Documentation Locations

| Need | Read |
|------|------|
| CLI flags reference | `CLAUDE.md` lines 597-770 |
| Setup instructions | `docs/howto/cpp-setup.md` |
| Architecture details | `docs/explanation/dual-backend-crossval.md` |
| Troubleshooting (detailed) | `docs/howto/cpp-setup.md` lines 290-466 |

## 🎯 Environment Variables

```bash
# Linux
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/bin:$LD_LIBRARY_PATH"

# macOS
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
export DYLD_LIBRARY_PATH="$BITNET_CPP_DIR/build/bin:$DYLD_LIBRARY_PATH"

# Make permanent (Linux/macOS)
echo 'export LD_LIBRARY_PATH="$HOME/.cache/bitnet_cpp/build/bin:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 🔧 Preflight Commands

```bash
# Check all backends
cargo run -p xtask --features crossval-all -- preflight

# Check BitNet backend
cargo run -p xtask --features crossval-all -- preflight --backend bitnet --verbose

# Check LLaMA backend
cargo run -p xtask --features crossval-all -- preflight --backend llama --verbose
```

## 📊 Output Formats

### Text (Default)
```
Position 0: OK (cos_sim: 0.9999, l2_dist: 0.0042)
Position 1: OK (cos_sim: 0.9997, l2_dist: 0.0051)

Summary: All positions parity OK
Minimum cosine similarity: 0.99950
Maximum L2 distance: 0.00840
```

### JSON (With --format json)
```json
{
  "status": "ok",
  "backend": "bitnet",
  "divergence_token": -1,
  "metrics": {
    "min_cosine_similarity": 0.99999,
    "max_l2_distance": 0.00042,
    "mean_abs_difference": 0.00018,
    "token_count": 4
  }
}
```

## 🏃 Quick Workflows

### Workflow 1: First-Time Setup
```bash
# 1. Auto-setup
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# 2. Verify
cargo run -p xtask --features crossval-all -- preflight

# 3. Validate
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf --tokenizer models/tokenizer.json \
  --prompt "Test" --max-tokens 4
```

### Workflow 2: Debug Divergence
```bash
# 1. Run with debug flags
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf --tokenizer models/tokenizer.json \
  --prompt "Test" --max-tokens 4 \
  --dump-ids --dump-cpp-ids --verbose

# 2. Check tokenization
# Look for "[TOKENIZE] Rust IDs:" and "[TOKENIZE] C++ IDs:" in output

# 3. If tokens match but logits diverge, check model/backend
cargo run -p xtask --features crossval-all -- preflight --backend bitnet --verbose
```

---

**For complete documentation, see**:
- `CLAUDE.md` - CLI reference
- `docs/howto/cpp-setup.md` - Setup guide
- `docs/explanation/dual-backend-crossval.md` - Architecture
