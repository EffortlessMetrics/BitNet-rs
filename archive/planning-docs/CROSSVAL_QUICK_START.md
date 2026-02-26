# BitNet-rs Cross-Validation Quick Start

**Jump to**: [Setup](#setup) | [Commands](#essential-commands) | [Debugging](#debugging) | [CI](#ci-integration) | [Troubleshooting](#troubleshooting)

---

## Setup

### One-Command Setup (Recommended)

```bash
# Bash/Zsh (Linux, macOS, WSL)
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Fish shell
cargo run -p xtask -- setup-cpp-auto --emit=fish | source

# PowerShell (Windows)
cargo run -p xtask -- setup-cpp-auto --emit=pwsh | Invoke-Expression
```

### Verify Installation

```bash
# Check both backends
cargo run -p xtask --features crossval-all -- preflight

# Check specific backend with diagnostics
cargo run -p xtask --features crossval-all -- preflight --backend bitnet --verbose
cargo run -p xtask --features crossval-all -- preflight --backend llama --verbose
```

---

## Essential Commands

### Basic Cross-Validation

```bash
# BitNet model (auto-detects bitnet.cpp backend)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4

# LLaMA model (auto-detects llama.cpp backend)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/llama-3-8b-instruct.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is the capital of France?" \
  --max-tokens 8
```

### Explicit Backend Selection

```bash
# Force llama.cpp backend
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --cpp-backend llama \
  --prompt "test" \
  --max-tokens 4

# Force bitnet.cpp backend
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --cpp-backend bitnet \
  --prompt "test" \
  --max-tokens 4
```

---

## Debugging

### Token Sequence Inspection

```bash
# See both Rust and C++ token sequences
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "test prompt" \
  --max-tokens 1 \
  --dump-ids \
  --dump-cpp-ids \
  --verbose
```

**Example Output**:
```
🦀 Rust tokens (5 total):
  [128000, 3923, 374, 220, 17]

🔧 C++ tokens (5 total, backend: bitnet):
  [128000, 3923, 374, 220, 17]
```

### JSON Output with Debug Logs

```bash
# Separate JSON output from debug logs
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "test" \
  --max-tokens 4 \
  --format json \
  --dump-ids \
  --dump-cpp-ids > output.json 2>debug.log
```

### Verbose Preflight Diagnostics

```bash
# See complete library detection info
cargo run -p xtask --features crossval-all -- preflight --backend bitnet --verbose
```

**Shows**:
- Environment variables (BITNET_CPP_DIR, LD_LIBRARY_PATH, etc.)
- Library search paths
- Libraries found in each path
- Build configuration status
- Recovery steps if failed

---

## CI Integration

### Manual Workflow Trigger (GitHub Actions)

```bash
# Via GitHub CLI
gh workflow run crossval.yml

# Or via GitHub web UI:
# Actions → Cross-Validation → Run workflow
```

### PR-Based Testing

```bash
# Add label to PR to trigger cross-validation
gh pr edit <PR-NUMBER> --add-label run-crossval

# Check status
gh run list --workflow=crossval.yml
```

### Download Artifacts

```bash
# Download parity receipts and logs
gh run download <RUN-ID>
```

---

## Troubleshooting

### Library Not Found

**Symptom**: `error: Backend 'bitnet.cpp' selected but required libraries not found`

**Solution**:
```bash
# Re-run setup
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Verify with verbose diagnostics
cargo run -p xtask --features crossval-all -- preflight --backend bitnet --verbose

# Rebuild xtask
cargo clean -p xtask && cargo build -p xtask --features crossval-all
```

### Token Mismatch (Exit Code 2)

**Symptom**: `Token parity failure: Rust tokens (5) != C++ tokens (6)`

**Solution**:
```bash
# 1. Inspect token sequences
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "your prompt" \
  --dump-ids \
  --dump-cpp-ids \
  --verbose

# 2. Try different prompt template
--prompt-template raw        # No formatting
--prompt-template instruct   # Q&A format
--prompt-template llama3-chat # Chat format

# 3. Check tokenizer file matches model
```

### Wrong Backend Selected

**Symptom**: Using llama.cpp when you expected bitnet.cpp (or vice versa)

**Solution**:
```bash
# Check what backend was selected
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "test" \
  --max-tokens 1 \
  --verbose | grep -i backend

# Explicitly override backend
--cpp-backend bitnet   # Force BitNet.cpp
--cpp-backend llama    # Force llama.cpp
```

### Preflight Passes but Inference Fails

**Symptom**: `preflight` shows AVAILABLE but `crossval-per-token` fails with library errors

**Solution**:
```bash
# 1. Check dynamic loader can find libraries
ldd $(which cargo) # Linux
otool -L $(which cargo) # macOS

# 2. Verify environment variables
echo $LD_LIBRARY_PATH    # Linux
echo $DYLD_LIBRARY_PATH  # macOS
echo $PATH               # Windows

# 3. Set library path explicitly
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src:$LD_LIBRARY_PATH"

# 4. Rebuild with verbose output
CARGO_LOG=cargo::core::compiler::fingerprint=info \
  cargo build -p xtask --features crossval-all
```

---

## Flag Reference

### crossval-per-token Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | path | (required) | Path to GGUF model file |
| `--tokenizer` | path | (required) | Path to tokenizer.json |
| `--prompt` | string | (required) | Input prompt |
| `--max-tokens` | int | 4 | Max tokens to generate |
| `--cos-tol` | float | 0.999 | Cosine similarity threshold (0.0-1.0) |
| `--format` | enum | text | Output format: `text` or `json` |
| `--prompt-template` | enum | auto | Template: `raw`, `instruct`, `llama3-chat`, `auto` |
| `--system-prompt` | string | - | System prompt for chat templates |
| `--cpp-backend` | enum | auto | Backend: `bitnet`, `llama` (auto-detects if omitted) |
| `--verbose` | flag | false | Show diagnostics |
| `--dump-ids` | flag | false | Show Rust token IDs |
| `--dump-cpp-ids` | flag | false | Show C++ token IDs |

### preflight Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--backend` | enum | (both) | Backend to check: `bitnet`, `llama` |
| `--verbose` | flag | false | Show detailed diagnostics |

---

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `BITNET_CPP_DIR` | C++ reference root directory | `/home/user/.cache/bitnet_cpp` |
| `LD_LIBRARY_PATH` | Linux shared library search path | `$BITNET_CPP_DIR/build:$LD_LIBRARY_PATH` |
| `DYLD_LIBRARY_PATH` | macOS shared library search path | `$BITNET_CPP_DIR/build:$DYLD_LIBRARY_PATH` |
| `BITNET_CROSSVAL_LIBDIR` | Override library search directory | `/custom/lib/path` |
| `CROSSVAL_HAS_BITNET` | Build-time bitnet.cpp availability | `true` (set by build.rs) |
| `CROSSVAL_HAS_LLAMA` | Build-time llama.cpp availability | `true` (set by build.rs) |

---

## Exit Codes

| Code | Meaning | Common Cause |
|------|---------|--------------|
| 0 | Success | All positions parity OK |
| 1 | General error | Invalid arguments, file not found |
| 2 | Token parity failure | Tokenization mismatch between Rust and C++ |
| 3 | Logits divergence | Inference divergence detected (cosine_sim < threshold) |

---

## Backend Auto-Detection Rules

The system automatically selects the C++ backend based on model path:

1. **Priority 1**: Path contains "bitnet" or "microsoft/bitnet" → **bitnet.cpp**
2. **Priority 2**: Path contains "llama" → **llama.cpp**
3. **Priority 3**: Default fallback → **llama.cpp** (conservative)
4. **Override**: Use `--cpp-backend bitnet|llama` to explicitly select

**Examples**:
```
models/microsoft-bitnet-b1.58-2B-4T-gguf/model.gguf  →  bitnet.cpp
models/llama-3-8b-instruct.gguf                      →  llama.cpp
models/custom-model.gguf                             →  llama.cpp (default)
models/custom-model.gguf --cpp-backend bitnet        →  bitnet.cpp (override)
```

---

## 6-Step Testing Ladder

Follow this progression for comprehensive parity validation:

1. **Smoke Test** (1 token, greedy)
   - Quick sanity check
   - Command: `--max-tokens 1 --temperature 0.0 --greedy`

2. **Short Sequence** (4 tokens)
   - Basic parity validation
   - Command: `--max-tokens 4 --cos-tol 0.999`

3. **Medium Sequence** (16 tokens)
   - Sampling stability check
   - Command: `--max-tokens 16 --cos-tol 0.995`

4. **Long Sequence** (64+ tokens)
   - Cumulative drift detection
   - Command: `--max-tokens 64 --cos-tol 0.990`

5. **Multi-Prompt Suite**
   - Template robustness testing
   - Test with: raw, instruct, llama3-chat templates

6. **Production Sweep**
   - Full model validation
   - Use: `./scripts/run_crossval_sweep.sh`

**See**: `docs/howto/parity-playbook.md` for complete guide

---

## Common Scenarios

### Scenario 1: First-Time Setup

```bash
# 1. Setup
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# 2. Verify
cargo run -p xtask --features crossval-all -- preflight --verbose

# 3. Test
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4
```

### Scenario 2: Debug Token Mismatch

```bash
# When you get exit code 2 (token parity failure)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "your failing prompt" \
  --dump-ids \
  --dump-cpp-ids \
  --verbose 2>&1 | grep -E "tokens|Parity"
```

### Scenario 3: Production Validation

```bash
# Comprehensive multi-scenario testing
./scripts/run_crossval_sweep.sh \
  models/model.gguf \
  models/tokenizer.json \
  /tmp/crossval-results

# Review results
cat /tmp/crossval-results/summary.md
```

### Scenario 4: CI/CD Integration

```bash
# Local simulation of CI workflow
# 1. Check no-FFI compilation
cargo build -p crossval --no-default-features

# 2. Check STUB mode
cargo build -p crossval --no-default-features --features crossval

# 3. Run cross-validation (if backends available)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "CI test" \
  --max-tokens 4 \
  --format json > /tmp/receipt.json
```

---

## Documentation Links

| Topic | Document | Purpose |
|-------|----------|---------|
| **Setup** | `docs/howto/cpp-setup.md` | Environment configuration |
| **Architecture** | `docs/explanation/dual-backend-crossval.md` | System design |
| **Parity Testing** | `docs/howto/parity-playbook.md` | Step-by-step workflows |
| **FFI Wiring** | `docs/specs/bitnet-available-wiring.md` | Build system integration |
| **CI/CD** | `docs/ci/SETUP.md` | GitHub Actions integration |
| **Quick Reference** | `docs/ci/crossval-quick-reference.md` | Command cheat sheet |
| **CLI Reference** | `CLAUDE.md` (lines 597-816) | Complete flag documentation |

---

## Getting Help

1. **Check diagnostics**: Run with `--verbose` flag
2. **Inspect tokens**: Use `--dump-ids --dump-cpp-ids`
3. **Verify setup**: Run `preflight --verbose`
4. **Review logs**: Check stderr output (`2>debug.log`)
5. **Read docs**: See links above
6. **Check issues**: Search GitHub issues for similar problems

---

**Last Updated**: October 25, 2025
**BitNet-rs Version**: v0.1.0-qna-mvp (with dual-backend cross-validation)
