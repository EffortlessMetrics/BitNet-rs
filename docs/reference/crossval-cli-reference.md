# Cross-Validation CLI Reference

Reference for bitnet-rs cross-validation commands comparing Rust inference against C++ reference implementations.

## Prerequisites

- C++ reference built: `cargo run --locked -p xtask -- fetch-cpp` or `setup-cpp-auto`
- Environment: `BITNET_CPP_DIR` and dynamic loader path set
- Build flag: `--features crossval-all` (or `--features inference`)

## crossval-per-token

Per-token parity comparison between Rust and C++ inference. Finds the first logits divergence position.

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | path | required | Path to GGUF model file |
| `--tokenizer` | path | required | Path to tokenizer.json |
| `--prompt` | string | required | Input prompt |
| `--max-tokens` | integer | 4 | Max tokens to generate (excluding prompt) |
| `--cos-tol` | float | 0.999 | Cosine similarity threshold; below = divergence |
| `--format` | string | text | Output format: `text` or `json` |
| `--prompt-template` | enum | auto | Template: raw, instruct, llama3-chat, auto |
| `--system-prompt` | string | — | System prompt for chat templates |
| `--cpp-backend` | enum | auto | C++ backend: bitnet, llama |
| `--verbose` | flag | false | Show backend selection, preflight, diagnostics |
| `--dump-ids` | flag | false | Dump Rust token IDs to stderr |
| `--dump-cpp-ids` | flag | false | Dump C++ token IDs to stderr |

### Backend Auto-Detection

| Path pattern | Backend |
|-------------|---------|
| Contains "bitnet" or "microsoft/bitnet" | bitnet.cpp |
| Contains "llama" | llama.cpp |
| Default | llama.cpp |

Override with `--cpp-backend bitnet` or `--cpp-backend llama`.

### Output Formats

**Text (default):**

```
Position 0: OK (cos_sim: 0.9999, l2_dist: 0.0042)
Position 1: OK (cos_sim: 0.9997, l2_dist: 0.0051)
Position 2: OK (cos_sim: 0.9995, l2_dist: 0.0084)

Summary: All positions parity OK
Minimum cosine similarity: 0.99950
Maximum L2 distance: 0.00840
```

**JSON:**

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

### Examples

```bash
# BitNet model (auto-detects bitnet.cpp)
cargo run --locked -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" --max-tokens 4 --cos-tol 0.999

# Full diagnostics
cargo run --locked -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf --tokenizer models/tokenizer.json \
  --cpp-backend bitnet --prompt-template raw \
  --prompt "2+2=" --max-tokens 1 \
  --dump-ids --dump-cpp-ids --verbose

# JSON output
cargo run --locked -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" --max-tokens 4 --format json

# With system prompt (chat template)
cargo run --locked -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf --tokenizer models/tokenizer.json \
  --prompt-template llama3-chat \
  --system-prompt "You are a helpful assistant" \
  --prompt "Explain photosynthesis" --max-tokens 32
```

## setup-cpp-auto

One-command C++ reference setup (fetch, build, emit environment exports).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--emit` | string | sh | Shell format: sh, fish, pwsh |

```bash
# Bash/Zsh
eval "$(cargo run --locked -p xtask -- setup-cpp-auto --emit=sh)"

# Fish
cargo run --locked -p xtask -- setup-cpp-auto --emit=fish | source

# PowerShell
cargo run --locked -p xtask -- setup-cpp-auto --emit=pwsh | Invoke-Expression
```

**What it does:**

1. Detects if bitnet.cpp is already built
2. Downloads and builds C++ reference if needed
3. Emits shell-specific environment variable exports (`BITNET_CPP_DIR`, loader paths)

## preflight

Check C++ backend availability for cross-validation.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--backend` | enum | — | Check specific backend: bitnet, llama. Omit to check both. |
| `--verbose` | flag | false | Detailed diagnostics |

```bash
# Check all backends
cargo run --locked -p xtask --features crossval-all -- preflight

# Check specific backend
cargo run --locked -p xtask --features crossval-all -- preflight --backend bitnet --verbose
```

**Example output:**

```
Backend Library Status:

  + bitnet.cpp: AVAILABLE
    Libraries: libbitnet*

  + llama.cpp: AVAILABLE
    Libraries: libllama*, libggml*

Both backends available. Dual-backend cross-validation supported.
```

## Related Commands

```bash
# Cross-validation sweep (multi-scenario comparison)
./scripts/run_crossval_sweep.sh model.gguf tokenizer.json /tmp/crossval

# Trace comparison (debug divergence)
cargo run --locked -p xtask -- trace-diff /tmp/rs_traces /tmp/cpp_traces
```

See also: [C++ setup guide](../howto/cpp-setup.md) | [Dual-backend architecture](../explanation/dual-backend-crossval.md)
