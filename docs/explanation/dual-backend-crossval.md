# Dual-Backend Cross-Validation Architecture

BitNet-rs supports systematic validation against two different C++ reference implementations: **bitnet.cpp** (for 1-bit models) and **llama.cpp** (for general GGUF models). This document explains the architecture, design decisions, and operational considerations.

## Overview

Cross-validation is a critical quality assurance mechanism that compares BitNet-rs inference results against official C++ reference implementations to ensure numerical parity and correctness. BitNet-rs implements dual-backend support to validate different model families:

- **Lane A**: BitNet-rs vs bitnet.cpp (1-bit quantization validation)
- **Lane B**: BitNet-rs vs llama.cpp (general GGUF model validation)

### Why Dual-Backend?

BitNet-rs must support multiple model types with different C++ reference implementations:

1. **BitNet Models** (microsoft-bitnet-b1.58-2B-4T-gguf):
   - Use 1-bit quantization (I2_S)
   - Have BitNet-specific weight organization
   - Validated against bitnet.cpp (official reference)

2. **LLaMA Models** (llama-3, llama-2, SmolLM3):
   - Use standard GGUF format
   - Compatible with llama.cpp
   - Validated against llama.cpp

3. **Other GGUF Models**:
   - SafeLM, Phi, etc.
   - Generally compatible with llama.cpp
   - Default to llama.cpp unless explicitly overridden

## Architecture

### Component Structure

```
┌─────────────────────────────────────────────────────────────┐
│ CrossvalPerToken Command (xtask CLI)                        │
│                                                             │
│  Arguments:                                                 │
│    --model <path>           GGUF model path                │
│    --tokenizer <path>       Tokenizer file                 │
│    --prompt <text>          Input prompt                   │
│    --cpp-backend <bitnet|llama>  Backend selection (opt)  │
│    --prompt-template <type>      Template override (opt)   │
│    --system-prompt <text>        System prompt (opt)       │
│    --dump-ids / --dump-cpp-ids   Debug token IDs (opt)    │
│    --verbose                     Diagnostics (opt)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Backend Selection   │
            │  (Auto-Detection)    │
            │                      │
            │ Path contains:        │
            │  "bitnet" → BitNet   │
            │  "llama"  → Llama    │
            │  else     → Llama    │
            │                      │
            │ Override with:        │
            │  --cpp-backend       │
            └──────────┬───────────┘
                       │
         ┌─────────────┴──────────────┐
         │                            │
         ▼                            ▼
    ┌─────────────┐          ┌─────────────┐
    │ BitNet Lane │          │ LLaMA Lane  │
    │             │          │             │
    │ bitnet.cpp  │          │ llama.cpp   │
    │ libbitnet.so│          │ libllama.so │
    │ libggml.so  │          │ libggml.so  │
    └──────┬──────┘          └──────┬──────┘
           │                        │
           ▼                        ▼
    ┌──────────────────────────────────────┐
    │ Parity Comparison                    │
    │                                      │
    │ For each token position:              │
    │ 1. Run Rust forward pass             │
    │ 2. Run C++ forward pass              │
    │ 3. Compare logits (cosine similarity)│
    │ 4. Report first divergence if any    │
    └──────┬───────────────────────────────┘
           │
           ▼
    ┌──────────────────────┐
    │ Results Output       │
    │                      │
    │ Text format:         │
    │  Position N: OK      │
    │  Similarity: 0.9999  │
    │                      │
    │ JSON format:         │
    │  {                   │
    │    "status": "ok",   │
    │    "divergence": -1, │
    │    "metrics": {...}  │
    │  }                   │
    └──────────────────────┘
```

### Backend Auto-Detection

The `CppBackend` enum provides path-based auto-detection with three fallback levels:

```rust
impl CppBackend {
    pub fn from_model_path(path: &Path) -> Self {
        let path_str = path.to_string_lossy().to_lowercase();

        // Priority 1: Explicit "bitnet" or "microsoft/bitnet" in path
        if path_str.contains("bitnet") || path_str.contains("microsoft/bitnet") {
            Self::BitNet
        }
        // Priority 2: "llama" in path (more common)
        else if path_str.contains("llama") {
            Self::Llama
        }
        // Priority 3: Conservative default to llama (safer, more tested)
        else {
            Self::Llama
        }
    }
}
```

**Design Rationale**:
- BitNet detection is more specific (requires "bitnet" keyword)
- LLaMA is the default (most models are LLaMA-compatible)
- Conservative fallback prevents library mismatches
- CLI flag `--cpp-backend` allows explicit override

### Backend-Specific Configuration

Each backend has distinct library requirements and initialization paths:

**BitNet Backend**:
- Required libraries: `libbitnet.so`, `libggml.so`
- Tokenizer: BitNet's specialized SentencePiece integration
- Preflight checks: Verify `libbitnet` present in `$BITNET_CPP_DIR/build/bin`
- Model constraints: BitNet-specific weight layout

**LLaMA Backend**:
- Required libraries: `libllama.so`, `libggml.so`
- Tokenizer: LLaMA's universal tokenizer API
- Preflight checks: Verify both `libllama` and `libggml` present
- Model constraints: Standard GGUF format

### Tokenization Pipeline

Both backends follow the same tokenization flow:

```
Input Text (--prompt)
    │
    ├─ Apply prompt template (--prompt-template)
    │  └─ Inject system prompt if provided (--system-prompt)
    │
    ▼
Template-Formatted String
    │
    ├─ Rust tokenizer (bitnet-tokenizers)
    │  └─ Produces token IDs for Rust inference
    │
    ├─ C++ tokenizer (via FFI)
    │  └─ Bitnet: bitnet.cpp's tokenizer API
    │  └─ LLaMA: llama.cpp's tokenizer API
    │
    ▼
Token ID Sequences (Rust & C++)
    │
    ├─ --dump-ids: Show Rust token IDs (stderr)
    ├─ --dump-cpp-ids: Show C++ token IDs (stderr)
    │
    ▼
Parity Check: Rust IDs == C++ IDs?
    │
    ├─ YES: Proceed to logits comparison
    ├─ NO:  Report tokenizer divergence, exit
    │
    ▼
Forward Pass Comparison (per token)
```

**Key Design Points**:
- Template application happens before tokenization (both sides)
- Tokenizer output is compared before forward pass (catch early divergence)
- Token ID mismatches immediately halt parity checks
- Separate `--dump-ids` / `--dump-cpp-ids` flags for debugging

## Operational Flows

### Flow 1: Auto-Detection (Most Common)

User provides model path without explicit backend selection:

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4
```

**Steps**:
1. Extract model path: `models/microsoft-bitnet-b1.58-2B-4T-gguf/...`
2. Auto-detect: Path contains "bitnet" → use BitNet backend
3. Preflight: Verify `libbitnet.so` available
4. Run parity: Tokenize, compare logits, report results

### Flow 2: Explicit Backend Override

User explicitly selects backend (for testing or non-standard models):

```bash
# Force llama backend even for BitNet model
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --cpp-backend llama \
  --prompt "Test" \
  --max-tokens 2
```

**Steps**:
1. Parse CLI: `--cpp-backend llama` → explicit LLaMA
2. Skip auto-detection (uses provided value)
3. Preflight: Verify `libllama.so` available
4. Run parity: Use LLaMA's tokenizer and forward pass APIs

### Flow 3: Template Override

User specifies custom prompt template (affects tokenization):

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt-template raw \
  --prompt "2+2=" \
  --max-tokens 1
```

**Steps**:
1. Parse CLI: `--prompt-template raw` → no formatting
2. Apply template: Raw mode passes prompt as-is
3. Tokenize: Both Rust and C++ tokenize the raw prompt
4. Compare and validate

### Flow 4: Debug Mode (Token ID Inspection)

User enables diagnostic flags to investigate tokenizer mismatches:

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test" \
  --dump-ids \
  --dump-cpp-ids \
  --verbose
```

**Outputs** (to stderr with `--verbose` enabled):
```
[BACKEND] Selected: llama (from path heuristics)
[PREFLIGHT] Checking libllama.so... OK
[PREFLIGHT] Checking libggml.so... OK
[TOKENIZE] Rust IDs: [1, 4872, 338]
[TOKENIZE] C++ IDs:  [1, 4872, 338]
[TOKENIZE] Status: Parity OK
[INFERENCE] Position 0: cos_sim=0.99999
[INFERENCE] Position 1: cos_sim=0.99998
[COMPLETE] No divergence detected
```

## Configuration & Environment Variables

### Build-Time Configuration

Configuration at build time (compile-time library detection):

```bash
# Build with C++ backend support
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/bin:$LD_LIBRARY_PATH"

# Build xtask with all cross-validation features
cargo build -p xtask --features crossval-all

# Verify library detection worked
cargo run -p xtask --features crossval-all -- crossval-per-token --help
```

### Runtime Configuration

Configuration at runtime (library loading, backend selection):

```bash
# Set library path for dynamic loader
export LD_LIBRARY_PATH="$HOME/.cache/bitnet_cpp/build/bin:$LD_LIBRARY_PATH"  # Linux
export DYLD_LIBRARY_PATH="$HOME/.cache/bitnet_cpp/build/bin:$DYLD_LIBRARY_PATH"  # macOS

# Set C++ directory (if not ~/.cache/bitnet_cpp)
export BITNET_CPP_DIR="/path/to/bitnet.cpp"

# Run cross-validation
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test"
```

## Error Handling & Diagnostics

### Backend Selection Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Backend 'bitnet' requires libbitnet" | BitNet libraries not built | Run `cargo run -p xtask -- fetch-cpp --force` |
| "Backend 'llama' requires libllama" | LLaMA libraries not built | Rebuild bitnet.cpp or standalone llama.cpp |
| "cannot open shared object file" | Library path not set | Set `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH` |
| "Preflight check failed" | Libraries found but unusable | Rebuild with matching architecture/toolchain |

### Tokenizer Parity Errors

| Error | Cause | Solution |
|-------|-------|----------|
| Token ID mismatch at position N | Tokenizer divergence | Use `--dump-ids --dump-cpp-ids --verbose` to compare |
| Different token counts | Template application differs | Verify `--prompt-template` consistent between runs |
| Special token mismatch | Different special token handling | Inspect GGUF metadata with `compat-check --show-kv` |

### Logits Divergence

| Error | Cause | Solution |
|-------|-------|----------|
| Cosine similarity < threshold at position N | Model divergence detected | Investigate differences in quantization, layer norms, kernels |
| L2 distance anomaly | Potential numerical instability | Check for intermediate overflow in forward pass |
| Max abs difference spike | Sharp divergence in layer | Trace that layer with `cargo run -p xtask -- trace-diff` |

## Usage Examples

### Example 1: Validate BitNet Model (Auto-Detection)

```bash
# Setup (one-time)
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Validate BitNet model
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --cos-tol 0.999 \
  --format json
```

**Expected Output**:
```json
{
  "status": "ok",
  "backend": "bitnet",
  "divergence_token": -1,
  "metrics": {
    "min_cosine_similarity": 0.99999,
    "max_l2_distance": 0.00042,
    "mean_abs_difference": 0.00018
  }
}
```

### Example 2: Validate LLaMA Model with Template Override

```bash
# LLaMA model with explicit template
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/llama-3-8b-instruct.gguf \
  --tokenizer models/tokenizer.json \
  --prompt-template llama3-chat \
  --system-prompt "You are a helpful assistant" \
  --prompt "Explain photosynthesis in 2 sentences" \
  --max-tokens 32 \
  --cos-tol 0.995 \
  --format text
```

**Expected Output**:
```
Position 0: OK (cos_sim: 0.9998, l2_dist: 0.0031)
Position 1: OK (cos_sim: 0.9997, l2_dist: 0.0042)
...
Position 31: OK (cos_sim: 0.9995, l2_dist: 0.0084)

Summary: All positions parity OK
Minimum cosine similarity: 0.99950
Maximum L2 distance: 0.00840
```

### Example 3: Debug Tokenizer Divergence

```bash
# Inspect token IDs to debug mismatch
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Hello world" \
  --max-tokens 2 \
  --dump-ids \
  --dump-cpp-ids \
  --verbose 2>&1 | grep -E "TOKENIZE|IDs|Parity"
```

**Example Output**:
```
[TOKENIZE] Rust IDs: [1, 3435, 1526]
[TOKENIZE] C++ IDs:  [1, 3435, 1526]
[TOKENIZE] Status: Parity OK
```

### Example 4: Force Backend Selection

```bash
# Force llama backend for BitNet model (testing compatibility)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet.gguf \
  --tokenizer models/tokenizer.json \
  --cpp-backend llama \
  --prompt "Test" \
  --max-tokens 1 \
  --verbose
```

## Performance Considerations

### Compute Time

Dual-backend validation runs two inference passes per position:
1. Rust forward pass
2. C++ forward pass

For a 4-token generation with 2B model:
- Rust inference: ~0.1 tok/s (QK256 MVP)
- C++ inference: ~1-5 tok/s (bitnet.cpp/llama.cpp)
- Bottleneck: Rust QK256 MVP (scalar kernels)

**Recommendation**: Use `--max-tokens 1-4` for quick validation, not full sequences.

### Memory Usage

Both Rust and C++ models loaded in memory simultaneously:
- BitNet 2B: ~600 MB (quantized)
- Tokenizer context: ~10-50 MB
- Total: ~1-2 GB for dual-backend validation

### Library Caching

Libraries are built once and cached in `~/.cache/bitnet_cpp/`:

```
~/.cache/bitnet_cpp/
├── build/
│   └── bin/
│       ├── libbitnet.so (or .dylib)
│       ├── libllama.so
│       └── libggml.so
└── src/
    └── (source repo)
```

Subsequent runs reuse cached libraries (fast startup).

## Design Tradeoffs

### Why Not Unified Backend?

A single unified C++ backend (e.g., always llama.cpp) would simplify:
- Library management
- Dependency tracking
- Build system complexity

**But we chose dual-backend because**:
- BitNet models specifically need bitnet.cpp for authoritative validation
- Different model families have different reference implementations
- llama.cpp is not drop-in compatible with BitNet quantization specifics
- Users benefit from validating against the official reference for their model type

### Why Auto-Detection by Path?

We chose path-based auto-detection over:
- **Model signature detection**: More complex, slower
- **User prompting**: Poor UX for automation
- **Always explicit**: Too verbose for common cases

**Path detection is**:
- Fast (single string check)
- User-friendly (works by default)
- Overrideable (--cpp-backend for exceptions)
- Deterministic (same model → same backend every time)

### Why Preflight Checks?

Preflight checks at build-time and runtime catch configuration errors early:
- Build-time: Catch missing libraries before user runs commands
- Runtime: Catch environment issues (LD_LIBRARY_PATH, etc.)
- Diagnostic output: Shows exactly which library is missing

Without preflight checks, users get cryptic "symbol not found" errors at the worst time.

## Related Documentation

- [`docs/howto/cpp-setup.md`](../howto/cpp-setup.md) - Installation and configuration guide
- [`docs/howto/validate-models.md`](../howto/validate-models.md) - Model validation workflow
- [`CLAUDE.md`](../../CLAUDE.md) - Cross-validation quick reference and CLI flags
- [`docs/development/validation-framework.md`](../development/validation-framework.md) - Complete validation architecture

## Glossary

- **Parity**: Numerical agreement between Rust and C++ results
- **Cosine Similarity**: Angle-based similarity metric (1.0 = identical direction)
- **L2 Distance**: Euclidean distance between logit vectors
- **Divergence**: First position where cosine similarity falls below threshold
- **Preflight Check**: Pre-execution validation of available resources
- **Auto-Detection**: Heuristic-based backend selection from model path
- **Lane**: Validation pathway (Lane A: BitNet, Lane B: LLaMA)
