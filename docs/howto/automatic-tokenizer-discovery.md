# How to Use Automatic Tokenizer Discovery with GGUF Models

**Problem**: You want to run neural network inference without manually specifying tokenizer files.

**Solution**: Use BitNet-rs automatic tokenizer discovery to extract tokenizers from GGUF metadata.

**Time Required**: 5 minutes

---

## Prerequisites

- BitNet-rs installed (`cargo build --no-default-features --features cpu`)
- GGUF model file (e.g., from HuggingFace Hub)

---

## Step 1: Basic Automatic Discovery

The simplest approach lets BitNet-rs handle everything:

```bash
# Download model with embedded tokenizer
cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf --file ggml-model-i2_s.gguf

# Run inference with automatic tokenizer discovery
cargo run -p xtask -- infer --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf --prompt "Explain quantization:"
```

**What Happens**:
1. BitNet-rs opens GGUF file with memory mapping
2. Extracts tokenizer from `tokenizer.json` or `tokenizer.ggml.tokens` metadata
3. Detects model architecture (BitNet, LLaMA, GPT-2, etc.)
4. Applies model-specific tokenizer configuration
5. Ready for inference

---

## Step 2: Rust API Usage

For programmatic control, use the Rust API:

```rust,no_run
use bitnet_tokenizers::{TokenizerDiscovery, TokenizerStrategyResolver};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Automatic discovery from GGUF
    let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;
    let resolver = TokenizerStrategyResolver::new(discovery).await?;
    let tokenizer = resolver.resolve_with_fallback().await?;

    // Use tokenizer
    let tokens = tokenizer.encode("Hello, world!", true, false)?;
    println!("Tokens: {:?}", tokens);

    Ok(())
}
```

---

## Step 3: Inspect Discovery Results

See what BitNet-rs discovered:

```bash
# Verify model and show tokenizer metadata
cargo run -p xtask -- verify --model model.gguf
```

```rust,no_run
use bitnet_tokenizers::TokenizerDiscovery;
use std::path::Path;

fn inspect_discovery() -> Result<(), Box<dyn std::error::Error>> {
    let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;

    println!("Model type: {}", discovery.model_type());
    println!("Vocabulary size: {}", discovery.vocab_size());
    println!("GPU optimization: {}", discovery.requires_large_vocab_optimization());

    // Check discovered strategy
    let strategy = discovery.discover_tokenizer_strategy()?;
    println!("Strategy: {}", strategy.description());

    Ok(())
}
```

---

## Step 4: Handle Fallback Scenarios

If GGUF doesn't have embedded tokenizer, BitNet-rs uses fallback chain:

```bash
# Place tokenizer.json in same directory as model.gguf
cp tokenizer.json /path/to/models/

# BitNet-rs automatically finds co-located tokenizer
cargo run -p xtask -- infer --model /path/to/models/model.gguf --prompt "Test"
```

**Fallback Order**:
1. **Embedded**: Extract from GGUF metadata
2. **Co-located**: Search model directory for `tokenizer.json`, `tokenizer.model`, etc.
3. **Cache**: Check `$BITNET_CACHE_DIR` and HuggingFace cache
4. **Download**: Fetch from HuggingFace Hub (if online)
5. **Mock**: Testing only (disabled in strict mode)

---

## Step 5: Production Deployment

Enable strict mode for production to prevent mock fallbacks:

```bash
# Strict mode: fail if no real tokenizer available
export BITNET_STRICT_TOKENIZERS=1
cargo run -p xtask -- infer --model model.gguf --prompt "Production test"
```

```rust,no_run
// Rust API: Enable strict mode
std::env::set_var("BITNET_STRICT_TOKENIZERS", "1");

let resolver = TokenizerStrategyResolver::new(discovery).await?;
let tokenizer = resolver.resolve_with_fallback().await?;
// ✅ Guaranteed real tokenizer or error
```

---

## Common Scenarios

### Scenario 1: Model with Embedded HuggingFace Tokenizer

```bash
# GGUF contains `tokenizer.json` string metadata
cargo run -p xtask -- infer --model llama3.gguf --prompt "Test"
# ✅ Extracts HuggingFace tokenizer automatically
```

### Scenario 2: Model with SentencePiece Vocabulary

```bash
# GGUF contains `tokenizer.ggml.tokens` array metadata
cargo run -p xtask -- infer --model llama2.gguf --prompt "Test"
# ✅ Creates tokenizer from embedded vocabulary
```

### Scenario 3: Model with Co-Located Tokenizer

```bash
# Directory structure:
# /models/gpt2/
#   ├── model.gguf
#   └── tokenizer.json

cargo run -p xtask -- infer --model /models/gpt2/model.gguf --prompt "Test"
# ✅ Finds co-located tokenizer.json automatically
```

### Scenario 4: Offline Deployment

```bash
# Disable downloads in air-gapped environment
export BITNET_OFFLINE=1
cargo run -p xtask -- infer --model model.gguf --prompt "Test"
# ✅ Uses only embedded, co-located, or cached tokenizers
```

---

## Troubleshooting

**Problem**: "No compatible tokenizer found (strict mode)"

**Solution**: Provide explicit tokenizer or disable strict mode:
```bash
cargo run -p xtask -- infer --model model.gguf --tokenizer tokenizer.json --prompt "Test"
# OR
unset BITNET_STRICT_TOKENIZERS
```

---

**Problem**: "Vocabulary size mismatch"

**Solution**: GGUF metadata may be incorrect. Use explicit tokenizer:
```bash
cargo run -p xtask -- infer --model model.gguf --tokenizer correct_tokenizer.json --prompt "Test"
```

---

**Problem**: "Download failed from HuggingFace Hub"

**Solution**: Enable offline mode or cache tokenizer manually:
```bash
export BITNET_OFFLINE=1
# OR
cp tokenizer.json ~/.cache/bitnet/llama/32000/
```

---

## Next Steps

- [Extract Embedded Tokenizers](extract-embedded-tokenizers.md) - Manual extraction guide
- [Troubleshoot Tokenizer Discovery](tokenizer-discovery-troubleshooting.md) - Advanced fallback configuration and error handling
- [Environment Variables](../environment-variables.md) - Offline mode and strict configuration

---

**Task Complete!** You now know how to use automatic tokenizer discovery with GGUF models. ✅
