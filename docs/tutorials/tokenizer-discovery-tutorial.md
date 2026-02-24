# Universal Tokenizer Discovery Tutorial

**Learning Goal**: Understand how BitNet.rs automatically discovers and configures tokenizers for neural network inference.

**Time Required**: 15 minutes

**Prerequisites**:
- Completed [quickstart.md](../quickstart.md)
- Basic understanding of neural network tokenization
- Rust development environment with BitNet.rs installed

## What You'll Learn

By the end of this tutorial, you'll understand:
1. How BitNet.rs automatically discovers tokenizers from GGUF files
2. The 4 embedded tokenizer extraction strategies
3. The 5-step fallback chain for tokenizer resolution
4. Model-specific tokenizer configurations (LLaMA, GPT-2, BitNet)
5. Production best practices with strict mode and offline mode

---

## Step 1: Understanding Automatic Tokenizer Discovery

BitNet.rs provides **automatic tokenizer discovery** that eliminates manual configuration for neural network inference.

### The Discovery Process

When you load a GGUF model, BitNet.rs:

1. **Parses GGUF metadata** using zero-copy memory mapping
2. **Detects model architecture** from tensor patterns (BitNet, LLaMA, GPT-2, etc.)
3. **Extracts vocabulary size** using 5 resolution strategies
4. **Discovers tokenizer** via fallback chain (embedded → co-located → cache → download)
5. **Applies model-specific wrapper** with optimal configuration

### Your First Discovery Example

```rust,no_run
use bitnet_tokenizers::TokenizerDiscovery;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Create discovery engine from GGUF file
    let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;

    // Step 2: Inspect discovered metadata
    println!("Model type: {}", discovery.model_type());
    println!("Vocabulary size: {}", discovery.vocab_size());
    println!("Requires GPU optimization: {}", discovery.requires_large_vocab_optimization());

    // Step 3: Discover optimal tokenizer strategy
    let strategy = discovery.discover_tokenizer_strategy()?;
    println!("Strategy: {}", strategy.description());

    Ok(())
}
```

**Expected Output:**
```
Model type: llama
Vocabulary size: 128256
Requires GPU optimization: true
Strategy: GGUF-embedded tokenizer
```

---

## Step 2: Embedded Tokenizer Extraction

BitNet.rs supports **4 embedded tokenizer extraction strategies** from GGUF metadata.

### Strategy 1: HuggingFace JSON Extraction

GGUF files can embed complete HuggingFace tokenizers as JSON strings:

```rust,no_run
use bitnet_tokenizers::TokenizerDiscovery;
use std::path::Path;

fn try_embedded_hf_tokenizer() -> Result<(), Box<dyn std::error::Error>> {
    let discovery = TokenizerDiscovery::from_gguf(Path::new("llama3.gguf"))?;

    // Try to extract embedded HuggingFace tokenizer
    if let Some(embedded) = discovery.try_extract_embedded_tokenizer()? {
        println!("✅ Found embedded HuggingFace tokenizer");
        println!("Vocabulary: {}", embedded.vocab_size());
        println!("BOS token: {:?}", embedded.bos_token_id());
        println!("EOS token: {:?}", embedded.eos_token_id());
    } else {
        println!("❌ No embedded tokenizer, will try other strategies");
    }

    Ok(())
}
```

**Metadata Key**: `tokenizer.json` (string metadata)

### Strategy 2: SentencePiece Vocabulary Extraction

GGUF files can embed SentencePiece vocabularies as token arrays:

```rust,no_run
// Metadata Key: `tokenizer.ggml.tokens` (array metadata)
// Validates vocabulary size matches expected size (±100 tokens tolerance)
// Extracts special tokens: BOS, EOS, PAD from metadata
```

### Strategy 3: Binary SentencePiece Model

GGUF files can embed complete SentencePiece models as binary blobs:

```rust,no_run
// Metadata Key: `tokenizer.ggml.model` (binary metadata)
// Minimum size: 1024 bytes for valid SentencePiece model
// Automatic fallback to BasicTokenizer if model corrupted
```

### Strategy 4: Minimal Metadata Extraction

Minimal configuration uses just special token IDs:

```rust,no_run
// Metadata Keys:
// - `tokenizer.ggml.bos_token_id`
// - `tokenizer.ggml.eos_token_id`
// - `tokenizer.ggml.pad_token_id`
// Creates BasicTokenizer with discovered vocabulary size
```

---

## Step 3: Vocabulary Size Resolution

BitNet.rs uses **5 strategies** to determine vocabulary size when metadata is incomplete.

### Strategy Flow

```rust,no_run
use bitnet_tokenizers::TokenizerDiscovery;
use std::path::Path;

fn demonstrate_vocab_resolution() -> Result<(), Box<dyn std::error::Error>> {
    let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;

    // Vocabulary resolution strategies (in order):
    // 1. Standard GGUF key: `tokenizer.ggml.vocab_size`
    // 2. Architecture-specific keys: `llama.vocab_size`, `gpt2.vocab_size`, etc.
    // 3. Alternative metadata keys: `vocab_size`, `vocabulary_size`, etc.
    // 4. Infer from embedding tensor shape (first dimension)
    // 5. Architecture defaults: LLaMA-2 (32000), LLaMA-3 (128256), GPT-2 (50257)

    let vocab_size = discovery.vocab_size();
    println!("Resolved vocabulary size: {}", vocab_size);

    Ok(())
}
```

### Architecture Default Vocabulary Sizes

| Architecture | Default Vocab | GPU Optimization |
|--------------|--------------|------------------|
| LLaMA-2      | 32,000       | No               |
| LLaMA-3      | 128,256      | Yes (>64K)       |
| GPT-2        | 50,257       | No               |
| BERT         | 30,522       | No               |
| T5           | 32,128       | No               |
| BitNet       | 50,257       | No               |

---

## Step 4: Fallback Chain Resolution

When embedded tokenizers aren't available, BitNet.rs uses a **5-step fallback chain**.

### Complete Fallback Example

```rust,no_run
use bitnet_tokenizers::{TokenizerDiscovery, TokenizerStrategyResolver};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;
    let resolver = TokenizerStrategyResolver::new(discovery).await?;

    // Automatic fallback chain:
    // 1. GGUF embedded tokenizer (try_extract_embedded_tokenizer)
    // 2. Co-located files (check_colocated_tokenizers)
    // 3. Standard cache (check_cache_locations)
    // 4. Smart download (infer_download_source + download)
    // 5. Mock fallback (non-strict mode only)

    let tokenizer = resolver.resolve_with_fallback().await?;
    println!("✅ Tokenizer resolved successfully");
    println!("Vocabulary: {}", tokenizer.vocab_size());

    Ok(())
}
```

### Fallback Step Details

**Step 1: GGUF Embedded**
- Fastest (0ms network latency)
- Zero external dependencies
- Embedded in model file

**Step 2: Co-Located Files**
- Searches model directory for:
  - `tokenizer.json` (HuggingFace)
  - `tokenizer.model` (SentencePiece)
  - `vocab.json`, `merges.txt` (BPE)
  - `{model_name}.tokenizer.json` (model-specific)

**Step 3: Standard Cache**
- Checks cache directories:
  - `$BITNET_CACHE_DIR/{model_type}/{vocab_size}/tokenizer.json`
  - `$BITNET_CACHE_DIR/{model_type}/tokenizer.json`
  - `~/.cache/huggingface/{repo}/tokenizer.json`

**Step 4: Smart Download**
- Downloads from HuggingFace Hub
- Automatic caching for subsequent runs
- Retry logic (3 attempts with exponential backoff)
- Offline mode: `BITNET_OFFLINE=1` skips download

**Step 5: Mock Fallback**
- Testing only (disabled in strict mode)
- Production: `BITNET_STRICT_TOKENIZERS=1` prevents mock fallback

---

## Step 5: Model-Specific Tokenizer Configurations

BitNet.rs applies **model-specific wrappers** for optimal tokenization.

### LLaMA Tokenizer Configuration

```rust,no_run
use bitnet_tokenizers::{LlamaTokenizerWrapper, LlamaVariant};
use std::sync::Arc;

fn configure_llama_tokenizer() -> Result<(), Box<dyn std::error::Error>> {
    // Variant detection based on vocabulary size:
    // - 32,000 → LLaMA-2
    // - 128,256 → LLaMA-3
    // - 32,016 → CodeLlama

    let base_tokenizer = load_base_tokenizer()?;
    let wrapper = LlamaTokenizerWrapper::new(base_tokenizer, 128256)?;

    // LLaMA-3 configuration:
    // - BOS token: 128000
    // - EOS token: 128001
    // - Special token range: 128000-128002
    // - GPU acceleration required for 128K vocabulary

    let tokens = wrapper.encode("Hello, LLaMA-3!", true, false)?;
    assert_eq!(tokens[0], 128000); // LLaMA-3 BOS token

    Ok(())
}

// Helper function (placeholder)
fn load_base_tokenizer() -> Result<Arc<dyn bitnet_tokenizers::Tokenizer>, Box<dyn std::error::Error>> {
    Ok(Arc::new(bitnet_tokenizers::BasicTokenizer::new()))
}
```

### GPT-2 Tokenizer Configuration

```rust,no_run
use bitnet_tokenizers::Gpt2TokenizerWrapper;

fn configure_gpt2_tokenizer() -> Result<(), Box<dyn std::error::Error>> {
    let base_tokenizer = load_base_tokenizer()?;
    let wrapper = Gpt2TokenizerWrapper::new(base_tokenizer)?;

    // GPT-2 configuration:
    // - No BOS token (add_bos=true is ignored with warning)
    // - EOS token: 50256
    // - Vocabulary: 50,257 tokens
    // - CPU-optimized (no GPU acceleration needed)

    let tokens = wrapper.encode("Hello, GPT-2!", true, false)?;
    // Note: add_bos=true is ignored for GPT-2

    assert_eq!(wrapper.bos_token_id(), None); // GPT-2 has no BOS
    assert_eq!(wrapper.eos_token_id(), Some(50256));

    Ok(())
}
```

### BitNet Quantization-Aware Configuration

```rust,no_run
use bitnet_tokenizers::BitNetTokenizerWrapper;
use bitnet_common::QuantizationType;

fn configure_bitnet_tokenizer() -> Result<(), Box<dyn std::error::Error>> {
    let base_tokenizer = load_base_tokenizer()?;
    let wrapper = BitNetTokenizerWrapper::new(base_tokenizer, QuantizationType::I2S)?;

    // BitNet configuration:
    // - Quantization-aware token validation
    // - I2S: Optimal for vocab ≤ 200K
    // - TL1/TL2: Optimal for vocab ≤ 65K
    // - Validates token IDs fit quantization format

    let tokens = wrapper.encode("BitNet inference", true, false)?;
    // Token IDs automatically validated for I2S compatibility

    println!("Quantization type: {:?}", wrapper.quantization_type());

    Ok(())
}
```

---

## Step 6: Production Best Practices

### Strict Mode (Production Recommended)

```bash
# Enable strict mode to prevent mock fallbacks
export BITNET_STRICT_TOKENIZERS=1

# Run inference - will fail if no real tokenizer available
cargo run -p xtask -- infer --model model.gguf --prompt "Test"
```

```rust,no_run
// Rust API: Enable strict mode programmatically
std::env::set_var("BITNET_STRICT_TOKENIZERS", "1");

// Now mock fallbacks are disabled
let resolver = TokenizerStrategyResolver::new(discovery).await?;
let tokenizer = resolver.resolve_with_fallback().await?;
// ✅ Guaranteed real tokenizer or error
```

### Offline Mode (Air-Gapped Deployments)

```bash
# Disable smart downloads for offline environments
export BITNET_OFFLINE=1

# Only use embedded tokenizers, co-located files, and cache
cargo run -p xtask -- infer --model model.gguf --prompt "Test"
```

### Deterministic Inference

```bash
# Reproducible tokenization and generation
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42

cargo run -p xtask -- infer --model model.gguf --prompt "Test" --deterministic
```

---

## Step 7: Complete Integration Example

Here's a complete example integrating tokenizer discovery with neural network inference:

```rust,no_run
use bitnet_tokenizers::{TokenizerDiscovery, TokenizerStrategyResolver};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Enable production mode
    std::env::set_var("BITNET_STRICT_TOKENIZERS", "1");

    // Step 2: Discover tokenizer from GGUF
    let model_path = Path::new("llama2-7b-chat.gguf");
    let discovery = TokenizerDiscovery::from_gguf(model_path)?;

    println!("📊 Model Analysis:");
    println!("  Type: {}", discovery.model_type());
    println!("  Vocabulary: {}", discovery.vocab_size());
    println!("  GPU Optimization: {}", discovery.requires_large_vocab_optimization());

    // Step 3: Resolve tokenizer with fallback
    let resolver = TokenizerStrategyResolver::new(discovery).await?;
    let tokenizer = resolver.resolve_with_fallback().await?;

    println!("\n✅ Tokenizer Ready:");
    println!("  Vocabulary: {}", tokenizer.vocab_size());
    println!("  BOS token: {:?}", tokenizer.bos_token_id());
    println!("  EOS token: {:?}", tokenizer.eos_token_id());

    // Step 4: Use tokenizer for inference
    let prompt = "Explain neural network quantization:";
    let tokens = tokenizer.encode(prompt, true, false)?;

    println!("\n🔢 Tokenization:");
    println!("  Input: {}", prompt);
    println!("  Tokens: {:?}", tokens);
    println!("  Token count: {}", tokens.len());

    // Step 5: Decode tokens back to text
    let decoded = tokenizer.decode(&tokens)?;
    println!("\n📝 Round-trip Validation:");
    println!("  Decoded: {}", decoded);

    Ok(())
}
```

**Expected Output:**
```
📊 Model Analysis:
  Type: llama
  Vocabulary: 32000
  GPU Optimization: false

✅ Tokenizer Ready:
  Vocabulary: 32000
  BOS token: Some(1)
  EOS token: Some(2)

🔢 Tokenization:
  Input: Explain neural network quantization:
  Tokens: [1, 10567, 368, 23547, 3564, 13949, 2133, 29901]
  Token count: 8

📝 Round-trip Validation:
  Decoded: Explain neural network quantization:
```

---

## What You've Learned

✅ **Automatic Discovery**: How BitNet.rs automatically discovers tokenizers from GGUF metadata

✅ **Extraction Strategies**: 4 embedded tokenizer extraction methods (HF JSON, SPM vocab, SPM binary, minimal)

✅ **Fallback Chain**: 5-step resolution (embedded → co-located → cache → download → mock)

✅ **Model-Specific**: LLaMA, GPT-2, and BitNet tokenizer configurations

✅ **Production Ready**: Strict mode, offline mode, and deterministic inference

---

## Next Steps

- **API Reference**: Complete API documentation in [tokenizer-discovery-api.md](../reference/tokenizer-discovery-api.md)
- **How-To Guides**: Task-oriented guides in [docs/howto/](../howto/)
- **Architecture**: System design in [issue-336-universal-tokenizer-discovery-spec.md](../explanation/issue-336-universal-tokenizer-discovery-spec.md)
- **Testing**: Cross-validation with [xtask crossval](../development/xtask.md)

---

## Troubleshooting

**Q: "No compatible tokenizer found for llama model (strict mode)"**

A: Ensure GGUF file contains embedded tokenizer metadata or provide explicit tokenizer path:
```bash
cargo run -p xtask -- infer --model model.gguf --tokenizer tokenizer.json --prompt "Test"
```

**Q: "Download failed from HuggingFace Hub"**

A: Enable offline mode or provide cached tokenizer:
```bash
export BITNET_OFFLINE=1
cargo run -p xtask -- infer --model model.gguf --prompt "Test"
```

**Q: "Vocabulary size mismatch"**

A: Check GGUF metadata consistency or override with explicit tokenizer for edge cases.

---

**Tutorial Complete!** You now understand BitNet.rs universal tokenizer discovery system. 🎉
