# How to Extract Embedded Tokenizers from GGUF Files

**Problem**: You want to manually extract and inspect tokenizers embedded in GGUF model files.

**Solution**: Use BitNet.rs tokenizer discovery API to extract HuggingFace JSON, SentencePiece vocabularies, or binary models.

**Time Required**: 10 minutes

---

## Prerequisites

- BitNet.rs installed (`cargo build --no-default-features --features cpu`)
- GGUF model file with embedded tokenizer metadata

---

## Step 1: Check for Embedded Tokenizer

First, verify if GGUF contains embedded tokenizer:

```rust,no_run
use bitnet_tokenizers::TokenizerDiscovery;
use std::path::Path;

fn check_embedded() -> Result<(), Box<dyn std::error::Error>> {
    let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;

    if let Some(embedded) = discovery.try_extract_embedded_tokenizer()? {
        println!("✅ Found embedded tokenizer");
        println!("Vocabulary: {}", embedded.vocab_size());
        println!("BOS token: {:?}", embedded.bos_token_id());
        println!("EOS token: {:?}", embedded.eos_token_id());
    } else {
        println!("❌ No embedded tokenizer found");
    }

    Ok(())
}
```

---

## Step 2: Extract HuggingFace JSON Tokenizer

GGUF files can embed complete HuggingFace tokenizers as JSON:

```rust,no_run
use bitnet_tokenizers::TokenizerDiscovery;
use std::path::Path;

fn extract_hf_json() -> Result<(), Box<dyn std::error::Error>> {
    let discovery = TokenizerDiscovery::from_gguf(Path::new("llama3.gguf"))?;

    // Try extraction
    match discovery.try_extract_embedded_tokenizer()? {
        Some(tokenizer) => {
            println!("✅ Extracted HuggingFace tokenizer");
            println!("Type: {}", discovery.model_type());
            println!("Vocabulary: {}", tokenizer.vocab_size());

            // Test tokenization
            let tokens = tokenizer.encode("Hello, LLaMA-3!", true, false)?;
            println!("Test tokens: {:?}", tokens);
        }
        None => {
            println!("No HuggingFace JSON found in GGUF metadata");
            println!("GGUF metadata key checked: `tokenizer.json`");
        }
    }

    Ok(())
}
```

**GGUF Metadata Key**: `tokenizer.json` (string metadata)

**Validation**:
- JSON must start with `{` and be at least 50 characters
- Parsed and validated as HuggingFace tokenizer format
- Special tokens (BOS, EOS, PAD) extracted from metadata

---

## Step 3: Extract SentencePiece Vocabulary

GGUF files can embed SentencePiece vocabularies as token arrays:

```rust,no_run
use bitnet_tokenizers::TokenizerDiscovery;
use std::path::Path;

fn extract_spm_vocab() -> Result<(), Box<dyn std::error::Error>> {
    let discovery = TokenizerDiscovery::from_gguf(Path::new("llama2.gguf"))?;

    // Extract SentencePiece vocabulary
    match discovery.try_extract_embedded_tokenizer()? {
        Some(tokenizer) => {
            println!("✅ Extracted SentencePiece vocabulary");
            println!("Vocabulary size: {}", tokenizer.vocab_size());

            // SentencePiece special tokens
            println!("BOS token: {:?}", tokenizer.bos_token_id());
            println!("EOS token: {:?}", tokenizer.eos_token_id());

            // Test encoding
            let tokens = tokenizer.encode("LLaMA-2 tokenization", true, false)?;
            println!("Tokens: {:?}", tokens);
        }
        None => {
            println!("No SentencePiece vocabulary found");
            println!("GGUF metadata key checked: `tokenizer.ggml.tokens`");
        }
    }

    Ok(())
}
```

**GGUF Metadata Keys**:
- `tokenizer.ggml.tokens` (string array metadata)
- `tokenizer.ggml.bos_token_id` (u32 metadata)
- `tokenizer.ggml.eos_token_id` (u32 metadata)
- `tokenizer.ggml.pad_token_id` (u32 metadata)

**Validation**:
- Vocabulary size must match expected size (±100 tokens tolerance)
- All special token IDs must be within vocabulary bounds
- Minimum vocabulary size: 1000 tokens

---

## Step 4: Extract Binary SentencePiece Model

GGUF files can embed complete SentencePiece models as binary blobs:

```rust,no_run
use bitnet_tokenizers::TokenizerDiscovery;
use std::path::Path;

fn extract_spm_binary() -> Result<(), Box<dyn std::error::Error>> {
    let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;

    // Extract binary SentencePiece model
    match discovery.try_extract_embedded_tokenizer()? {
        Some(tokenizer) => {
            println!("✅ Extracted binary SentencePiece model");
            println!("Model type: {}", discovery.model_type());
            println!("Vocabulary: {}", tokenizer.vocab_size());

            // Binary model validation passed
            println!("Binary model size: >=1024 bytes");
        }
        None => {
            println!("No binary SentencePiece model found");
            println!("GGUF metadata key checked: `tokenizer.ggml.model`");
        }
    }

    Ok(())
}
```

**GGUF Metadata Key**: `tokenizer.ggml.model` (binary metadata)

**Validation**:
- Minimum size: 1024 bytes (valid SentencePiece model)
- Automatic fallback to BasicTokenizer if corrupted
- Special tokens extracted from separate metadata keys

---

## Step 5: Extract Minimal Tokenizer Metadata

For minimal GGUF files with only special token IDs:

```rust,no_run
use bitnet_tokenizers::TokenizerDiscovery;
use std::path::Path;

fn extract_minimal() -> Result<(), Box<dyn std::error::Error>> {
    let discovery = TokenizerDiscovery::from_gguf(Path::new("minimal.gguf"))?;

    // Extract minimal tokenizer from special token IDs only
    match discovery.try_extract_embedded_tokenizer()? {
        Some(tokenizer) => {
            println!("✅ Created minimal tokenizer from metadata");
            println!("Vocabulary: {}", tokenizer.vocab_size());
            println!("BOS token: {:?}", tokenizer.bos_token_id());
            println!("EOS token: {:?}", tokenizer.eos_token_id());

            // Uses vocabulary size from metadata or tensor inference
            println!("Fallback: BasicTokenizer with discovered vocab size");
        }
        None => {
            println!("No tokenizer metadata found");
            println!("Required: At least BOS or EOS token ID in GGUF metadata");
        }
    }

    Ok(())
}
```

**GGUF Metadata Requirements** (at least one):
- `tokenizer.ggml.bos_token_id`
- `tokenizer.ggml.eos_token_id`
- `tokenizer.ggml.pad_token_id`

**Fallback Behavior**:
- Creates `BasicTokenizer` with discovered vocabulary size
- Uses architecture defaults if vocabulary size unavailable
- Minimal but functional tokenizer for basic inference

---

## Step 6: Extraction Strategy Priority

BitNet.rs tries extraction strategies in this order:

```text
1. HuggingFace JSON (`tokenizer.json` string metadata)
   ✅ Most complete, includes all tokenizer configuration
   ✅ Ready for production use
   ✅ Fastest extraction (<50ms)

2. SentencePiece Vocabulary (`tokenizer.ggml.tokens` array)
   ✅ Complete vocabulary with special tokens
   ✅ Good for LLaMA-2/3 models
   ✅ Requires vocabulary size validation

3. Binary SentencePiece Model (`tokenizer.ggml.model` binary)
   ✅ Complete SentencePiece model
   ✅ Requires minimum 1024 bytes
   ✅ Fallback to BasicTokenizer if corrupted

4. Minimal Metadata (special token IDs only)
   ⚠️  Basic functionality only
   ⚠️  May lack proper tokenization rules
   ⚠️  Use for testing or simple models only
```

---

## Complete Extraction Example

```rust,no_run
use bitnet_tokenizers::TokenizerDiscovery;
use std::path::Path;

fn complete_extraction_demo() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new("llama3-8b.gguf");
    let discovery = TokenizerDiscovery::from_gguf(model_path)?;

    println!("📊 Model Analysis:");
    println!("  Type: {}", discovery.model_type());
    println!("  Vocabulary: {}", discovery.vocab_size());
    println!("  GPU Optimization: {}", discovery.requires_large_vocab_optimization());

    println!("\n🔍 Attempting Tokenizer Extraction:");

    match discovery.try_extract_embedded_tokenizer()? {
        Some(tokenizer) => {
            println!("✅ Extraction Successful!");
            println!("\n📝 Tokenizer Details:");
            println!("  Vocabulary: {}", tokenizer.vocab_size());
            println!("  BOS token: {:?}", tokenizer.bos_token_id());
            println!("  EOS token: {:?}", tokenizer.eos_token_id());

            // Test tokenization
            let test_text = "BitNet neural network inference";
            let tokens = tokenizer.encode(test_text, true, false)?;

            println!("\n🧪 Tokenization Test:");
            println!("  Input: {}", test_text);
            println!("  Tokens: {:?}", tokens);
            println!("  Token count: {}", tokens.len());

            // Round-trip validation
            let decoded = tokenizer.decode(&tokens)?;
            println!("\n🔄 Round-trip Validation:");
            println!("  Decoded: {}", decoded);
            println!("  Match: {}", test_text.trim() == decoded.trim());
        }
        None => {
            println!("❌ No embedded tokenizer found");
            println!("\n💡 Fallback Options:");
            println!("  1. Use co-located tokenizer.json");
            println!("  2. Download from HuggingFace Hub");
            println!("  3. Provide explicit tokenizer path");
        }
    }

    Ok(())
}
```

---

## Troubleshooting

**Problem**: "Embedded tokenizer extraction failed"

**Solution**: Check GGUF metadata keys:
```bash
# Inspect GGUF metadata (requires gguf-py or similar)
python -c "import gguf; reader = gguf.GGUFReader('model.gguf'); print(reader.metadata)"
```

---

**Problem**: "Vocabulary size mismatch"

**Solution**: Allow tolerance or use explicit vocabulary size:
```rust,no_run
// Tolerance: ±100 tokens is acceptable
let vocab_matches = (vocab.len() as i64 - expected_size as i64).abs() < 100;
```

---

**Problem**: "Binary SentencePiece model too small"

**Solution**: Model may be corrupted. Check file integrity or use alternative extraction:
```rust,no_run
// Minimum size check
if binary_model.len() < 1024 {
    eprintln!("Warning: Binary model too small ({} bytes), may be corrupted", binary_model.len());
}
```

---

## Next Steps

- [Automatic Tokenizer Discovery](automatic-tokenizer-discovery.md) - Full automatic discovery
- [Configure Fallback Strategies](configure-fallback-strategies.md) - Custom fallback chains
- [Detect Model Architecture](detect-model-architecture.md) - Architecture detection guide

---

**Task Complete!** You now know how to extract embedded tokenizers from GGUF files. ✅
