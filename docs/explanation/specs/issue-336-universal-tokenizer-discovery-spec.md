# Issue #336: Universal Tokenizer Discovery System - Neural Network Technical Specification

## Executive Summary

This specification defines the technical implementation approach for fixing the Universal Tokenizer Discovery System in BitNet.rs neural network inference engine. The implementation addresses 5 critical deficiencies preventing automatic tokenizer resolution from GGUF files, HuggingFace Hub downloads, and production neural network inference.

**Issue**: [Tokenizers] Implement Universal Tokenizer Discovery System for Production Neural Network Inference
**Labels**: enhancement, priority/high, area/developer-tools
**Flow Status**: generative:gate:spec = pass (Neural Network Implementation Analysis Complete)

**Problem Areas (5 Components)**:
1. **Embedded Tokenizer Extraction** - Returns BasicTokenizer instead of proper HuggingFace/SentencePiece tokenizers
2. **Model Type Detection** - Incomplete tensor pattern analysis for architecture detection
3. **Vocabulary Size Fallback** - Missing default vocabulary sizes for common architectures
4. **Smart Download Strategy** - Returns Ok(None) instead of downloading from HuggingFace Hub
5. **Fallback Chain Integration** - Unused `_fallback_chain` field with no systematic fallback

**Acceptance Criteria**:
- **AC1**: Embedded Tokenizer Support - GGUF files with embedded HuggingFace/SentencePiece tokenizers load correctly
- **AC2**: Model Architecture Detection - Correctly identifies BitNet, LLaMA, GPT-2, GPT-Neo, BERT, T5 architectures
- **AC3**: Vocabulary Size Resolution - Extracts/infers vocabulary size with sensible defaults for known architectures
- **AC4**: Smart Download Integration - Downloads compatible tokenizers from HuggingFace Hub with caching
- **AC5**: Production Readiness - Cross-validates with Microsoft BitNet C++ reference, >99% compatibility

---

## Neural Network Context Analysis

### Current BitNet.rs Tokenizer Infrastructure

**Existing Crates Structure**:
```
crates/bitnet-tokenizers/src/
├── discovery.rs           # TokenizerDiscovery (Lines 462-522: Embedded extraction)
├── strategy.rs            # TokenizerStrategyResolver (Lines 285-320: Model type detection)
├── download.rs            # SmartTokenizerDownload (Working implementation)
├── error_handling.rs      # TokenizerErrorHandler, CacheManager, ModelTypeDetector
├── fallback.rs            # Fallback chain infrastructure
├── hf_tokenizer.rs        # HuggingFace tokenizer backend
├── spm_tokenizer.rs       # SentencePiece tokenizer backend (feature-gated)
├── gguf_tokenizer.rs      # GGUF-embedded tokenizer (Basic implementation)
└── universal.rs           # Universal tokenizer interface
```

**Current Tokenizer Discovery Flow**:
```
GGUF Model File → TokenizerDiscovery::from_gguf()
     ↓
Extract Metadata (vocab_size, model_type)
     ↓
TokenizerDiscovery::discover_tokenizer_strategy()
     ↓
┌─────────────────────────────────────────┐
│ 1. try_extract_embedded_tokenizer()    │ ← PROBLEM AREA #1 (Lines 462-522)
│    - Currently returns BasicTokenizer   │
│    - Should return HfTokenizer/SpmTokenizer
│ 2. check_colocated_tokenizers()        │ ← Works
│ 3. check_cache_locations()             │ ← Works
│ 4. infer_download_source()             │ ← PROBLEM AREA #4 (Returns Ok(None))
│    - Should return TokenizerDownloadInfo
│ 5. Fallback to Mock (non-strict)       │ ← PROBLEM AREA #5 (No systematic fallback)
└─────────────────────────────────────────┘
     ↓
TokenizerStrategyResolver::resolve_tokenizer()
     ↓
Concrete Tokenizer Implementation (Arc<dyn Tokenizer>)
```

### Neural Network Requirements Analysis

**Vocabulary Scale Impact on Tokenization**:
- **LLaMA-3** (128,256 vocab): Large embedding tables require GPU acceleration for lookup efficiency
- **LLaMA-2** (32,000 vocab): CPU-compatible embedding tables with vectorized operations
- **GPT-2** (50,257 vocab): Standard BPE tokenization with 50K vocabulary
- **BitNet Custom**: Variable vocabulary sizes requiring adaptive detection

**Quantization Format Compatibility**:
- **I2S**: 2-bit signed quantization - optimal for large vocabularies with GPU acceleration
- **TL1/TL2**: Table lookup quantization - efficient for smaller vocabularies
- **Tokenizer Compatibility**: Must support all quantization formats without pipeline interference

**Performance Considerations**:
- **Token Lookup**: O(1) performance critical for large vocabularies
- **Memory Bandwidth**: Device-aware tokenization to minimize GPU/CPU transfer overhead
- **Embedded Extraction**: Zero-copy GGUF metadata parsing for embedded tokenizers
- **Download Caching**: Persistent cache to avoid repeated HuggingFace Hub downloads

---

## Problem Area #1: Embedded Tokenizer Extraction

### Current Implementation Analysis

**Location**: `crates/bitnet-tokenizers/src/discovery.rs:462-522`

**Current Behavior**:
```rust
pub fn try_extract_embedded_tokenizer(&self) -> Result<Option<Arc<dyn Tokenizer>>> {
    debug!("Attempting to extract embedded tokenizer from GGUF metadata");

    // Check if tokenizer model is embedded as bytes
    if let Some(tokenizer_model) = self.gguf_reader.get_array_metadata("tokenizer.ggml.model") {
        debug!("Found embedded tokenizer.ggml.model ({} bytes)", tokenizer_model.len());

        // ❌ PROBLEM: Creates BasicTokenizer instead of proper HfTokenizer/SpmTokenizer
        if tokenizer_model.len() > 100 {
            let basic_tokenizer = crate::BasicTokenizer::with_config(
                self.vocab_size,
                Some(1), // BOS token
                Some(2), // EOS token
                Some(0), // PAD token
            );

            debug!("Created basic tokenizer from GGUF metadata");
            return Ok(Some(Arc::new(basic_tokenizer)));
        }
    }

    // ❌ PROBLEM: Also creates BasicTokenizer for embedded vocabulary
    if let Some(vocab) = self.gguf_reader.get_string_array_metadata("tokenizer.ggml.tokens")
        && vocab.len() == self.vocab_size
    {
        debug!("Found embedded vocabulary with {} tokens", vocab.len());

        let basic_tokenizer = crate::BasicTokenizer::with_config(
            self.vocab_size,
            self.gguf_reader.get_u32_metadata("tokenizer.ggml.bos_token_id"),
            self.gguf_reader.get_u32_metadata("tokenizer.ggml.eos_token_id"),
            self.gguf_reader.get_u32_metadata("tokenizer.ggml.pad_token_id"),
        );

        debug!("Created tokenizer with embedded vocabulary");
        return Ok(Some(Arc::new(basic_tokenizer)));
    }

    // ❌ PROBLEM: Creates BasicTokenizer for embedded tokenizer.json
    if let Some(tokenizer_json) = self.gguf_reader.get_string_metadata("tokenizer.json") {
        debug!("Found embedded tokenizer.json ({} chars)", tokenizer_json.len());

        // Should parse JSON and create HfTokenizer, but currently returns BasicTokenizer
        let basic_tokenizer = crate::BasicTokenizer::with_config(
            self.vocab_size,
            Some(1), // BOS token
            Some(2), // EOS token
            Some(0), // PAD token
        );

        debug!("Created tokenizer from embedded JSON metadata");
        return Ok(Some(Arc::new(basic_tokenizer)));
    }

    debug!("No embedded tokenizer found in GGUF metadata");
    Ok(None)
}
```

**Issues**:
1. **BasicTokenizer Fallback**: Always returns `BasicTokenizer` instead of proper `HfTokenizer` or `SpmTokenizer`
2. **No JSON Parsing**: `tokenizer.json` string metadata not parsed into HuggingFace tokenizer
3. **No SentencePiece Support**: `tokenizer.ggml.model` bytes not interpreted as SentencePiece model
4. **Missing Type Detection**: No detection of tokenizer type from metadata (BPE, WordPiece, Unigram, SentencePiece)

### Technical Approach

**Implementation Strategy**:
```rust
// crates/bitnet-tokenizers/src/discovery.rs

pub fn try_extract_embedded_tokenizer(&self) -> Result<Option<Arc<dyn Tokenizer>>> {
    debug!("Attempting to extract embedded tokenizer from GGUF metadata");

    // Strategy 1: Check for embedded tokenizer.json (HuggingFace format)
    if let Some(tokenizer_json) = self.gguf_reader.get_string_metadata("tokenizer.json") {
        debug!("Found embedded tokenizer.json ({} chars)", tokenizer_json.len());

        // Parse JSON and create HfTokenizer
        match self.parse_hf_tokenizer_from_json(&tokenizer_json) {
            Ok(hf_tokenizer) => {
                debug!("Successfully created HuggingFace tokenizer from embedded JSON");
                return Ok(Some(Arc::new(hf_tokenizer)));
            }
            Err(e) => {
                warn!("Failed to parse embedded tokenizer.json: {}", e);
                // Continue to next strategy
            }
        }
    }

    // Strategy 2: Check for embedded SentencePiece model (tokenizer.ggml.model)
    #[cfg(feature = "spm")]
    if let Some(tokenizer_model) = self.gguf_reader.get_array_metadata("tokenizer.ggml.model") {
        debug!("Found embedded tokenizer.ggml.model ({} bytes)", tokenizer_model.len());

        // Parse SentencePiece model bytes
        match self.parse_spm_tokenizer_from_bytes(&tokenizer_model) {
            Ok(spm_tokenizer) => {
                debug!("Successfully created SentencePiece tokenizer from embedded model");
                return Ok(Some(Arc::new(spm_tokenizer)));
            }
            Err(e) => {
                warn!("Failed to parse embedded SentencePiece model: {}", e);
                // Continue to next strategy
            }
        }
    }

    // Strategy 3: Check for embedded vocabulary with merges (BPE format)
    if let Some(vocab) = self.gguf_reader.get_string_array_metadata("tokenizer.ggml.tokens")
        && let Some(merges) = self.gguf_reader.get_string_array_metadata("tokenizer.ggml.merges")
    {
        debug!("Found embedded BPE vocabulary with {} tokens and {} merges", vocab.len(), merges.len());

        match self.create_bpe_tokenizer_from_vocab(&vocab, &merges) {
            Ok(bpe_tokenizer) => {
                debug!("Successfully created BPE tokenizer from embedded vocabulary");
                return Ok(Some(Arc::new(bpe_tokenizer)));
            }
            Err(e) => {
                warn!("Failed to create BPE tokenizer from embedded vocabulary: {}", e);
                // Continue to next strategy
            }
        }
    }

    // Strategy 4: Check for embedded vocabulary only (WordPiece/Unigram format)
    if let Some(vocab) = self.gguf_reader.get_string_array_metadata("tokenizer.ggml.tokens")
        && vocab.len() == self.vocab_size
    {
        debug!("Found embedded vocabulary with {} tokens", vocab.len());

        // Detect tokenizer type from metadata
        let tokenizer_type = self.gguf_reader
            .get_string_metadata("tokenizer.ggml.type")
            .unwrap_or_else(|| "bpe".to_string());

        match self.create_tokenizer_from_vocab(&vocab, &tokenizer_type) {
            Ok(tokenizer) => {
                debug!("Successfully created {} tokenizer from embedded vocabulary", tokenizer_type);
                return Ok(Some(Arc::new(tokenizer)));
            }
            Err(e) => {
                warn!("Failed to create tokenizer from embedded vocabulary: {}", e);
                // Fallback to BasicTokenizer for compatibility
            }
        }

        // Fallback: Create BasicTokenizer with special tokens
        let basic_tokenizer = crate::BasicTokenizer::with_config(
            self.vocab_size,
            self.gguf_reader.get_u32_metadata("tokenizer.ggml.bos_token_id"),
            self.gguf_reader.get_u32_metadata("tokenizer.ggml.eos_token_id"),
            self.gguf_reader.get_u32_metadata("tokenizer.ggml.pad_token_id"),
        );

        debug!("Created BasicTokenizer fallback from embedded vocabulary");
        return Ok(Some(Arc::new(basic_tokenizer)));
    }

    debug!("No embedded tokenizer found in GGUF metadata");
    Ok(None)
}

// Helper: Parse HuggingFace tokenizer from embedded JSON
fn parse_hf_tokenizer_from_json(&self, json_str: &str) -> Result<crate::HfTokenizer> {
    use serde_json::Value;

    let json: Value = serde_json::from_str(json_str).map_err(|e| {
        BitNetError::Config(format!("Failed to parse embedded tokenizer.json: {}", e))
    })?;

    // Extract tokenizer configuration from JSON
    let model_type = json.get("model")
        .and_then(|m| m.get("type"))
        .and_then(|t| t.as_str())
        .ok_or_else(|| BitNetError::Config("Missing model.type in tokenizer.json".to_string()))?;

    let vocab = json.get("model")
        .and_then(|m| m.get("vocab"))
        .ok_or_else(|| BitNetError::Config("Missing model.vocab in tokenizer.json".to_string()))?;

    // Create HfTokenizer from JSON configuration
    crate::HfTokenizer::from_json_value(json)
}

// Helper: Parse SentencePiece tokenizer from embedded bytes
#[cfg(feature = "spm")]
fn parse_spm_tokenizer_from_bytes(&self, model_bytes: &[u8]) -> Result<crate::SpmTokenizer> {
    // SentencePiece model format validation
    if model_bytes.len() < 10 {
        return Err(BitNetError::Config("SentencePiece model too small".to_string()));
    }

    // Parse SentencePiece protobuf format
    crate::SpmTokenizer::from_bytes(model_bytes)
}

// Helper: Create BPE tokenizer from vocabulary and merges
fn create_bpe_tokenizer_from_vocab(
    &self,
    vocab: &[String],
    merges: &[String],
) -> Result<crate::HfTokenizer> {
    // Create HuggingFace BPE tokenizer from vocabulary and merges
    let mut tokenizer_config = serde_json::json!({
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": {},
            "merges": merges,
        },
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false,
        },
    });

    // Build vocabulary map
    let vocab_map: serde_json::Map<String, serde_json::Value> = vocab
        .iter()
        .enumerate()
        .map(|(i, token)| (token.clone(), serde_json::Value::Number(i.into())))
        .collect();

    tokenizer_config["model"]["vocab"] = serde_json::Value::Object(vocab_map);

    crate::HfTokenizer::from_json_value(tokenizer_config)
}

// Helper: Create tokenizer from vocabulary and type
fn create_tokenizer_from_vocab(
    &self,
    vocab: &[String],
    tokenizer_type: &str,
) -> Result<crate::HfTokenizer> {
    match tokenizer_type {
        "bpe" => {
            // BPE without merges - use character-level fallback
            let tokenizer_config = serde_json::json!({
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "vocab": vocab.iter().enumerate()
                        .map(|(i, token)| (token.clone(), i))
                        .collect::<serde_json::Map<_, _>>(),
                    "merges": [],
                },
                "normalizer": null,
                "pre_tokenizer": null,
            });
            crate::HfTokenizer::from_json_value(tokenizer_config)
        }
        "wordpiece" => {
            // WordPiece tokenizer
            let tokenizer_config = serde_json::json!({
                "version": "1.0",
                "model": {
                    "type": "WordPiece",
                    "vocab": vocab.iter().enumerate()
                        .map(|(i, token)| (token.clone(), i))
                        .collect::<serde_json::Map<_, _>>(),
                    "unk_token": "[UNK]",
                },
                "normalizer": null,
                "pre_tokenizer": null,
            });
            crate::HfTokenizer::from_json_value(tokenizer_config)
        }
        "unigram" => {
            // Unigram tokenizer (SentencePiece variant)
            let tokenizer_config = serde_json::json!({
                "version": "1.0",
                "model": {
                    "type": "Unigram",
                    "vocab": vocab.iter().enumerate()
                        .map(|(i, token)| (token.clone(), 0.0)) // Default score
                        .collect::<Vec<_>>(),
                },
                "normalizer": null,
                "pre_tokenizer": null,
            });
            crate::HfTokenizer::from_json_value(tokenizer_config)
        }
        _ => Err(BitNetError::Config(format!("Unsupported tokenizer type: {}", tokenizer_type)))
    }
}
```

**Dependencies**:
- **Existing**: `bitnet-tokenizers/src/hf_tokenizer.rs` - HuggingFace tokenizer backend
- **Existing**: `bitnet-tokenizers/src/spm_tokenizer.rs` - SentencePiece tokenizer backend (feature-gated)
- **New**: `HfTokenizer::from_json_value()` - Constructor from JSON value
- **New**: `SpmTokenizer::from_bytes()` - Constructor from protobuf bytes

**Risk Assessment**: **Medium Risk**
- **JSON Parsing**: HuggingFace tokenizer.json format is complex with nested configuration
- **SentencePiece Protobuf**: Binary format parsing requires careful handling
- **Backward Compatibility**: Must maintain fallback to BasicTokenizer for unsupported formats

**Mitigation Strategies**:
1. **Validation Commands**: Test with real GGUF files containing embedded tokenizers
2. **Error Handling**: Graceful fallback to next strategy on parsing failures
3. **Feature Gating**: SentencePiece support behind `spm` feature flag
4. **Logging**: Debug-level logging for tokenizer type detection and parsing progress

**Validation Commands**:
```bash
# Test embedded HuggingFace tokenizer extraction
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_extract_embedded_hf_tokenizer

# Test embedded SentencePiece tokenizer extraction (feature-gated)
cargo test --no-default-features --features cpu,spm -p bitnet-tokenizers test_extract_embedded_spm_tokenizer

# Test embedded BPE tokenizer with vocabulary and merges
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_extract_embedded_bpe_tokenizer

# Integration test with real GGUF files
cargo run -p bitnet-cli --no-default-features --features cpu -- compat-check test-models/llama3-embedded.gguf
```

---

## Problem Area #2: Model Type Detection

### Current Implementation Analysis

**Location**: `crates/bitnet-tokenizers/src/discovery.rs:285-320`

**Current Behavior**:
```rust
fn extract_model_type(reader: &GgufReader) -> Result<String> {
    // Try to get architecture from metadata
    if let Some(arch) = reader.get_string_metadata("general.architecture") {
        return Ok(arch);
    }

    // Alternative metadata keys
    let alt_keys = [
        "model.architecture",
        "transformer.architecture",
        "llama.architecture",
        "gpt.architecture",
    ];

    for key in &alt_keys {
        if let Some(arch) = reader.get_string_metadata(key) {
            return Ok(arch);
        }
    }

    // Try to infer from model name
    if let Some(name) = reader.get_string_metadata("general.name") {
        let name_lower = name.to_lowercase();
        if name_lower.contains("llama") {
            return Ok("llama".to_string());
        } else if name_lower.contains("gpt") {
            return Ok("gpt2".to_string());
        } else if name_lower.contains("bitnet") {
            return Ok("bitnet".to_string());
        }
    }

    // ❌ PROBLEM: Fallback based on tensor patterns is too simplistic
    let tensor_names = reader.tensor_names();
    let has_llama_patterns = tensor_names.iter().any(|name| {
        name.contains("attn_q") || name.contains("attn_k") || name.contains("attn_v")
    });

    if has_llama_patterns {
        Ok("llama".to_string())
    } else {
        Ok("transformer".to_string()) // ❌ Too generic
    }
}
```

**Issues**:
1. **Incomplete Tensor Pattern Analysis**: Only checks for `attn_q/k/v` patterns, missing other architectures
2. **No BitNet Detection**: Cannot identify BitNet-specific tensor patterns (1.58-bit quantization signatures)
3. **No GPT-Neo Detection**: Missing GPT-Neo specific patterns (rotary embeddings, parallel attention/MLP)
4. **No BERT Detection**: Missing encoder-only patterns (pooler, bidirectional attention)
5. **No T5 Detection**: Missing encoder-decoder patterns (cross-attention layers)

### Technical Approach

**Implementation Strategy**:
```rust
// crates/bitnet-tokenizers/src/discovery.rs

fn extract_model_type(reader: &GgufReader) -> Result<String> {
    // Strategy 1: Check explicit metadata keys
    if let Some(arch) = reader.get_string_metadata("general.architecture") {
        return Ok(normalize_architecture_name(&arch));
    }

    // Strategy 2: Check alternative metadata keys
    let alt_keys = [
        "model.architecture",
        "transformer.architecture",
        "llama.architecture",
        "gpt.architecture",
        "bert.architecture",
        "t5.architecture",
        "gpt_neox.architecture",
    ];

    for key in &alt_keys {
        if let Some(arch) = reader.get_string_metadata(key) {
            return Ok(normalize_architecture_name(&arch));
        }
    }

    // Strategy 3: Infer from model name with expanded pattern matching
    if let Some(name) = reader.get_string_metadata("general.name") {
        if let Some(arch) = infer_architecture_from_name(&name) {
            return Ok(arch);
        }
    }

    // Strategy 4: Comprehensive tensor pattern analysis
    let tensor_names: Vec<String> = reader.tensor_names().iter().map(|s| s.to_string()).collect();
    detect_architecture_from_tensors(&tensor_names)
}

// Helper: Normalize architecture names to canonical forms
fn normalize_architecture_name(arch: &str) -> String {
    let arch_lower = arch.to_lowercase();

    match arch_lower.as_str() {
        // LLaMA variants
        "llama" | "llama2" | "llama3" | "codellama" => "llama",

        // GPT variants
        "gpt2" | "gpt-2" | "gpt" => "gpt2",
        "gpt-neo" | "gptneo" | "gpt_neo" => "gpt-neo",
        "gpt-neox" | "gptneox" | "gpt_neox" => "gpt-neox",
        "gpt-j" | "gptj" | "gpt_j" => "gpt-j",

        // BERT variants
        "bert" | "roberta" | "albert" | "distilbert" => "bert",

        // T5 variants
        "t5" | "mt5" | "byt5" => "t5",

        // BitNet variants
        "bitnet" | "bitnet_b1_58" | "bitnet-b1.58" => "bitnet",

        // Generic transformer
        _ => arch_lower,
    }
}

// Helper: Infer architecture from model name patterns
fn infer_architecture_from_name(name: &str) -> Option<String> {
    let name_lower = name.to_lowercase();

    // BitNet patterns (highest priority)
    if name_lower.contains("bitnet") || name_lower.contains("1.58") || name_lower.contains("b1_58") {
        return Some("bitnet".to_string());
    }

    // LLaMA patterns
    if name_lower.contains("llama") {
        return Some("llama".to_string());
    }

    // GPT patterns
    if name_lower.contains("gpt-neo") {
        return Some("gpt-neo".to_string());
    }
    if name_lower.contains("gpt-j") {
        return Some("gpt-j".to_string());
    }
    if name_lower.contains("gpt-neox") || name_lower.contains("pythia") {
        return Some("gpt-neox".to_string());
    }
    if name_lower.contains("gpt") {
        return Some("gpt2".to_string());
    }

    // BERT patterns
    if name_lower.contains("bert") {
        return Some("bert".to_string());
    }

    // T5 patterns
    if name_lower.contains("t5") {
        return Some("t5".to_string());
    }

    None
}

// Helper: Detect architecture from comprehensive tensor pattern analysis
fn detect_architecture_from_tensors(tensor_names: &[String]) -> Result<String> {
    // Count tensor pattern occurrences for each architecture
    let mut pattern_scores: HashMap<&str, usize> = HashMap::new();

    // BitNet-specific patterns
    let bitnet_patterns = [
        "bitnet",           // Explicit BitNet naming
        "bitlinear",        // BitLinear layers
        "scale",            // 1.58-bit scale parameters
        "1_58",             // 1.58-bit naming
    ];
    let bitnet_score = count_pattern_matches(tensor_names, &bitnet_patterns);
    if bitnet_score > 0 {
        pattern_scores.insert("bitnet", bitnet_score);
    }

    // LLaMA-specific patterns
    let llama_patterns = [
        "attn_q", "attn_k", "attn_v",  // Standard attention projections
        "attn_output",                  // Output projection
        "ffn_gate", "ffn_up", "ffn_down", // SwiGLU FFN
        "attn_norm", "ffn_norm",        // RMSNorm layers
    ];
    let llama_score = count_pattern_matches(tensor_names, &llama_patterns);
    if llama_score >= 7 {  // Require most LLaMA patterns
        pattern_scores.insert("llama", llama_score);
    }

    // GPT-Neo specific patterns
    let gpt_neo_patterns = [
        "attn.attention",   // GPT-Neo attention naming
        "attn.out_proj",    // Output projection naming
        "mlp.c_fc",         // MLP first layer
        "mlp.c_proj",       // MLP projection
        "ln_1", "ln_2",     // Layer norm naming
    ];
    let gpt_neo_score = count_pattern_matches(tensor_names, &gpt_neo_patterns);
    if gpt_neo_score >= 5 {
        pattern_scores.insert("gpt-neo", gpt_neo_score);
    }

    // GPT-NeoX specific patterns
    let gpt_neox_patterns = [
        "attention.query_key_value", // Fused QKV
        "attention.dense",           // Output projection
        "mlp.dense_h_to_4h",        // MLP expansion
        "mlp.dense_4h_to_h",        // MLP reduction
        "input_layernorm",           // Input LayerNorm
        "post_attention_layernorm",  // Post-attention LayerNorm
    ];
    let gpt_neox_score = count_pattern_matches(tensor_names, &gpt_neox_patterns);
    if gpt_neox_score >= 6 {
        pattern_scores.insert("gpt-neox", gpt_neox_score);
    }

    // GPT-2 specific patterns
    let gpt2_patterns = [
        "wte",              // Token embeddings
        "wpe",              // Position embeddings
        "attn.c_attn",      // Fused QKV
        "attn.c_proj",      // Output projection
        "mlp.c_fc",         // MLP first layer
        "mlp.c_proj",       // MLP projection
        "ln_1", "ln_2",     // Layer norms
    ];
    let gpt2_score = count_pattern_matches(tensor_names, &gpt2_patterns);
    if gpt2_score >= 7 {
        pattern_scores.insert("gpt2", gpt2_score);
    }

    // BERT-specific patterns (encoder-only)
    let bert_patterns = [
        "encoder.layer",         // Encoder layers
        "attention.self.query",  // Self-attention query
        "attention.self.key",    // Self-attention key
        "attention.self.value",  // Self-attention value
        "attention.output.dense", // Attention output
        "pooler.dense",          // Pooler layer (BERT-specific)
        "LayerNorm",             // LayerNorm naming
    ];
    let bert_score = count_pattern_matches(tensor_names, &bert_patterns);
    if bert_score >= 6 {
        pattern_scores.insert("bert", bert_score);
    }

    // T5-specific patterns (encoder-decoder)
    let t5_patterns = [
        "encoder.block",         // Encoder blocks
        "decoder.block",         // Decoder blocks
        "SelfAttention",         // Self-attention
        "EncDecAttention",       // Cross-attention (T5-specific)
        "DenseReluDense",        // FFN naming
        "layer_norm",            // Layer norm naming
        "relative_attention_bias", // Relative position embeddings (T5-specific)
    ];
    let t5_score = count_pattern_matches(tensor_names, &t5_patterns);
    if t5_score >= 6 {
        pattern_scores.insert("t5", t5_score);
    }

    // Find architecture with highest pattern score
    if let Some((arch, score)) = pattern_scores.iter().max_by_key(|(_, &score)| score) {
        debug!("Detected architecture '{}' from tensor patterns (score: {})", arch, score);
        return Ok(arch.to_string());
    }

    // Fallback: Generic transformer detection
    let generic_transformer_patterns = [
        "attn", "attention",    // Any attention layers
        "mlp", "ffn",           // Any feed-forward layers
        "norm", "layernorm",    // Any normalization
    ];
    let transformer_score = count_pattern_matches(tensor_names, &generic_transformer_patterns);

    if transformer_score >= 3 {
        debug!("Detected generic transformer architecture from tensor patterns");
        Ok("transformer".to_string())
    } else {
        Err(BitNetError::Config(
            "Cannot determine model architecture from tensor patterns".to_string()
        ))
    }
}

// Helper: Count how many patterns match in tensor names
fn count_pattern_matches(tensor_names: &[String], patterns: &[&str]) -> usize {
    patterns.iter()
        .filter(|&&pattern| {
            tensor_names.iter().any(|name| name.to_lowercase().contains(pattern))
        })
        .count()
}
```

**Dependencies**:
- **Existing**: `bitnet-models/src/gguf_simple.rs` - GGUF tensor name extraction
- **New**: Comprehensive tensor pattern database for architecture detection

**Risk Assessment**: **Low Risk**
- **Pattern Matching**: Simple string matching with no complex parsing
- **Backward Compatibility**: Maintains existing metadata-based detection as primary strategy
- **Fallback**: Generic "transformer" classification ensures no failures

**Validation Commands**:
```bash
# Test LLaMA architecture detection
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_detect_llama_architecture

# Test GPT-Neo architecture detection
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_detect_gpt_neo_architecture

# Test BitNet architecture detection
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_detect_bitnet_architecture

# Test BERT architecture detection
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_detect_bert_architecture

# Test T5 architecture detection
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_detect_t5_architecture

# Integration test with real GGUF files
cargo run -p bitnet-cli --no-default-features --features cpu -- compat-check test-models/various-architectures/*.gguf
```

---

## Problem Area #3: Vocabulary Size Fallback

### Current Implementation Analysis

**Location**: `crates/bitnet-tokenizers/src/discovery.rs:322-365` and `crates/bitnet-tokenizers/src/discovery.rs:256-294`

**Current Behavior**:
```rust
fn extract_vocab_size(reader: &GgufReader) -> Result<usize> {
    // Try to get vocab size from metadata first
    if let Some(vocab_size) = reader.get_u32_metadata("tokenizer.ggml.vocab_size") {
        return Ok(vocab_size as usize);
    }

    // Alternative metadata keys for different model architectures
    let alt_keys = [
        "llama.vocab_size",
        "gpt2.vocab_size",
        "transformer.vocab_size",
        "model.vocab_size"
    ];

    for key in &alt_keys {
        if let Some(vocab_size) = reader.get_u32_metadata(key) {
            return Ok(vocab_size as usize);
        }
    }

    // Look for embedding tensor to infer vocab size
    let tensor_names = reader.tensor_names();
    for name in tensor_names {
        if (name.contains("token_embd") || name.contains("wte") || name.contains("embed"))
            && let Some(info) = reader.get_tensor_info_by_name(name)
        {
            // Embeddings are typically [vocab_size, hidden_dim]
            let shape = &info.shape;
            if shape.len() >= 2 {
                let possible_vocab = std::cmp::max(shape[0], shape[1]);
                // Sanity check - vocab size should be reasonable
                if possible_vocab > 1000 && possible_vocab < 2_000_000 {
                    return Ok(possible_vocab);
                }
            }
        }
    }

    // ❌ PROBLEM: No default fallback for known architectures
    Err(TokenizerErrorHandler::config_error(
        "Could not extract vocabulary size from GGUF metadata or tensors".to_string(),
    ))
}
```

**Issues**:
1. **No Architecture-Specific Defaults**: Missing default vocabulary sizes for LLaMA-2 (32K), LLaMA-3 (128K), GPT-2 (50K)
2. **No Model Name Inference**: Doesn't use model name to infer likely vocabulary size
3. **Incomplete Metadata Keys**: Missing architecture-specific metadata keys (bert.vocab_size, t5.vocab_size)
4. **Poor Error Messages**: Generic error doesn't suggest likely vocabulary sizes for detected architecture

### Technical Approach

**Implementation Strategy**:
```rust
// crates/bitnet-tokenizers/src/discovery.rs

fn extract_vocab_size(reader: &GgufReader) -> Result<usize> {
    // Strategy 1: Try to get vocab size from metadata first
    if let Some(vocab_size) = reader.get_u32_metadata("tokenizer.ggml.vocab_size") {
        return Ok(vocab_size as usize);
    }

    // Strategy 2: Alternative metadata keys for different model architectures
    let alt_keys = [
        "llama.vocab_size",
        "gpt2.vocab_size",
        "gpt_neox.vocab_size",
        "bert.vocab_size",
        "t5.vocab_size",
        "transformer.vocab_size",
        "model.vocab_size",
    ];

    for key in &alt_keys {
        if let Some(vocab_size) = reader.get_u32_metadata(key) {
            return Ok(vocab_size as usize);
        }
    }

    // Strategy 3: Infer from embedded vocabulary
    if let Some(vocab) = reader.get_string_array_metadata("tokenizer.ggml.tokens") {
        debug!("Inferred vocab size from embedded vocabulary: {}", vocab.len());
        return Ok(vocab.len());
    }

    // Strategy 4: Look for embedding tensor to infer vocab size
    let tensor_names = reader.tensor_names();
    for name in tensor_names {
        if (name.contains("token_embd")
            || name.contains("wte")
            || name.contains("embed_tokens")
            || name.contains("word_embeddings"))
            && let Some(info) = reader.get_tensor_info_by_name(name)
        {
            // Embeddings are typically [vocab_size, hidden_dim]
            let shape = &info.shape;
            if shape.len() >= 2 {
                let possible_vocab = std::cmp::max(shape[0], shape[1]);
                // Sanity check - vocab size should be reasonable
                if possible_vocab > 1000 && possible_vocab < 2_000_000 {
                    debug!("Inferred vocab size from embedding tensor '{}': {}", name, possible_vocab);
                    return Ok(possible_vocab);
                }
            }
        }
    }

    // Strategy 5: Use architecture-specific defaults based on model type
    if let Ok(model_type) = Self::extract_model_type(reader) {
        if let Some(default_vocab) = get_default_vocab_size(&model_type) {
            warn!(
                "Could not extract vocabulary size from GGUF metadata, using default for {}: {}",
                model_type, default_vocab
            );
            return Ok(default_vocab);
        }
    }

    // Strategy 6: Use model name to infer vocabulary size
    if let Some(name) = reader.get_string_metadata("general.name") {
        if let Some(vocab_size) = infer_vocab_from_model_name(&name) {
            warn!(
                "Could not extract vocabulary size from GGUF metadata, inferred from model name '{}': {}",
                name, vocab_size
            );
            return Ok(vocab_size);
        }
    }

    // Final fallback: Use conservative default with clear warning
    warn!("Could not extract vocabulary size from GGUF metadata or tensors, using conservative default: 32000");
    Ok(32000) // LLaMA-2 default as conservative fallback
}

// Helper: Get default vocabulary size for known architectures
fn get_default_vocab_size(model_type: &str) -> Option<usize> {
    match model_type {
        // LLaMA variants
        "llama" => Some(32000),     // LLaMA-2 default
        "llama2" => Some(32000),
        "llama3" => Some(128256),   // LLaMA-3 extended vocabulary
        "codellama" => Some(32016), // CodeLlama extended with special tokens

        // GPT variants
        "gpt2" => Some(50257),      // GPT-2 standard vocabulary
        "gpt-neo" => Some(50257),   // GPT-Neo uses GPT-2 tokenizer
        "gpt-j" => Some(50400),     // GPT-J extended vocabulary
        "gpt-neox" => Some(50432),  // GPT-NeoX extended vocabulary

        // BERT variants
        "bert" => Some(30522),      // BERT base vocabulary
        "roberta" => Some(50265),   // RoBERTa extended vocabulary
        "albert" => Some(30000),    // ALBERT vocabulary

        // T5 variants
        "t5" => Some(32128),        // T5 SentencePiece vocabulary
        "mt5" => Some(250000),      // mT5 multilingual vocabulary

        // BitNet variants
        "bitnet" => Some(32000),    // BitNet typically uses LLaMA-2 tokenizer

        // Unknown architecture
        _ => None,
    }
}

// Helper: Infer vocabulary size from model name patterns
fn infer_vocab_from_model_name(name: &str) -> Option<usize> {
    let name_lower = name.to_lowercase();

    // LLaMA-3 patterns
    if name_lower.contains("llama-3") || name_lower.contains("llama3") {
        return Some(128256);
    }

    // LLaMA-2 patterns
    if name_lower.contains("llama-2") || name_lower.contains("llama2") {
        return Some(32000);
    }

    // CodeLlama patterns
    if name_lower.contains("codellama") || name_lower.contains("code-llama") {
        return Some(32016);
    }

    // GPT-2 patterns
    if name_lower.contains("gpt2") || name_lower.contains("gpt-2") {
        return Some(50257);
    }

    // GPT-Neo patterns
    if name_lower.contains("gpt-neo") || name_lower.contains("gptneo") {
        return Some(50257);
    }

    // GPT-J patterns
    if name_lower.contains("gpt-j") || name_lower.contains("gptj") {
        return Some(50400);
    }

    // GPT-NeoX / Pythia patterns
    if name_lower.contains("gpt-neox") || name_lower.contains("pythia") {
        return Some(50432);
    }

    // BERT patterns
    if name_lower.contains("bert") && !name_lower.contains("roberta") {
        return Some(30522);
    }

    // RoBERTa patterns
    if name_lower.contains("roberta") {
        return Some(50265);
    }

    // T5 patterns
    if name_lower.contains("t5") && !name_lower.contains("mt5") {
        return Some(32128);
    }

    // mT5 patterns
    if name_lower.contains("mt5") {
        return Some(250000);
    }

    // BitNet patterns
    if name_lower.contains("bitnet") || name_lower.contains("1.58") {
        return Some(32000);
    }

    None
}
```

**Dependencies**:
- **Existing**: `crates/bitnet-tokenizers/src/error_handling.rs::ModelTypeDetector` - Architecture detection utilities
- **Enhanced**: Architecture-specific vocabulary size database

**Risk Assessment**: **Low Risk**
- **Conservative Defaults**: Uses well-known vocabulary sizes for established architectures
- **Fallback Strategy**: Multiple strategies with progressive fallback to conservative defaults
- **Warning Logging**: Clear warnings when using inferred vocabulary sizes

**Validation Commands**:
```bash
# Test vocabulary size extraction from metadata
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_extract_vocab_size_metadata

# Test vocabulary size extraction from embedding tensors
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_extract_vocab_size_tensor

# Test architecture-specific vocabulary size defaults
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_vocab_size_defaults

# Test model name inference
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_vocab_size_from_model_name

# Integration test with incomplete GGUF files
cargo run -p bitnet-cli --no-default-features --features cpu -- compat-check test-models/incomplete-metadata.gguf
```

---

## Problem Area #4: Smart Download Strategy

### Current Implementation Analysis

**Location**: `crates/bitnet-tokenizers/src/strategy.rs:93-211` (resolve_with_fallback method)

**Current Behavior**:
```rust
pub async fn resolve_with_fallback(&self) -> Result<Arc<dyn Tokenizer>> {
    let mut errors = Vec::new();

    // Strategy 1: Try GGUF embedded tokenizer
    match self.discovery.try_extract_embedded_tokenizer() {
        Ok(Some(embedded_tokenizer)) => {
            info!("Successfully resolved tokenizer from GGUF metadata");
            return self.configure_model_specific_wrapper(embedded_tokenizer);
        }
        Ok(None) => {
            debug!("No embedded tokenizer found in GGUF");
        }
        Err(e) => {
            warn!("Failed to extract embedded tokenizer: {}", e);
            errors.push(("GGUF embedded", e));
        }
    }

    // Strategy 2: Try co-located files
    match self.discovery.check_colocated_tokenizers() {
        Ok(Some(path)) => {
            info!("Found co-located tokenizer at: {}", path.display());
            match self.load_tokenizer_from_path(&path) {
                Ok(tokenizer) => return Ok(tokenizer),
                Err(e) => {
                    warn!("Failed to load co-located tokenizer: {}", e);
                    errors.push(("co-located files", e));
                }
            }
        }
        Ok(None) => {
            debug!("No co-located tokenizer files found");
        }
        Err(e) => {
            warn!("Error checking co-located tokenizers: {}", e);
            errors.push(("co-located search", e));
        }
    }

    // Strategy 3: Try standard cache directories
    match self.discovery.check_cache_locations() {
        Ok(Some(path)) => {
            info!("Found cached tokenizer at: {}", path.display());
            match self.load_tokenizer_from_path(&path) {
                Ok(tokenizer) => return Ok(tokenizer),
                Err(e) => {
                    warn!("Failed to load cached tokenizer: {}", e);
                    errors.push(("cache directories", e));
                }
            }
        }
        Ok(None) => {
            debug!("No cached tokenizer found");
        }
        Err(e) => {
            warn!("Error checking cache locations: {}", e);
            errors.push(("cache search", e));
        }
    }

    // ❌ PROBLEM: Strategy 4 returns Ok(None) instead of downloading
    if std::env::var("BITNET_OFFLINE").is_err() {
        match self.discovery.infer_download_source() {
            Ok(Some(download_info)) => {
                info!("Attempting smart download from: {}", download_info.repo);
                // ❌ SHOULD DOWNLOAD HERE, but currently just returns Ok(None)
                match self.downloader.download_tokenizer(&download_info).await {
                    Ok(downloaded_path) => {
                        match self.load_tokenizer_from_path(&downloaded_path) {
                            Ok(tokenizer) => return Ok(tokenizer),
                            Err(e) => {
                                warn!("Failed to load downloaded tokenizer: {}", e);
                                errors.push(("smart download loading", e));
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Smart download failed: {}", e);
                        errors.push(("smart download", e));
                    }
                }
            }
            Ok(None) => {
                debug!("No download source available for this model");
                // ❌ PROBLEM: Returns Ok(None) here instead of attempting download
            }
            Err(e) => {
                warn!("Error determining download source: {}", e);
                errors.push(("download source detection", e));
            }
        }
    } else {
        debug!("Skipping smart download (offline mode)");
    }

    // Strategy 5: Mock fallback (non-strict mode only)
    if std::env::var("BITNET_STRICT_TOKENIZERS").is_err() {
        info!("Falling back to mock tokenizer");
        let mock_tokenizer = Arc::new(crate::MockTokenizer::new());
        return self.configure_model_specific_wrapper(mock_tokenizer);
    }

    // All strategies failed
    let error_summary = format!(
        "All tokenizer resolution strategies failed. Tried: {}. Errors: {:?}",
        errors.len(),
        errors.iter().map(|(strategy, _)| strategy).collect::<Vec<_>>()
    );

    Err(TokenizerErrorHandler::config_error(error_summary))
}
```

**Issues**:
1. **Download Strategy Actually Implemented**: Code shows download is attempted, but `infer_download_source()` returns `Ok(None)`
2. **Root Cause**: `TokenizerDiscovery::infer_download_source()` has incomplete compatibility matrix
3. **Missing Model Coverage**: Only covers LLaMA-2/3 and GPT-2, missing many common models
4. **No Generic Fallback**: Doesn't attempt generic download based on detected architecture

### Technical Approach

**Implementation Strategy**:
```rust
// crates/bitnet-tokenizers/src/discovery.rs

/// Comprehensive model compatibility matrix for tokenizer downloads
pub struct ModelCompatibilityMatrix {
    /// LLaMA family
    pub llama3_128k: TokenizerDownloadInfo,
    pub llama2_32k: TokenizerDownloadInfo,
    pub codellama_32k: TokenizerDownloadInfo,

    /// GPT family
    pub gpt2_50k: TokenizerDownloadInfo,
    pub gpt_neo_50k: TokenizerDownloadInfo,
    pub gpt_j_50k: TokenizerDownloadInfo,
    pub gpt_neox_50k: TokenizerDownloadInfo,

    /// BERT family
    pub bert_30k: TokenizerDownloadInfo,
    pub roberta_50k: TokenizerDownloadInfo,

    /// T5 family
    pub t5_32k: TokenizerDownloadInfo,
    pub mt5_250k: TokenizerDownloadInfo,

    /// BitNet family
    pub bitnet_custom: TokenizerDownloadInfo,
}

impl Default for ModelCompatibilityMatrix {
    fn default() -> Self {
        Self {
            // LLaMA family
            llama3_128k: TokenizerDownloadInfo {
                repo: "meta-llama/Meta-Llama-3-8B".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "llama3-128k".to_string(),
                expected_vocab: Some(128256),
            },
            llama2_32k: TokenizerDownloadInfo {
                repo: "meta-llama/Llama-2-7b-hf".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "llama2-32k".to_string(),
                expected_vocab: Some(32000),
            },
            codellama_32k: TokenizerDownloadInfo {
                repo: "codellama/CodeLlama-7b-hf".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "codellama-32k".to_string(),
                expected_vocab: Some(32016),
            },

            // GPT family
            gpt2_50k: TokenizerDownloadInfo {
                repo: "openai-community/gpt2".to_string(),
                files: vec!["tokenizer.json".to_string(), "merges.txt".to_string(), "vocab.json".to_string()],
                cache_key: "gpt2-50k".to_string(),
                expected_vocab: Some(50257),
            },
            gpt_neo_50k: TokenizerDownloadInfo {
                repo: "EleutherAI/gpt-neo-125M".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "gpt-neo-50k".to_string(),
                expected_vocab: Some(50257),
            },
            gpt_j_50k: TokenizerDownloadInfo {
                repo: "EleutherAI/gpt-j-6B".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "gpt-j-50k".to_string(),
                expected_vocab: Some(50400),
            },
            gpt_neox_50k: TokenizerDownloadInfo {
                repo: "EleutherAI/gpt-neox-20b".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "gpt-neox-50k".to_string(),
                expected_vocab: Some(50432),
            },

            // BERT family
            bert_30k: TokenizerDownloadInfo {
                repo: "google-bert/bert-base-uncased".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "bert-30k".to_string(),
                expected_vocab: Some(30522),
            },
            roberta_50k: TokenizerDownloadInfo {
                repo: "FacebookAI/roberta-base".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "roberta-50k".to_string(),
                expected_vocab: Some(50265),
            },

            // T5 family
            t5_32k: TokenizerDownloadInfo {
                repo: "google-t5/t5-base".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "t5-32k".to_string(),
                expected_vocab: Some(32128),
            },
            mt5_250k: TokenizerDownloadInfo {
                repo: "google/mt5-base".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "mt5-250k".to_string(),
                expected_vocab: Some(250000),
            },

            // BitNet family
            bitnet_custom: TokenizerDownloadInfo {
                repo: "1bitLLM/bitnet_b1_58-large".to_string(),
                files: vec!["tokenizer.json".to_string(), "tokenizer.model".to_string()],
                cache_key: "bitnet-custom".to_string(),
                expected_vocab: None, // Variable
            },
        }
    }
}

/// Enhanced inference of download source with comprehensive model coverage
pub fn infer_download_source(&self) -> Result<Option<TokenizerDownloadInfo>> {
    // Exact match strategy: (model_type, vocab_size) → download_info
    let exact_match = match (self.model_type.as_str(), self.vocab_size) {
        // LLaMA family
        ("llama", 128256) | ("llama3", 128256) => Some(self.compatibility_matrix.llama3_128k.clone()),
        ("llama", 32000) | ("llama2", 32000) => Some(self.compatibility_matrix.llama2_32k.clone()),
        ("codellama", 32016) => Some(self.compatibility_matrix.codellama_32k.clone()),

        // GPT family
        ("gpt2", 50257) => Some(self.compatibility_matrix.gpt2_50k.clone()),
        ("gpt-neo", 50257) => Some(self.compatibility_matrix.gpt_neo_50k.clone()),
        ("gpt-j", 50400) => Some(self.compatibility_matrix.gpt_j_50k.clone()),
        ("gpt-neox", 50432) => Some(self.compatibility_matrix.gpt_neox_50k.clone()),

        // BERT family
        ("bert", 30522) => Some(self.compatibility_matrix.bert_30k.clone()),
        ("roberta", 50265) => Some(self.compatibility_matrix.roberta_50k.clone()),

        // T5 family
        ("t5", 32128) => Some(self.compatibility_matrix.t5_32k.clone()),
        ("mt5", 250000) => Some(self.compatibility_matrix.mt5_250k.clone()),

        // BitNet family (use LLaMA-2 tokenizer as default)
        ("bitnet", _) => Some(self.compatibility_matrix.bitnet_custom.clone()),

        _ => None,
    };

    if exact_match.is_some() {
        return Ok(exact_match);
    }

    // Fuzzy match strategy: Use architecture-specific default if vocab_size is close
    let fuzzy_match = match self.model_type.as_str() {
        "llama" | "llama2" | "llama3" => {
            // LLaMA-3 if vocab > 100K, else LLaMA-2
            if self.vocab_size > 100000 {
                Some(self.compatibility_matrix.llama3_128k.clone())
            } else {
                Some(self.compatibility_matrix.llama2_32k.clone())
            }
        }

        "gpt2" | "gpt" => Some(self.compatibility_matrix.gpt2_50k.clone()),
        "gpt-neo" => Some(self.compatibility_matrix.gpt_neo_50k.clone()),
        "gpt-j" => Some(self.compatibility_matrix.gpt_j_50k.clone()),
        "gpt-neox" => Some(self.compatibility_matrix.gpt_neox_50k.clone()),

        "bert" => Some(self.compatibility_matrix.bert_30k.clone()),
        "roberta" => Some(self.compatibility_matrix.roberta_50k.clone()),

        "t5" => Some(self.compatibility_matrix.t5_32k.clone()),
        "mt5" => Some(self.compatibility_matrix.mt5_250k.clone()),

        "bitnet" => Some(self.compatibility_matrix.bitnet_custom.clone()),

        _ => None,
    };

    if fuzzy_match.is_some() {
        warn!(
            "Using fuzzy match for {}:{} tokenizer download (exact vocab size not in compatibility matrix)",
            self.model_type, self.vocab_size
        );
        return Ok(fuzzy_match);
    }

    // Final fallback: Generic transformer uses GPT-2 tokenizer if vocab size is close
    if self.model_type == "transformer" && (self.vocab_size >= 50000 && self.vocab_size <= 51000) {
        warn!(
            "Using GPT-2 tokenizer as generic fallback for transformer model with vocab_size {}",
            self.vocab_size
        );
        return Ok(Some(self.compatibility_matrix.gpt2_50k.clone()));
    }

    debug!("No compatible tokenizer download source found for {}:{}", self.model_type, self.vocab_size);
    Ok(None)
}
```

**Dependencies**:
- **Existing**: `crates/bitnet-tokenizers/src/download.rs::SmartTokenizerDownload` - Download implementation (working)
- **Enhanced**: Comprehensive model compatibility matrix with expanded architecture coverage

**Risk Assessment**: **Low Risk**
- **Download Infrastructure Exists**: `SmartTokenizerDownload` is fully implemented and working
- **Conservative Fallback**: Fuzzy matching only for well-known architectures
- **Offline Mode Respected**: Downloads only attempted when `BITNET_OFFLINE` is not set

**Validation Commands**:
```bash
# Test exact match download inference
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_infer_download_exact_match

# Test fuzzy match download inference
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_infer_download_fuzzy_match

# Test download integration (requires network)
cargo test --no-default-features --features cpu,downloads -p bitnet-tokenizers test_download_tokenizer_integration

# Test offline mode behavior
BITNET_OFFLINE=1 cargo test --no-default-features --features cpu -p bitnet-tokenizers test_offline_mode_no_download

# Integration test with real model
cargo run -p bitnet-cli --no-default-features --features cpu,downloads -- infer --model test-models/llama3-no-tokenizer.gguf --prompt "Test"
```

---

## Problem Area #5: Fallback Chain Integration

### Current Implementation Analysis

**Location**: `crates/bitnet-tokenizers/src/strategy.rs:18-35`

**Current Behavior**:
```rust
pub struct TokenizerStrategyResolver {
    discovery: TokenizerDiscovery,
    downloader: SmartTokenizerDownload,
    _fallback_chain: TokenizerFallbackChain, // ❌ PROBLEM: Unused field with _ prefix
}

impl TokenizerStrategyResolver {
    pub async fn new(discovery: TokenizerDiscovery) -> Result<Self> {
        info!("Initializing TokenizerStrategyResolver for {} model", discovery.model_type());

        let downloader = SmartTokenizerDownload::new()?;
        let fallback_chain = TokenizerFallbackChain::new(); // Created but never used

        Ok(Self {
            discovery,
            downloader,
            _fallback_chain: fallback_chain // ❌ Prefixed with _ to suppress unused warning
        })
    }

    // ❌ PROBLEM: resolve_with_fallback() implements its own ad-hoc fallback logic
    //            instead of using _fallback_chain field
}
```

**Issues**:
1. **Unused Fallback Chain**: `TokenizerFallbackChain` created but never utilized
2. **Ad-Hoc Fallback Logic**: `resolve_with_fallback()` implements custom fallback instead of using systematic chain
3. **No Systematic Prioritization**: Fallback strategies hardcoded in method instead of configurable chain
4. **Duplicate Logic**: Both `TokenizerFallbackChain` and `resolve_with_fallback()` implement fallback logic

### Technical Approach

**Implementation Strategy**:

**Option A: Use Existing Fallback Chain (Recommended)**
```rust
// crates/bitnet-tokenizers/src/strategy.rs

pub struct TokenizerStrategyResolver {
    discovery: TokenizerDiscovery,
    downloader: SmartTokenizerDownload,
    fallback_chain: TokenizerFallbackChain, // ✅ Remove _ prefix - use the field
}

impl TokenizerStrategyResolver {
    pub async fn new(discovery: TokenizerDiscovery) -> Result<Self> {
        info!("Initializing TokenizerStrategyResolver for {} model", discovery.model_type());

        let downloader = SmartTokenizerDownload::new()?;
        let fallback_chain = TokenizerFallbackChain::new();

        Ok(Self {
            discovery,
            downloader,
            fallback_chain // ✅ Use without _ prefix
        })
    }

    /// Resolve tokenizer using systematic fallback chain
    pub async fn resolve_with_fallback(&self) -> Result<Arc<dyn Tokenizer>> {
        info!("Resolving tokenizer using systematic fallback chain");

        // Delegate to fallback chain for strategy resolution
        let resolution = self.fallback_chain.resolve_tokenizer(&self.discovery).await?;

        // Convert resolution to concrete tokenizer
        match resolution {
            TokenizerResolution::File(path) => {
                self.load_tokenizer_from_path(&path)
            }
            TokenizerResolution::Embedded(tokenizer) => {
                self.configure_model_specific_wrapper(tokenizer)
            }
            TokenizerResolution::Download(download_info) => {
                // Use downloader to fetch tokenizer
                let downloaded_path = self.downloader.download_tokenizer(&download_info).await?;
                self.load_tokenizer_from_path(&downloaded_path)
            }
            TokenizerResolution::Mock(mock_tokenizer) => {
                if std::env::var("BITNET_STRICT_TOKENIZERS").is_ok() {
                    return Err(BitNetError::Config(
                        "Mock tokenizers not allowed in strict mode".to_string()
                    ));
                }
                self.configure_model_specific_wrapper(Arc::new(mock_tokenizer))
            }
        }
    }
}

// Enhanced TokenizerFallbackChain to include download strategy
// crates/bitnet-tokenizers/src/fallback.rs

pub struct TokenizerFallbackChain {
    strategies: Vec<FallbackStrategy>,
    strict_mode: bool,
}

impl TokenizerFallbackChain {
    pub fn new() -> Self {
        let strict_mode = std::env::var("BITNET_STRICT_TOKENIZERS").is_ok();

        let strategies = if strict_mode {
            vec![
                FallbackStrategy::GgufMetadata,
                FallbackStrategy::ColocatedFiles,
                FallbackStrategy::StandardCache,
                FallbackStrategy::SmartDownload, // ✅ Include download in chain
            ]
        } else {
            vec![
                FallbackStrategy::GgufMetadata,
                FallbackStrategy::ColocatedFiles,
                FallbackStrategy::StandardCache,
                FallbackStrategy::SmartDownload, // ✅ Include download in chain
                FallbackStrategy::MockFallback,
            ]
        };

        Self { strategies, strict_mode }
    }

    pub async fn resolve_tokenizer(
        &self,
        discovery: &TokenizerDiscovery,
    ) -> Result<TokenizerResolution> {
        info!("Resolving tokenizer using {} fallback strategies", self.strategies.len());

        for (i, strategy) in self.strategies.iter().enumerate() {
            debug!("Trying fallback strategy {}/{}: {:?}", i + 1, self.strategies.len(), strategy);

            let result: Result<Option<TokenizerResolution>> = match strategy {
                FallbackStrategy::GgufMetadata => {
                    if let Ok(Some(embedded)) = discovery.try_extract_embedded_tokenizer() {
                        Ok(Some(TokenizerResolution::Embedded(embedded)))
                    } else {
                        Ok(None)
                    }
                }
                FallbackStrategy::ColocatedFiles => {
                    if let Ok(Some(path)) = discovery.check_colocated_tokenizers() {
                        Ok(Some(TokenizerResolution::File(path)))
                    } else {
                        Ok(None)
                    }
                }
                FallbackStrategy::StandardCache => {
                    if let Ok(Some(path)) = discovery.check_cache_locations() {
                        Ok(Some(TokenizerResolution::File(path)))
                    } else {
                        Ok(None)
                    }
                }
                FallbackStrategy::SmartDownload => {
                    // ✅ ENHANCEMENT: Actually attempt download
                    if std::env::var("BITNET_OFFLINE").is_ok() {
                        debug!("Skipping smart download (offline mode)");
                        Ok(None)
                    } else if let Ok(Some(download_info)) = discovery.infer_download_source() {
                        Ok(Some(TokenizerResolution::Download(download_info)))
                    } else {
                        debug!("No download source available for this model");
                        Ok(None)
                    }
                }
                FallbackStrategy::MockFallback => {
                    if !self.strict_mode {
                        Ok(Some(TokenizerResolution::Mock(crate::MockTokenizer::new())))
                    } else {
                        Ok(None)
                    }
                }
            };

            match result {
                Ok(Some(resolution)) => {
                    info!("Fallback strategy {:?} succeeded", strategy);
                    return Ok(resolution);
                }
                Ok(None) => {
                    debug!("Fallback strategy {:?} returned no result", strategy);
                    continue;
                }
                Err(e) => {
                    warn!("Fallback strategy {:?} failed: {}", strategy, e);
                    continue;
                }
            }
        }

        if self.strict_mode {
            Err(BitNetError::Config("No tokenizer found and strict mode is enabled".to_string()))
        } else {
            // Final fallback to mock
            info!("All systematic strategies failed, using mock tokenizer");
            Ok(TokenizerResolution::Mock(crate::MockTokenizer::new()))
        }
    }
}

/// Enhanced tokenizer resolution result types
pub enum TokenizerResolution {
    File(PathBuf),
    Embedded(Arc<dyn Tokenizer>),
    Download(TokenizerDownloadInfo), // ✅ Add download resolution type
    Mock(crate::MockTokenizer),
}

#[derive(Debug)]
enum FallbackStrategy {
    GgufMetadata,
    ColocatedFiles,
    StandardCache,
    SmartDownload, // ✅ Add download strategy
    MockFallback,
}
```

**Option B: Remove Unused Fallback Chain (Alternative)**
```rust
// If fallback chain infrastructure is deemed unnecessary, remove it entirely

pub struct TokenizerStrategyResolver {
    discovery: TokenizerDiscovery,
    downloader: SmartTokenizerDownload,
    // ✅ Remove fallback_chain field entirely
}

impl TokenizerStrategyResolver {
    pub async fn new(discovery: TokenizerDiscovery) -> Result<Self> {
        let downloader = SmartTokenizerDownload::new()?;
        Ok(Self { discovery, downloader })
    }

    // Keep existing resolve_with_fallback() implementation
    // Document that this IS the fallback chain implementation
}
```

**Recommendation**: **Option A (Use Existing Fallback Chain)**
- Maintains separation of concerns between strategy resolution and fallback logic
- Provides systematic, testable fallback chain
- Enables future extensibility (custom strategies, priority configuration)
- Reduces code duplication between fallback.rs and strategy.rs

**Dependencies**:
- **Existing**: `crates/bitnet-tokenizers/src/fallback.rs::TokenizerFallbackChain` - Fallback infrastructure
- **Enhanced**: Add `SmartDownload` strategy and `Download` resolution type

**Risk Assessment**: **Low Risk**
- **Refactoring Only**: No new functionality, just uses existing infrastructure
- **Backward Compatible**: Same fallback behavior, just reorganized
- **Well-Tested**: Fallback chain infrastructure already has comprehensive tests

**Validation Commands**:
```bash
# Test systematic fallback chain execution
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_fallback_chain_execution

# Test fallback chain with all strategies
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_fallback_chain_comprehensive

# Test strict mode fallback behavior
BITNET_STRICT_TOKENIZERS=1 cargo test --no-default-features --features cpu -p bitnet-tokenizers test_fallback_strict_mode

# Integration test with systematic fallback
cargo run -p bitnet-cli --no-default-features --features cpu,downloads -- infer --model test-models/no-tokenizer.gguf --prompt "Test"
```

---

## Cross-Cutting Concerns

### Feature Flag Architecture

**Required Feature Flags**:
```toml
# crates/bitnet-tokenizers/Cargo.toml

[features]
default = [] # Empty defaults per BitNet.rs standard

# Core backends
cpu = ["bitnet-common/cpu"]
gpu = ["bitnet-common/gpu", "cuda"]

# Tokenizer backends
hf = ["tokenizers"] # HuggingFace tokenizers library
spm = ["sentencepiece"] # SentencePiece support (feature-gated)

# Download support
downloads = ["reqwest", "futures-util"]

# Full feature set
full = ["cpu", "hf", "spm", "downloads"]
```

**Build Commands**:
```bash
# CPU-only build (no SentencePiece)
cargo build --no-default-features --features cpu,hf

# CPU with SentencePiece support
cargo build --no-default-features --features cpu,hf,spm

# GPU with downloads
cargo build --no-default-features --features gpu,hf,downloads

# Full feature set
cargo build --no-default-features --features full
```

### Error Handling Strategy

**Consistent Error Patterns**:
```rust
// Use centralized error handling with actionable suggestions

// Example: Embedded tokenizer extraction failure
if let Err(e) = self.parse_hf_tokenizer_from_json(&tokenizer_json) {
    warn!("Failed to parse embedded tokenizer.json: {}", e);
    TokenizerErrorHandler::log_error(&e, "embedded_tokenizer_extraction");
    // Continue to next strategy instead of failing
}

// Example: Download failure with actionable error
match self.downloader.download_tokenizer(&download_info).await {
    Ok(path) => Ok(path),
    Err(e) => {
        TokenizerErrorHandler::create_actionable_error(
            e,
            &format!("downloading tokenizer from {}", download_info.repo)
        )
    }
}
```

**Logging Strategy**:
- **debug!**: Detailed progress (strategy attempts, metadata parsing)
- **info!**: Successful operations (tokenizer found, download complete)
- **warn!**: Fallback usage, fuzzy matching, inferred values
- **error!**: Critical failures requiring user intervention

### Performance Considerations

**Zero-Copy GGUF Parsing**:
```rust
// Embedded tokenizer extraction should avoid unnecessary allocations

// ✅ Good: Parse JSON directly from GGUF string metadata
if let Some(tokenizer_json) = self.gguf_reader.get_string_metadata("tokenizer.json") {
    // Parse directly from string slice
    match serde_json::from_str::<Value>(tokenizer_json) {
        Ok(json) => { /* create tokenizer */ }
        Err(e) => { /* fallback */ }
    }
}

// ❌ Bad: Copy bytes and convert to string
// let bytes = self.gguf_reader.get_array_metadata("tokenizer.json")?;
// let string = String::from_utf8(bytes.to_vec())?; // Unnecessary allocation
```

**Download Caching**:
- Use persistent cache directory (`~/.cache/bitnet/tokenizers/`)
- Check cache before network requests
- Validate cached files against expected vocabulary size
- Resume interrupted downloads using HTTP range requests

**Memory Efficiency**:
- Embedded tokenizers: Share GGUF mmap with model loading (zero-copy)
- Downloaded tokenizers: Load once, cache in memory with Arc<dyn Tokenizer>
- Vocabulary: Use efficient data structures (HashMap for BPE, Vec for WordPiece)

---

## Testing Strategy

### Unit Tests (Per Problem Area)

**Problem Area #1: Embedded Tokenizer Extraction**
```bash
# Test HuggingFace tokenizer extraction from embedded JSON
cargo test --no-default-features --features cpu,hf -p bitnet-tokenizers test_extract_hf_from_json

# Test SentencePiece tokenizer extraction from embedded bytes
cargo test --no-default-features --features cpu,spm -p bitnet-tokenizers test_extract_spm_from_bytes

# Test BPE tokenizer creation from vocabulary and merges
cargo test --no-default-features --features cpu,hf -p bitnet-tokenizers test_create_bpe_from_vocab

# Test fallback to BasicTokenizer when parsing fails
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_embedded_fallback_basic
```

**Problem Area #2: Model Type Detection**
```bash
# Test architecture detection from metadata
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_detect_arch_metadata

# Test architecture detection from tensor patterns
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_detect_arch_tensors

# Test architecture normalization
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_normalize_architecture

# Test edge cases (unknown architectures)
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_detect_unknown_arch
```

**Problem Area #3: Vocabulary Size Fallback**
```bash
# Test vocabulary size extraction from metadata
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_vocab_from_metadata

# Test vocabulary size extraction from tensors
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_vocab_from_tensors

# Test architecture-specific defaults
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_vocab_defaults

# Test model name inference
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_vocab_from_name
```

**Problem Area #4: Smart Download Strategy**
```bash
# Test exact match download inference
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_download_exact_match

# Test fuzzy match download inference
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_download_fuzzy_match

# Test offline mode behavior
BITNET_OFFLINE=1 cargo test --no-default-features --features cpu -p bitnet-tokenizers test_offline_no_download

# Test download integration (requires network)
cargo test --no-default-features --features cpu,downloads -p bitnet-tokenizers test_download_integration -- --ignored
```

**Problem Area #5: Fallback Chain Integration**
```bash
# Test systematic fallback chain execution
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_fallback_systematic

# Test fallback chain with download strategy
cargo test --no-default-features --features cpu,downloads -p bitnet-tokenizers test_fallback_with_download

# Test strict mode behavior
BITNET_STRICT_TOKENIZERS=1 cargo test --no-default-features --features cpu -p bitnet-tokenizers test_strict_mode
```

### Integration Tests

**End-to-End Tokenizer Discovery**:
```bash
# Test with real GGUF files containing embedded tokenizers
cargo test --no-default-features --features cpu,hf,spm -p bitnet-tokenizers test_discovery_e2e -- --ignored

# Test with GGUF files requiring download
cargo test --no-default-features --features cpu,downloads -p bitnet-tokenizers test_discovery_download_e2e -- --ignored

# Test with incomplete GGUF files (fallback scenarios)
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_discovery_fallback_e2e -- --ignored
```

**CLI Integration**:
```bash
# Test automatic tokenizer discovery with inference
cargo run -p bitnet-cli --no-default-features --features cpu,hf -- infer --model test-models/llama3-embedded.gguf --prompt "Hello world"

# Test smart download integration
cargo run -p bitnet-cli --no-default-features --features cpu,downloads -- infer --model test-models/llama2-no-tokenizer.gguf --prompt "Hello world"

# Test strict mode enforcement
BITNET_STRICT_TOKENIZERS=1 cargo run -p bitnet-cli --no-default-features --features cpu -- infer --model test-models/unknown-model.gguf --prompt "Test"
```

### Cross-Validation Tests

**Microsoft BitNet C++ Reference Parity**:
```bash
# Validate tokenizer compatibility with BitNet C++ reference
cargo run -p xtask --no-default-features --features crossval -- verify-tokenizer test-models/bitnet-b1.58-2B.gguf

# Cross-validate token IDs for standard prompts
cargo test --no-default-features --features crossval -p bitnet-tokenizers test_token_parity_cpp_reference

# Validate vocabulary size consistency
cargo test --no-default-features --features crossval -p bitnet-tokenizers test_vocab_size_parity
```

---

## Risk Assessment and Mitigation

### Problem Area #1 Risks: Embedded Tokenizer Extraction

**Risk**: **Medium** - Complex JSON parsing and binary format handling

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| HuggingFace JSON parsing failures | Medium | Medium | Graceful fallback to next strategy, comprehensive error logging |
| SentencePiece protobuf corruption | Medium | Low | Binary format validation before parsing, fallback to BasicTokenizer |
| Memory allocation for large vocabularies | Low | Low | Zero-copy parsing where possible, streaming for large JSON |
| Backward compatibility with BasicTokenizer | Low | Low | Maintain BasicTokenizer fallback for unsupported formats |

**Mitigation Commands**:
```bash
# Test with corrupted embedded JSON
cargo test --no-default-features --features cpu,hf -p bitnet-tokenizers test_corrupted_embedded_json

# Test with large vocabulary models (LLaMA-3: 128K)
cargo test --no-default-features --features cpu,hf -p bitnet-tokenizers test_large_vocab_embedded

# Validate memory usage with embedded tokenizers
cargo run -p bitnet-cli --no-default-features --features cpu,hf -- infer --model test-models/llama3-large-vocab.gguf --prompt "Test" --max-memory-mb 500
```

### Problem Area #2 Risks: Model Type Detection

**Risk**: **Low** - Deterministic pattern matching

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| False positive architecture detection | Low | Low | Require minimum pattern score (>= 7 for LLaMA, >= 6 for BERT) |
| New architecture not in pattern database | Low | Medium | Graceful fallback to "transformer", periodic pattern updates |
| Tensor naming variations across quantization | Low | Low | Normalized pattern matching (lowercase, contains() checks) |

**Mitigation Commands**:
```bash
# Test edge cases with minimal tensor sets
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_minimal_tensors

# Test with unknown architectures
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_unknown_architecture_fallback
```

### Problem Area #3 Risks: Vocabulary Size Fallback

**Risk**: **Low** - Well-defined defaults for known architectures

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Incorrect default vocabulary size | Medium | Low | Conservative defaults, clear warnings when using inferred values |
| Vocabulary size mismatch with model | Low | Low | Validation against tensor shapes, warning logging |
| Performance impact of incorrect vocab size | Low | Low | Early validation, fail-fast on obvious mismatches |

**Mitigation Commands**:
```bash
# Test vocabulary size validation
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_vocab_size_validation

# Test mismatch detection
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_vocab_mismatch_warning
```

### Problem Area #4 Risks: Smart Download Strategy

**Risk**: **Low** - Existing download infrastructure works

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Network failures during download | Low | Medium | Resume capability, retry with exponential backoff, offline mode support |
| HuggingFace Hub rate limiting | Low | Low | Persistent caching, retry after delay, user-friendly error messages |
| Incorrect tokenizer for model architecture | Medium | Low | Fuzzy matching validation, expected_vocab verification, fallback chain |
| Disk space exhaustion during download | Low | Low | Pre-download space check, cleanup on failure, cache size limits |

**Mitigation Commands**:
```bash
# Test download resume capability
cargo test --no-default-features --features cpu,downloads -p bitnet-tokenizers test_download_resume -- --ignored

# Test network failure handling
cargo test --no-default-features --features cpu,downloads -p bitnet-tokenizers test_download_network_failure -- --ignored

# Test offline mode behavior
BITNET_OFFLINE=1 cargo test --no-default-features --features cpu -p bitnet-tokenizers test_offline_mode_strict
```

### Problem Area #5 Risks: Fallback Chain Integration

**Risk**: **Low** - Refactoring existing code

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Regression in fallback behavior | Low | Low | Comprehensive integration tests, manual testing with various GGUF files |
| Performance degradation from systematic chain | Low | Low | Early exit on success, minimal overhead per strategy attempt |
| Strict mode enforcement issues | Low | Low | Explicit strict mode tests, clear documentation |

**Mitigation Commands**:
```bash
# Test fallback chain regression
cargo test --no-default-features --features cpu -p bitnet-tokenizers test_fallback_regression_suite

# Performance benchmark
cargo bench --no-default-features --features cpu -p bitnet-tokenizers bench_fallback_chain

# Strict mode validation
BITNET_STRICT_TOKENIZERS=1 cargo test --workspace --no-default-features --features cpu
```

---

## Implementation Roadmap

### Phase 1: Problem Area #2 & #3 (Low Risk, High Value)

**Timeline**: 1-2 days

**Tasks**:
1. Enhance `extract_model_type()` with comprehensive tensor pattern analysis
2. Implement architecture normalization and name-based inference
3. Enhance `extract_vocab_size()` with architecture-specific defaults
4. Add model name-based vocabulary inference
5. Write unit tests for all detection paths

**Success Criteria**:
- All test cases pass for LLaMA, GPT-2, GPT-Neo, BERT, T5 detection
- Vocabulary size defaults cover 95% of common architectures
- No regression in existing metadata-based detection

### Phase 2: Problem Area #4 (Low Risk, High Impact)

**Timeline**: 1 day

**Tasks**:
1. Expand `ModelCompatibilityMatrix` with all common architectures
2. Implement exact match and fuzzy match download inference
3. Add offline mode and network failure handling
4. Write integration tests with real downloads (--ignored)

**Success Criteria**:
- Download inference covers LLaMA, GPT, BERT, T5 families
- Fuzzy matching works for close vocabulary sizes
- Offline mode gracefully falls back to cache/mock

### Phase 3: Problem Area #1 (Medium Risk, High Complexity)

**Timeline**: 2-3 days

**Tasks**:
1. Implement `parse_hf_tokenizer_from_json()` for HuggingFace JSON parsing
2. Implement `parse_spm_tokenizer_from_bytes()` for SentencePiece models (feature-gated)
3. Implement `create_bpe_tokenizer_from_vocab()` for BPE construction
4. Add graceful fallback to BasicTokenizer on parsing failures
5. Write comprehensive unit and integration tests

**Success Criteria**:
- HuggingFace tokenizer.json parsing works for common formats
- SentencePiece model parsing works (behind spm feature flag)
- BPE tokenizer construction works from vocab + merges
- Graceful fallback to BasicTokenizer on any parsing failure

### Phase 4: Problem Area #5 (Low Risk, Code Cleanup)

**Timeline**: 0.5-1 day

**Tasks**:
1. Remove `_` prefix from `_fallback_chain` field
2. Refactor `resolve_with_fallback()` to use `TokenizerFallbackChain`
3. Add `SmartDownload` strategy to fallback chain
4. Add `Download` variant to `TokenizerResolution` enum
5. Update tests to use systematic fallback chain

**Success Criteria**:
- Fallback chain field is actively used
- No ad-hoc fallback logic in `resolve_with_fallback()`
- All integration tests pass with systematic chain

### Phase 5: Integration and Documentation

**Timeline**: 1 day

**Tasks**:
1. End-to-end integration testing with real GGUF files
2. Cross-validation against Microsoft BitNet C++ reference
3. Update documentation (CLAUDE.md, getting-started.md)
4. Performance benchmarking and optimization
5. Final code review and cleanup

**Success Criteria**:
- All acceptance criteria met (AC1-AC5)
- >99% compatibility with Microsoft BitNet C++ reference
- Comprehensive documentation updated
- No performance regressions

**Total Timeline**: 5.5-7.5 days

---

## Success Criteria Validation

### AC1: Embedded Tokenizer Support

**Validation**:
```bash
# Test with GGUF file containing embedded HuggingFace tokenizer.json
cargo run -p bitnet-cli --no-default-features --features cpu,hf -- infer \
  --model test-models/llama3-embedded-hf.gguf \
  --prompt "Hello, world!" \
  --max-tokens 10

# Verify correct tokenizer type loaded (should be HfTokenizer, not BasicTokenizer)
cargo test --no-default-features --features cpu,hf -p bitnet-tokenizers \
  test_embedded_hf_tokenizer_type

# Test with GGUF file containing embedded SentencePiece model
cargo run -p bitnet-cli --no-default-features --features cpu,spm -- infer \
  --model test-models/llama2-embedded-spm.gguf \
  --prompt "Hello, world!" \
  --max-tokens 10
```

**Expected Outcome**:
- HfTokenizer created from embedded tokenizer.json
- SpmTokenizer created from embedded SentencePiece model bytes
- No fallback to BasicTokenizer unless parsing fails

### AC2: Model Architecture Detection

**Validation**:
```bash
# Test architecture detection for various models
for model in llama3 llama2 gpt2 gpt-neo bert t5 bitnet; do
  cargo run -p bitnet-cli --no-default-features --features cpu -- compat-check \
    test-models/${model}-model.gguf | grep "Architecture:"
done

# Unit tests for architecture detection
cargo test --no-default-features --features cpu -p bitnet-tokenizers \
  test_detect_architecture_comprehensive
```

**Expected Outcome**:
- Correctly identifies: BitNet, LLaMA-2, LLaMA-3, GPT-2, GPT-Neo, BERT, T5
- Fallback to "transformer" for unknown architectures
- No false positives or incorrect classifications

### AC3: Vocabulary Size Resolution

**Validation**:
```bash
# Test vocabulary size extraction with various metadata completeness
cargo test --no-default-features --features cpu -p bitnet-tokenizers \
  test_vocab_size_extraction_complete \
  test_vocab_size_extraction_partial \
  test_vocab_size_extraction_minimal \
  test_vocab_size_defaults

# Integration test with incomplete GGUF files
cargo run -p bitnet-cli --no-default-features --features cpu -- compat-check \
  test-models/incomplete-metadata.gguf | grep "Vocabulary size:"
```

**Expected Outcome**:
- Extracts vocabulary size from metadata when available
- Falls back to tensor shapes when metadata missing
- Uses architecture-specific defaults as last resort
- Clear warnings when using inferred values

### AC4: Smart Download Integration

**Validation**:
```bash
# Test smart download for various architectures (requires network)
cargo run -p bitnet-cli --no-default-features --features cpu,downloads -- infer \
  --model test-models/llama2-no-tokenizer.gguf \
  --prompt "Test download" \
  --max-tokens 5

# Verify tokenizer cached after download
ls ~/.cache/bitnet/tokenizers/llama2-32k/tokenizer.json

# Test offline mode behavior
BITNET_OFFLINE=1 cargo run -p bitnet-cli --no-default-features --features cpu -- infer \
  --model test-models/llama2-no-tokenizer.gguf \
  --prompt "Test offline" \
  --max-tokens 5
```

**Expected Outcome**:
- Downloads compatible tokenizer from HuggingFace Hub
- Caches downloaded tokenizer persistently
- Reuses cached tokenizer on subsequent runs
- Respects BITNET_OFFLINE=1 environment variable

### AC5: Production Readiness & Cross-Validation

**Validation**:
```bash
# Cross-validate with Microsoft BitNet C++ reference
cargo run -p xtask --no-default-features --features crossval -- verify-tokenizer \
  test-models/bitnet-b1.58-2B.gguf

# Token ID parity test
cargo test --no-default-features --features crossval -p bitnet-tokenizers \
  test_token_id_parity_cpp_reference

# End-to-end inference parity test
cargo run -p xtask --no-default-features --features crossval -- crossval \
  --prompt "The quick brown fox jumps over the lazy dog" \
  --max-tokens 20
```

**Expected Outcome**:
- >99% token ID compatibility with Microsoft BitNet C++ reference
- Identical tokenization for standard prompts
- No regression in inference accuracy

---

## Routing Decision

**Flow Status**: **generative:gate:spec = pass**

**Analysis Complete**: Comprehensive technical specification created for all 5 problem areas with detailed implementation approaches, risk assessments, and validation strategies.

**Routing**: **FINALIZE → spec-finalizer**

**Evidence**:
- ✅ All 5 problem areas analyzed with technical approaches
- ✅ Dependencies identified and mapped to existing BitNet.rs infrastructure
- ✅ Risk assessment completed with specific mitigation strategies
- ✅ Validation commands provided for each problem area
- ✅ Implementation roadmap created with timeline estimates
- ✅ Success criteria defined with measurable validation procedures
- ✅ Cross-cutting concerns addressed (feature flags, error handling, performance)
- ✅ Testing strategy comprehensive (unit, integration, cross-validation)

**Next Steps (for spec-finalizer)**:
1. Review specification for completeness and accuracy
2. Validate technical approaches align with BitNet.rs architecture
3. Confirm feature flag strategy and build configurations
4. Verify testing strategy covers all acceptance criteria
5. Approve specification for implementation phase

---

## Appendix: GGUF Tokenizer Metadata Reference

### Standard GGUF Tokenizer Metadata Keys

**Vocabulary Metadata**:
```
tokenizer.ggml.vocab_size        # u32: Total vocabulary size
tokenizer.ggml.tokens            # string[]: Token strings
tokenizer.ggml.scores            # f32[]: Token scores (SentencePiece)
tokenizer.ggml.token_type        # u32[]: Token types (normal, control, etc.)
tokenizer.ggml.merges            # string[]: BPE merge rules
```

**Special Tokens**:
```
tokenizer.ggml.bos_token_id      # u32: Beginning-of-sequence token
tokenizer.ggml.eos_token_id      # u32: End-of-sequence token
tokenizer.ggml.pad_token_id      # u32: Padding token
tokenizer.ggml.unk_token_id      # u32: Unknown token
tokenizer.ggml.sep_token_id      # u32: Separator token (BERT)
tokenizer.ggml.cls_token_id      # u32: Classification token (BERT)
```

**Tokenizer Configuration**:
```
tokenizer.ggml.type              # string: "bpe", "wordpiece", "unigram", "sentencepiece"
tokenizer.ggml.model             # bytes: SentencePiece model (protobuf format)
tokenizer.json                   # string: HuggingFace tokenizer.json (embedded)
```

**Model Architecture Metadata**:
```
general.architecture             # string: "llama", "gpt2", "bert", "t5", etc.
general.name                     # string: Model name (e.g., "LLaMA-3-8B")
llama.vocab_size                 # u32: LLaMA-specific vocabulary size
gpt2.vocab_size                  # u32: GPT-2-specific vocabulary size
bert.vocab_size                  # u32: BERT-specific vocabulary size
```

### Example GGUF Tokenizer Extraction

**LLaMA-3 with Embedded HuggingFace Tokenizer**:
```rust
// Metadata keys present:
// - tokenizer.json: "<full HuggingFace tokenizer.json as string>"
// - general.architecture: "llama"
// - llama.vocab_size: 128256

let tokenizer_json = reader.get_string_metadata("tokenizer.json")?;
let hf_tokenizer = HfTokenizer::from_json_value(serde_json::from_str(&tokenizer_json)?)?;
// Returns: HfTokenizer with 128,256 vocabulary
```

**LLaMA-2 with Embedded SentencePiece Model**:
```rust
// Metadata keys present:
// - tokenizer.ggml.model: <SentencePiece protobuf bytes>
// - tokenizer.ggml.vocab_size: 32000
// - tokenizer.ggml.bos_token_id: 1
// - tokenizer.ggml.eos_token_id: 2

let spm_model_bytes = reader.get_array_metadata("tokenizer.ggml.model")?;
let spm_tokenizer = SpmTokenizer::from_bytes(&spm_model_bytes)?;
// Returns: SpmTokenizer with 32,000 vocabulary
```

**GPT-2 with Embedded BPE Vocabulary**:
```rust
// Metadata keys present:
// - tokenizer.ggml.tokens: ["!", "\"", "#", ..., "▁world"] (50,257 tokens)
// - tokenizer.ggml.merges: ["Ġ t", "Ġ a", "h e", ..., "Ġ world"] (merge rules)
// - tokenizer.ggml.type: "bpe"

let vocab = reader.get_string_array_metadata("tokenizer.ggml.tokens")?;
let merges = reader.get_string_array_metadata("tokenizer.ggml.merges")?;
let bpe_tokenizer = create_bpe_tokenizer_from_vocab(&vocab, &merges)?;
// Returns: HfTokenizer with BPE configuration
```
