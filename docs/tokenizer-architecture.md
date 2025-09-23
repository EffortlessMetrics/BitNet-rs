# Tokenizer Architecture Guide

Comprehensive guide to BitNet.rs tokenizer system, with focus on SentencePiece (SPM) workflow and test fixtures.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Universal Tokenizer Design](#universal-tokenizer-design)
3. [SentencePiece (SPM) Workflow](#sentencepiece-spm-workflow)
4. [Test Fixtures and Environment Variables](#test-fixtures-and-environment-variables)
5. [Contract Tests](#contract-tests)
6. [Developer Examples](#developer-examples)
7. [Troubleshooting](#troubleshooting)

## Architecture Overview

BitNet.rs implements a **Universal Tokenizer** architecture that auto-detects and handles multiple tokenizer formats:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Universal Tokenizer                         │
├─────────────────────────────────────────────────────────────────┤
│  Auto-Detection Layer                                          │
│  ├─ GGUF Metadata Parser                                       │
│  ├─ File Extension Detection (.json, .model)                  │
│  └─ Model Type Classification                                  │
├─────────────────────────────────────────────────────────────────┤
│  Backend Implementations                                       │
│  ├─ SentencePiece (SPM) [feature = "spm"]                     │
│  ├─ HuggingFace JSON (BPE, WordPiece)                         │
│  ├─ Mock Tokenizer (testing/fallback)                         │
│  └─ GGUF Embedded (future)                                    │
├─────────────────────────────────────────────────────────────────┤
│  Common Interface                                              │
│  ├─ encode(text, add_bos, add_special) -> Vec<u32>            │
│  ├─ decode(tokens) -> String                                  │
│  ├─ token_to_piece(token) -> String                           │
│  └─ vocab_size() -> usize                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Auto-Detection**: Automatically chooses appropriate backend based on file format and metadata
2. **Feature Gating**: SentencePiece support requires `--features spm` compilation flag
3. **Graceful Fallback**: Falls back to mock tokenizer for unsupported formats (unless strict mode)
4. **Strict Mode**: `BITNET_STRICT_TOKENIZERS=1` prevents mock fallbacks in CI/performance tests
5. **O(1) Performance**: Optimized implementations with pre-built lookup tables

## Universal Tokenizer Design

### Supported Tokenizer Types

| Model Type | Backend | Features | Build Requirements |
|------------|---------|----------|-------------------|
| `gpt2`, `bpe` | Mock* | BPE-style tokenization | Default |
| `llama`, `llama3` | Mock* | GPT-2 variant with different vocab | Default |
| `sentencepiece`, `smp` | SentencePiece | Real SPM tokenization | `--features spm` |
| `tiktoken`, `gpt4` | Mock* | OpenAI tokenization | Default |
| `falcon` | Mock* | Falcon-specific | Default |

*Mock tokenizer provides testing-compatible behavior but not real tokenization

### Configuration Structure

```rust
use bitnet_tokenizers::TokenizerConfig;

let config = TokenizerConfig {
    model_type: "sentencepiece".to_string(),
    vocab_size: 32000,
    pre_tokenizer: Some("/path/to/tokenizer.model".to_string()),
    add_bos: true,
    add_eos: false,
    add_space_prefix: false,
    byte_fallback: true,
    bos_token_id: Some(1),
    eos_token_id: Some(2),
    pad_token_id: Some(0),
    unk_token_id: Some(0),
    vocabulary: None,      // Extracted from GGUF metadata
    bpe_merges: None,      // For BPE tokenizers
};
```

## SentencePiece (SPM) Workflow

### End-to-End SPM Usage

#### 1. Build with SPM Support

```bash
# Build with SentencePiece support
cargo build --no-default-features --features "cpu,spm"

# Test SPM functionality
cargo test -p bitnet-tokenizers --features "spm,integration-tests"
```

#### 2. Create SPM Tokenizer from File

```rust
use bitnet_tokenizers::{SpmTokenizer, Tokenizer};
use std::path::Path;

// Load SentencePiece model directly
let spm_path = Path::new("models/tokenizer.model");
let tokenizer = SpmTokenizer::from_file(spm_path)?;

// Use the tokenizer
let tokens = tokenizer.encode("Hello world", true, false)?;
let text = tokenizer.decode(&tokens)?;
let piece = tokenizer.token_to_piece(1); // BOS token

println!("Tokens: {:?}", tokens);
println!("Decoded: {}", text);
println!("BOS piece: {:?}", piece);
```

#### 3. Create Universal Tokenizer with SPM Backend

```rust
use bitnet_tokenizers::{TokenizerConfig, UniversalTokenizer};

let config = TokenizerConfig {
    model_type: "sentencepiece".to_string(),
    vocab_size: 32000,
    pre_tokenizer: Some("models/tokenizer.model".to_string()), // Path to .model file
    add_bos: true,
    add_eos: false,
    bos_token_id: Some(1),
    eos_token_id: Some(2),
    ..Default::default()
};

let tokenizer = UniversalTokenizer::new(config)?;

// SPM tokenizer with BOS token handling
let tokens = tokenizer.encode("Test text", true, false)?;
assert_eq!(tokens[0], 1); // BOS token automatically added
```

#### 4. Load from GGUF Model with SPM Metadata

```rust
use bitnet_tokenizers::UniversalTokenizer;
use std::path::Path;

// Extract tokenizer configuration from GGUF model
let model_path = Path::new("models/bitnet-spm.gguf");
let tokenizer = UniversalTokenizer::from_gguf(model_path)?;

// Tokenizer automatically configured from GGUF metadata:
// - tokenizer.ggml.model = "sentencepiece"
// - vocab extracted from tokenizer.ggml.tokens
// - special tokens from metadata
```

### SPM Performance Optimizations

The SPM tokenizer includes several performance optimizations:

#### O(1) Token-to-Piece Lookup

```rust
// Pre-built lookup table for O(1) token_to_piece
pub struct SpmTokenizer {
    inner: sentencepiece::SentencePieceProcessor,
    id2piece: Box<[String]>, // Indexed by token ID
    // ...
}

impl Tokenizer for SpmTokenizer {
    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.id2piece.get(token as usize).cloned() // O(1) lookup
    }
}
```

#### Optimized Encoding/Decoding

```rust
// Efficient encoding with special token handling
fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
    let pieces = self.inner.encode(text)?;
    let mut ids: Vec<u32> = pieces.into_iter().map(|p| p.id).collect();

    // Add BOS if requested and not already present
    if add_bos && let Some(bos) = self.bos_id && ids.first().copied() != Some(bos) {
        ids.insert(0, bos);
    }

    // Add EOS if requested and not already present
    if add_special && let Some(eos) = self.eos_id && ids.last().copied() != Some(eos) {
        ids.push(eos);
    }

    Ok(ids)
}
```

## Test Fixtures and Environment Variables

### Environment Variables

| Variable | Purpose | Values | Usage |
|----------|---------|--------|-------|
| `BITNET_STRICT_TOKENIZERS` | Prevent mock fallbacks | `1` (strict), unset (normal) | CI/performance testing |
| `SPM_MODEL` | Path to test SPM model | File path | Integration tests |
| `BITNET_DETERMINISTIC` | Force deterministic behavior | `1` | Testing |

### Creating SPM Test Fixtures

#### 1. Minimal SPM Model for Testing

Create a minimal SentencePiece model for tests:

```bash
# Install SentencePiece tools
pip install sentencepiece

# Create training data
echo "hello world test example sentence piece tokenizer" > /tmp/train.txt

# Train minimal model
spm_train \
  --input=/tmp/train.txt \
  --model_prefix=tiny_spm \
  --vocab_size=100 \
  --model_type=bpe \
  --character_coverage=1.0 \
  --split_by_whitespace=true

# Results: tiny_spm.model and tiny_spm.vocab
```

#### 2. Test Fixture Integration

```rust
// tests/fixtures/spm/mod.rs
use std::path::PathBuf;

pub fn get_tiny_spm_model() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(path) = std::env::var("SPM_MODEL") {
        let p = PathBuf::from(path);
        if p.exists() { return Some(p); }
    }

    // Check standard fixture location
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/spm/tiny.model");

    if fixture_path.exists() {
        Some(fixture_path)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spm_fixture_availability() {
        if let Some(model_path) = get_tiny_spm_model() {
            assert!(model_path.exists(), "SPM fixture should exist");
            // Test actual loading
            let tokenizer = bitnet_tokenizers::SpmTokenizer::from_file(&model_path)
                .expect("Should load SPM fixture");
            assert!(tokenizer.vocab_size() > 0);
        } else {
            eprintln!("SPM fixture not available - set SPM_MODEL environment variable");
        }
    }
}
```

#### 3. Fixture Directory Structure

```
tests/fixtures/
├── spm/
│   ├── tiny.model          # Minimal SPM model (100 vocab)
│   ├── tiny.vocab          # Corresponding vocabulary
│   └── README.md           # Instructions for creating fixtures
├── minimal_tokenizer.json   # HF JSON fixture
└── broken/                 # Fixtures for error testing
    ├── corrupt.model
    └── invalid.json
```

### Environment-Driven Testing

#### Strict Mode Testing

```bash
# Normal mode - allows mock fallbacks
cargo test -p bitnet-tokenizers

# Strict mode - no mock fallbacks (CI)
BITNET_STRICT_TOKENIZERS=1 cargo test -p bitnet-tokenizers

# SPM integration tests with fixture
SPM_MODEL=tests/fixtures/spm/tiny.model \
cargo test -p bitnet-tokenizers --features "spm,integration-tests" -- --ignored
```

#### CI Integration Pattern

```bash
# CI workflow for tokenizer tests
if [ -f "tests/fixtures/spm/tiny.model" ]; then
    echo "Running full SPM tests with fixture"
    cargo test -p bitnet-tokenizers --features "spm,integration-tests"
else
    echo "Running tests without SPM fixture"
    BITNET_STRICT_TOKENIZERS=1 cargo test -p bitnet-tokenizers --features spm
fi
```

## Contract Tests

### Understanding Contract Tests

Contract tests ensure tokenizer implementations maintain compatibility guarantees:

```rust
/// Contract: GPT-2 tokenizers must handle space prefix correctly
#[test]
fn test_gpt2_tokenizer_contract() {
    let config = TokenizerConfig {
        model_type: "gpt2".to_string(),
        add_space_prefix: true,
        // ...
    };

    let tokenizer = UniversalTokenizer::new(config)?;

    // Contract: space prefix should be added automatically
    let tokens1 = tokenizer.encode("test", false, false)?;
    let tokens2 = tokenizer.encode(" test", false, false)?;
    // Implementation detail: tokens1 and tokens2 should be equivalent
    // due to automatic space prefix handling
}
```

### SPM Contract Tests

```rust
/// Contract: SentencePiece tokenizers must handle BOS/EOS correctly
#[test]
fn test_sentencepiece_tokenizer_contract() {
    // Skip if no SPM fixture available
    let model_path = match get_tiny_spm_model() {
        Some(path) => path,
        None => {
            eprintln!("Skipping SPM contract test - no fixture available");
            return;
        }
    };

    let config = TokenizerConfig {
        model_type: "sentencepiece".to_string(),
        pre_tokenizer: Some(model_path.to_string_lossy().to_string()),
        add_bos: true,
        bos_token_id: Some(1),
        // ...
    };

    let tokenizer = UniversalTokenizer::new(config)?;

    // Contract: BOS token must be added when requested
    let tokens = tokenizer.encode("test", true, false)?;
    assert_eq!(tokens[0], 1, "BOS token must be first");

    // Contract: round-trip must preserve meaning
    let decoded = tokenizer.decode(&tokens)?;
    assert!(!decoded.is_empty(), "Decoded text must not be empty");
}
```

### Test Execution Patterns

#### With Fixtures Available

```rust
#[test]
fn test_with_real_spm_model() {
    let model_path = get_tiny_spm_model().expect("SPM fixture required");

    let tokenizer = SpmTokenizer::from_file(&model_path)?;

    // Test real SPM behavior
    let tokens = tokenizer.encode("hello world", true, false)?;
    let decoded = tokenizer.decode(&tokens)?;

    // Verify round-trip accuracy
    assert!(decoded.contains("hello"));
    assert!(decoded.contains("world"));
}
```

#### Without Fixtures (Mock Fallback)

```rust
#[test]
fn test_without_spm_fixture() {
    // When SPM fixture unavailable, test mock behavior
    std::env::remove_var("SPM_MODEL"); // Ensure no fixture

    let config = TokenizerConfig {
        model_type: "sentencepiece".to_string(),
        pre_tokenizer: None, // No model path
        // ...
    };

    // Should fall back to mock tokenizer (unless strict mode)
    let tokenizer = UniversalTokenizer::new(config);

    if std::env::var("BITNET_STRICT_TOKENIZERS").is_ok() {
        assert!(tokenizer.is_err(), "Should fail in strict mode");
    } else {
        let tokenizer = tokenizer.expect("Should use mock fallback");
        // Test mock behavior is deterministic
        let a = tokenizer.encode("test", false, false)?;
        let b = tokenizer.encode("test", false, false)?;
        assert_eq!(a, b, "Mock tokenizer must be deterministic");
    }
}
```

## Developer Examples

### Quick Start: Universal Tokenizer

```rust
use bitnet_tokenizers::{TokenizerConfig, UniversalTokenizer};

fn main() -> anyhow::Result<()> {
    // Auto-detect tokenizer from GGUF model
    let tokenizer = UniversalTokenizer::from_gguf("model.gguf")?;

    // Or create with explicit configuration
    let config = TokenizerConfig {
        model_type: "sentencepiece".to_string(),
        vocab_size: 32000,
        pre_tokenizer: Some("tokenizer.model".to_string()),
        add_bos: true,
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        ..Default::default()
    };
    let tokenizer = UniversalTokenizer::new(config)?;

    // Use tokenizer
    let tokens = tokenizer.encode("Hello, world!", true, false)?;
    println!("Tokens: {:?}", tokens);

    let decoded = tokenizer.decode(&tokens)?;
    println!("Decoded: {}", decoded);

    Ok(())
}
```

### Loading Different Tokenizer Types

```rust
use bitnet_tokenizers::{TokenizerBuilder, from_path};
use std::path::Path;

// Load from file with auto-detection
let (tokenizer, kind) = from_path(Path::new("tokenizer.json"))?;
match kind {
    TokenizerFileKind::HfJson => println!("Loaded HuggingFace JSON tokenizer"),
    #[cfg(feature = "spm")]
    TokenizerFileKind::Spm => println!("Loaded SentencePiece tokenizer"),
}

// Or use builder pattern
let tokenizer = TokenizerBuilder::from_file("tokenizer.model")?;

// Or load pretrained (mock implementation)
let tokenizer = TokenizerBuilder::from_pretrained("gpt2")?;
```

### Testing with Environment Control

```rust
use temp_env::with_var;

#[test]
fn test_strict_mode_behavior() {
    // Test normal mode
    with_var("BITNET_STRICT_TOKENIZERS", None::<&str>, || {
        let config = TokenizerConfig {
            model_type: "unknown".to_string(),
            ..Default::default()
        };

        let tokenizer = UniversalTokenizer::new(config);
        assert!(tokenizer.is_ok(), "Should use mock fallback");
    });

    // Test strict mode
    with_var("BITNET_STRICT_TOKENIZERS", Some("1"), || {
        let config = TokenizerConfig {
            model_type: "unknown".to_string(),
            ..Default::default()
        };

        let tokenizer = UniversalTokenizer::new(config);
        assert!(tokenizer.is_err(), "Should reject mock fallback");
    });
}
```

### Integration with BitNet Models

```rust
use bitnet_tokenizers::UniversalTokenizer;
use bitnet_models::GgufModel;

fn load_model_with_tokenizer(model_path: &str) -> anyhow::Result<()> {
    // Load model
    let model = GgufModel::from_file(model_path)?;

    // Extract tokenizer from same file
    let tokenizer = UniversalTokenizer::from_gguf(model_path.as_ref())?;

    // Verify compatibility
    let model_vocab_size = model.config().vocab_size;
    let tokenizer_vocab_size = tokenizer.vocab_size();

    if model_vocab_size != tokenizer_vocab_size {
        eprintln!("Warning: vocab size mismatch");
        eprintln!("Model: {}, Tokenizer: {}", model_vocab_size, tokenizer_vocab_size);
    }

    // Use for inference
    let text = "The capital of France is";
    let tokens = tokenizer.encode(text, true, false)?;

    // Run inference with model...

    Ok(())
}
```

### Custom Tokenizer Implementation

```rust
use bitnet_tokenizers::{Tokenizer, TokenizerConfig};
use bitnet_common::Result;

pub struct CustomTokenizer {
    vocab_size: usize,
    // Add custom fields
}

impl Tokenizer for CustomTokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        // Custom tokenization logic
        let mut tokens = Vec::new();

        if add_bos {
            tokens.push(0); // BOS token
        }

        // Simple word-based tokenization
        for word in text.split_whitespace() {
            let token_id = word.len() as u32 % self.vocab_size as u32;
            tokens.push(token_id);
        }

        if add_special {
            tokens.push(1); // EOS token
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        Ok(format!("Decoded {} tokens", tokens.len()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        Some(format!("<{}>", token))
    }
}
```

## Troubleshooting

### Common Issues

#### 1. SPM Compilation Errors

```bash
# Error: sentencepiece crate not found
error[E0432]: unresolved import `sentencepiece`

# Solution: Build with SPM feature
cargo build --no-default-features --features "cpu,spm"
```

#### 2. Mock Tokenizer in Production

```bash
# Error: Using mock tokenizer in strict mode
Error: Mock tokenizer fallback disabled (BITNET_STRICT_TOKENIZERS=1)

# Solution: Provide real tokenizer or disable strict mode
unset BITNET_STRICT_TOKENIZERS
# OR
export SPM_MODEL="path/to/tokenizer.model"
```

#### 3. Vocab Size Mismatch

```rust
// Error: Model expects 128256 tokens, tokenizer has 50257
Warning: vocab size mismatch
Model: 128256, Tokenizer: 50257

// Solution: Verify tokenizer matches model
let config = TokenizerConfig {
    model_type: "llama3".to_string(), // Use correct model type
    vocab_size: 128256,               // Match model vocab size
    // ...
};
```

#### 4. Missing SPM Model File

```bash
# Error: Failed to load SentencePiece tokenizer: No such file
Failed to load SentencePiece tokenizer: /path/to/tokenizer.model

# Solution: Check file path and create fixture if needed
ls -la /path/to/tokenizer.model
# OR create test fixture
python -c "
import sentencepiece as spm
spm.SentencePieceTrainer.train('--input=sample.txt --model_prefix=test --vocab_size=100')
"
```

### Debug Commands

```bash
# Test tokenizer loading
cargo test -p bitnet-tokenizers --features "spm,integration-tests" test_sentencepiece_tokenizer_contract

# Verify environment setup
echo "SPM_MODEL: ${SPM_MODEL:-unset}"
echo "BITNET_STRICT_TOKENIZERS: ${BITNET_STRICT_TOKENIZERS:-unset}"

# Check available fixtures
find tests/fixtures -name "*.model" -o -name "*.json"

# Test strict mode
BITNET_STRICT_TOKENIZERS=1 cargo test -p bitnet-tokenizers --features spm -- --quiet

# Run contract tests with SPM fixture
SPM_MODEL=tests/fixtures/spm/tiny.model cargo test -p bitnet-tokenizers --features "spm,integration-tests" -- test_sentencepiece
```

### Performance Optimization

#### O(1) Token Lookup

```rust
// Optimized: Pre-build lookup table
let id2piece: Box<[String]> = model.pieces()
    .iter()
    .map(|p| p.piece().to_owned())
    .collect::<Vec<_>>()
    .into_boxed_slice();

// O(1) lookup instead of O(n) search
fn token_to_piece(&self, token: u32) -> Option<String> {
    self.id2piece.get(token as usize).cloned()
}
```

#### Memory Efficiency

```rust
// Use Box<[String]> instead of Vec<String> for immutable data
// Avoids extra capacity allocation
pub struct SpmTokenizer {
    id2piece: Box<[String]>, // Exact size allocation
    // ...
}
```

This documentation provides comprehensive coverage of the tokenizer architecture with specific focus on SentencePiece workflow, test fixtures, and practical examples for developers working with issue #241 and related sub-issues.