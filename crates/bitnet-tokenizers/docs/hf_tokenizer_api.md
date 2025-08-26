# HuggingFace Tokenizer API Contract

## Overview
The `HfTokenizer` provides support for loading and using tokenizers in the Hugging Face tokenizer.json format, which is widely used across the ML community for transformer models.

## API Contract

### Loading Tokenizers

```rust
pub fn load_tokenizer(path: &Path) -> Result<Box<dyn Tokenizer>>
```

**Supported formats:**
- `.json` files with `model.type` field (HuggingFace format)
- Must contain valid tokenizer model definition

**Validation:**
1. File must be readable and valid JSON
2. Must contain `model.type` field
3. Tokenizer must successfully initialize from the JSON structure

**Error handling:**
- Returns descriptive error if JSON is invalid
- Returns error if `model.type` is missing
- Propagates tokenizer initialization errors with context

### HfTokenizer Implementation

```rust
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
}
```

**Special Token Detection:**
Automatically detects and stores special tokens from vocabulary:
- BOS tokens: `<s>`, `<bos>`, `<|startoftext|>` (case-insensitive)
- EOS tokens: `</s>`, `<eos>`, `<|endoftext|>` (case-insensitive)

### Tokenizer Trait Implementation

All methods from the `Tokenizer` trait are implemented:

1. **encode(text, add_bos, add_special) -> Result<Vec<u32>>**
   - Encodes text to token IDs
   - Optionally adds BOS token at start if `add_bos` is true
   - Optionally adds EOS token at end if `add_special` is true
   - Prevents duplicate special tokens

2. **decode(ids) -> Result<String>**
   - Decodes token IDs back to text
   - Handles special tokens appropriately

3. **vocab_size() -> usize**
   - Returns the total vocabulary size including special tokens

4. **token_to_piece(token) -> Option<String>**
   - Converts a token ID to its string representation
   - Returns None for invalid token IDs

5. **bos_token_id() -> Option<u32>**
   - Returns the detected BOS token ID if present

6. **eos_token_id() -> Option<u32>**
   - Returns the detected EOS token ID if present

## Compatibility

The implementation is fully compatible with tokenizers created by the Hugging Face `tokenizers` library and can load any valid tokenizer.json file that contains the required `model.type` field.

## Testing

Comprehensive tests are provided in `tests/hf_json.rs` that verify:
- Basic encoding and decoding
- Vocabulary size reporting
- Unknown token handling
- Token to piece conversion
- Special token detection

## Example Usage

```rust
use bitnet_tokenizers::load_tokenizer;
use std::path::Path;

let tokenizer = load_tokenizer(Path::new("tokenizer.json"))?;
let ids = tokenizer.encode("Hello world", false, false)?;
let text = tokenizer.decode(&ids)?;
```