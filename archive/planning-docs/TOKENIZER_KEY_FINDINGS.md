# Tokenizer Integration: Key Findings & Recommendations

## Critical Gap: crossval-per-token Template Support

### The Problem
`crossval-per-token` command tokenizes prompts **without applying templates**, while C++ reference may use template formatting. This creates silent token mismatches at index 0 when comparing inference outputs.

**Current Code** (`xtask/src/main.rs`):
```rust
let tokens = tokenizer.encode(prompt, false, false)?;  // Always raw!
```

### Impact
- ❌ Token sequences diverge when C++ uses template formatting
- ❌ Token parity check fails immediately (exit code 2)
- ❌ Users can't cross-validate template-formatted prompts
- ❌ Forces workaround: use `--prompt-template raw` flag (doesn't exist)

### Solution
Add template support to `crossval-per-token`:

1. **New flags**:
   ```bash
   cargo run -p xtask -- crossval-per-token \
     --model model.gguf \
     --tokenizer tokenizer.json \
     --prompt "What is 2+2?" \
     --prompt-template auto          # NEW
     --system-prompt "You are math"  # NEW
   ```

2. **Implementation**:
   ```rust
   // 1. Detect template from GGUF metadata or flag
   let template_type = if prompt_template == "auto" {
       TemplateType::detect(tokenizer_name, chat_template_jinja)
   } else {
       prompt_template.parse()?
   };
   
   // 2. Apply template
   let formatted_prompt = template_type.apply(&prompt, system_prompt.as_deref());
   
   // 3. Encode with template-aware flags
   let tokens = tokenizer.encode(
       &formatted_prompt,
       template_type.should_add_bos(),    // false for LLaMA-3
       template_type.parse_special()      // true for LLaMA-3
   )?;
   ```

---

## Architecture: Three-Layer Template System

### Layer 1: Tokenizer Trait (`bitnet-tokenizers`)
```rust
pub trait Tokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>>;
    fn token_to_id(&self, token: &str) -> Option<u32>;  // For special tokens
    fn bos_token_id(&self) -> Option<u32>;
    fn eos_token_id(&self) -> Option<u32>;
}
```

**BOS/EOS Handling**: Deduplication prevents double insertion
```rust
if add_bos && let Some(bos) = self.bos_id && (ids.is_empty() || ids[0] != bos) {
    ids.insert(0, bos);  // Only if not already present
}
```

### Layer 2: Template Types (`bitnet-inference`)
```rust
pub enum TemplateType {
    Raw,        // No formatting
    Instruct,   // "Q: {prompt}\nA:"
    Llama3Chat, // "<|begin_of_text|>...<|start_header_id|>user<|end_header_id|>..."
}
```

| Template | BOS | Special Parsing | Stops | Use Case |
|----------|-----|-----------------|-------|----------|
| Raw | true | false | none | Completion |
| Instruct | true | false | `\n\nQ:` | Q&A |
| LLaMA-3 | **false** | true | `<\|eot_id\|>` | Chat |

### Layer 3: CLI Integration (`bitnet-cli`)
```bash
# Auto-detect (recommended)
bitnet-cli run --model model.gguf --prompt "Question?" --prompt-template auto

# Explicit template
bitnet-cli run --model model.gguf --prompt "Question?" --prompt-template llama3-chat

# System prompt
bitnet-cli run --model model.gguf --system-prompt "You are helpful" --prompt "..."

# Override BOS
bitnet-cli run --model model.gguf --prompt "..." --no-bos
```

---

## Template Auto-Detection Priority

```
[1] GGUF chat_template metadata (Jinja2)
    ├─ "<|start_header_id|>" + "<|eot_id|>" → LLaMA-3
    └─ "for message in messages" → Instruct

[2] Tokenizer family heuristics
    ├─ "llama3" or "llama-3" → LLaMA-3
    └─ "instruct" or "mistral" → Instruct

[3] Fallback
    └─ Raw (or Instruct for safety)
```

---

## Special Token Resolution

### Token ID Lookup
```rust
pub fn token_to_id(&self, token: &str) -> Option<u32>

// Example: LLaMA-3 chat
tokenizer.token_to_id("<|eot_id|>") → Some(128009)
```

### Stop Sequence Resolution
```rust
template_type.resolve_stop_token_ids(&tokenizer) → Vec<u32>

// LLaMA-3: ["<|eot_id|>", "<|end_of_text|>"] → [128009, 128010]
```

### Template-Aware Encoding
```rust
// Control token parsing for special tokens like <|eot_id|>
let parse_special = template_type.parse_special();  // true for LLaMA-3

tokenizer.encode(formatted_text, add_bos, parse_special)
```

---

## BOS/EOS Handling Rules

### When to Add BOS
- **Raw**: YES (default)
- **Instruct**: YES (default)
- **LLaMA-3 Chat**: NO (template includes `<|begin_of_text|>`)

### Implementation
```rust
if add_bos && let Some(bos) = self.bos_id && 
   (ids.is_empty() || ids[0] != bos) {
    ids.insert(0, bos);  // Avoid duplicate
}
```

### Template Control
```rust
template_type.should_add_bos()
match template {
    Raw | Instruct => true,
    Llama3Chat => false,  // Template has its own BOS
}
```

---

## Key APIs for crossval-per-token Integration

### Tokenizer Loading
```rust
use bitnet_tokenizers::loader;
let tokenizer = loader::load_tokenizer(tokenizer_path)?;
```

### Template Detection & Application
```rust
use bitnet_inference::TemplateType;

// Detect
let template = TemplateType::detect(tokenizer_name, chat_template_jinja);

// Or parse flag
let template: TemplateType = prompt_template_str.parse()?;

// Apply
let formatted = template.apply(&prompt, system_prompt.as_deref());

// Get encoding flags
let add_bos = template.should_add_bos();
let parse_special = template.parse_special();

// Get stop sequences
let default_stops = template.default_stop_sequences();
let stop_ids = template.resolve_stop_token_ids(&tokenizer);
```

### Token Encoding
```rust
let tokens = tokenizer.encode(&formatted_prompt, add_bos, parse_special)?;
```

---

## Mock Tokenizer for Testing

```rust
use bitnet_tokenizers::MockTokenizer;

// Basic mock
let tokenizer = MockTokenizer::new();

// With special token mappings
let tokenizer = MockTokenizer::with_special_tokens(&[
    ("<|eot_id|>", 128009),
    ("<|end_of_text|>", 128010),
]);

// Usage in tests
let stop_ids = template.resolve_stop_token_ids(&tokenizer);
```

---

## Model-Specific Wrappers

### LlamaTokenizerWrapper
```rust
// Auto-detects variant by vocab_size
// 32000 → LLaMA-2
// 32016 → CodeLlama
// 128256 → LLaMA-3

// Sets correct BOS/EOS:
fn bos_token_id(&self) -> Option<u32> { Some(1) }
fn eos_token_id(&self) -> Option<u32> { Some(2) }
```

### BitNetTokenizerWrapper
```rust
// Quantization-aware
pub fn validate_quantization_compatibility(&self, tokens: &[u32]) -> Result<()>
```

---

## Stop Sequence Merging

```rust
// Template defaults
let mut stop_ids = template_type.resolve_stop_token_ids(&tokenizer);

// User-provided overrides
stop_ids.extend(&user_provided_stop_ids);

// Used during generation
// Generation stops when token in stop_ids appears, or string match in stop_sequences
```

---

## Test Coverage

### Template Tests (`bitnet-inference/src/prompt_template.rs`)
- ✅ Raw, Instruct, LLaMA-3 formatting
- ✅ BOS/EOS control per template
- ✅ Special token parsing flags
- ✅ Stop sequence resolution with mock tokenizer
- ✅ End-to-end template→stops→token_ids flow

### Token Parity Tests (`crossval/src/token_parity.rs`)
- ✅ Mismatch detection
- ✅ Error message with suggestions
- ✅ First diff position identification

---

## Files to Modify for crossval-per-token Support

1. **`xtask/src/main.rs`**
   - Add `prompt_template: String` field
   - Add `system_prompt: Option<String>` field
   - Modify `crossval_per_token_cmd()` to apply template

2. **`crossval/src/token_parity.rs`** (optional)
   - Error message already suggests `--prompt-template raw` flag

3. **Documentation** (optional)
   - Update crossval examples with template usage

---

## Backward Compatibility

Current command still works with sensible default:
```bash
cargo run -p xtask -- crossval-per-token \
  --model model.gguf --tokenizer tokenizer.json --prompt "..."
  # Uses --prompt-template raw (current behavior)
```

New capability available:
```bash
cargo run -p xtask -- crossval-per-token \
  --model model.gguf --tokenizer tokenizer.json --prompt "..." \
  --prompt-template auto  # NEW: auto-detect from GGUF
```

---

## Conclusion

The tokenizer integration system is comprehensive and well-designed. The **only gap** is that `crossval-per-token` doesn't apply templates, causing silent token mismatches when testing template-formatted prompts against C++ reference implementations.

**Recommendation**: Add `--prompt-template` and `--system-prompt` flags to `crossval-per-token` (approximately 30 lines of code) to enable template-aware cross-validation.

