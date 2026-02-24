# BitNet-rs Tokenizer Integration: Comprehensive Architecture Report

**Date**: 2025-10-25  
**Focus**: Tokenizer architecture, template system, and crossval-per-token integration  
**Scope**: Complete exploration of tokenizer integration patterns and gaps

---

## Executive Summary

The BitNet-rs tokenizer system provides a comprehensive, production-ready architecture for multi-model tokenization with three distinct template types (Raw, Instruct, LLaMA-3 Chat). However, **crossval-per-token currently does NOT use template support**, creating a critical parity gap where template-applied prompts don't match C++ reference tokenization.

**Key Finding**: The prompt template system exists in `bitnet-inference` and is integrated into the CLI (`run`, `chat` commands), but the cross-validation command bypasses templates entirely, causing silent token mismatches when comparing Rust vs C++ inference.

---

## Architecture Overview

### 1. Tokenizer Crate Structure (`crates/bitnet-tokenizers/src/`)

```
bitnet-tokenizers/src/
├── lib.rs                          # Core Tokenizer trait & factories
├── universal.rs                    # UniversalTokenizer (auto-detect backend)
├── strategy.rs                     # Model-specific tokenizer wrappers
│   ├── TokenizerStrategyResolver
│   ├── LlamaTokenizerWrapper       # LLaMA-2/3 with variant detection
│   ├── Gpt2TokenizerWrapper        # GPT-2 with special token handling
│   └── BitNetTokenizerWrapper      # BitNet with quantization awareness
├── hf_tokenizer.rs                 # HuggingFace JSON tokenizer
├── spm_tokenizer.rs                # SentencePiece (SPM) tokenizer
├── sp_tokenizer.rs                 # Alternative SPM support
├── gguf_tokenizer.rs               # Pure-Rust GGUF-embedded tokenizer
├── gguf_loader.rs                  # GGUF tokenizer extraction
├── auto.rs                         # Auto-discovery & routing
├── discovery.rs                    # Tokenizer source detection
├── strategy.rs                     # Fallback chain & resolution
├── mock.rs                         # Mock tokenizer for testing
└── fallback.rs                     # Fallback chain implementation
```

### 2. Tokenizer Trait API

**Core trait** (`lib.rs` lines 81-146):

```rust
pub trait Tokenizer: Send + Sync {
    // Core encoding/decoding
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    
    // Vocabulary & token lookup
    fn vocab_size(&self) -> usize;
    fn real_vocab_size(&self) -> usize;          // For padding-aware vocab
    fn token_to_piece(&self, token: u32) -> Option<String>;
    fn token_to_id(&self, token: &str) -> Option<u32>;  // KEY: Special token resolution
    
    // Special token access
    fn bos_token_id(&self) -> Option<u32>;
    fn eos_token_id(&self) -> Option<u32>;
    fn pad_token_id(&self) -> Option<u32>;
    
    // Legacy compatibility
    fn encode_legacy(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;
    fn decode_legacy(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String>;
}
```

**Key Design**: Parameters `add_bos` and `add_special` control BOS/EOS insertion at encoding time.

---

## Template System Architecture

### 1. Template Types (`bitnet-inference/src/prompt_template.rs`)

**Three supported templates**:

```rust
pub enum TemplateType {
    Raw,           // No formatting
    Instruct,      // Q&A format: "Q: {prompt}\nA:"
    Llama3Chat,    // LLaMA-3 chat with special tokens
}
```

### 2. Template Application Flow

#### Raw Template
- **Purpose**: No formatting, raw text completion
- **Application**: `user_text` → `user_text` (identity)
- **BOS**: Yes (add_bos=true)
- **Special Tokens**: No parsing
- **Stop Sequences**: None

#### Instruct Template
- **Purpose**: Q&A format for instruction-tuned models
- **Application**:
  ```
  System: {system_prompt}
  
  Q: {user_text}
  A:
  ```
- **BOS**: Yes (add_bos=true)
- **Special Tokens**: No parsing required
- **Stop Sequences**: `["\n\nQ:", "\n\nHuman:"]`
- **Example Models**: BitNet base, instruction-tuned variants

#### LLaMA-3 Chat Template
- **Purpose**: Conversational format with special token markers
- **Application**:
  ```
  <|begin_of_text|>
  [<|start_header_id|>system<|end_header_id|>
  {system_prompt}<|eot_id|>]
  <|start_header_id|>user<|end_header_id|>
  {user_text}<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>
  ```
- **BOS**: No (template includes `<|begin_of_text|>`)
- **Special Tokens**: Yes (parse_special=true for `<|...|>` tokens)
- **Stop Sequences**: `["<|eot_id|>", "<|end_of_text|>"]`
- **Token Resolution**: Uses `tokenizer.token_to_id("<|eot_id|>")` → 128009
- **Example Models**: LLaMA-3, LLaMA-3.1

### 3. Template Auto-Detection

**Location**: `bitnet-inference/src/prompt_template.rs` lines 83-109

**Detection Priority**:
1. **GGUF `chat_template` metadata** (Jinja2)
   - Checks for `<|start_header_id|>` + `<|eot_id|>` → LLaMA-3
   - Checks for `{% for message in messages %}` → Instruct
2. **Tokenizer family heuristics**
   - Contains "llama3" or "llama-3" → LLaMA-3
   - Contains "instruct" or "mistral" → Instruct
3. **Fallback**: Raw

### 4. BOS/EOS Control Per Template

| Template | Should Add BOS? | Parse Special? | Default Stops |
|----------|-----------------|----------------|--------------|
| Raw | true | false | none |
| Instruct | true | false | `\n\nQ:`, `\n\nHuman:` |
| Llama3Chat | **false** | true | `<\|eot_id\|>`, `<\|end_of_text\|>` |

**Critical**: LLaMA-3 template includes its own BOS (`<|begin_of_text|>`), so CLI must NOT add additional BOS.

---

## CLI Integration (`crates/bitnet-cli/src/`)

### 1. Template Flag

```bash
bitnet-cli run --prompt-template {raw|instruct|llama3-chat|auto}
```

- **Default**: `auto` (detects from GGUF metadata + tokenizer hints)
- **Storage**: `InferenceCommand.prompt_template: String`
- **Backward compatibility**: `--chat-template` (deprecated, redirects to `--prompt-template`)

### 2. CLI Integration Points

#### InferenceCommand (`commands/inference.rs`)

```rust
pub struct InferenceCommand {
    // ...
    /// Prompt template: auto (detect), raw (no formatting), 
    /// instruct (Q&A format), llama3-chat (LLaMA-3 format)
    #[arg(long, value_name = "TEMPLATE", default_value = "auto")]
    pub prompt_template: String,
    
    /// System prompt for chat models
    #[arg(long, value_name = "TEXT")]
    pub system_prompt: Option<String>,
    
    /// Stop sequences (aliases: --stop-sequence, --stop_sequences)
    #[arg(long = "stop", visible_alias = "stop-sequence", ...)]
    pub stop: Vec<String>,
    
    /// Stop token IDs (numeric token IDs to stop generation)
    #[arg(long = "stop-id", value_name = "ID")]
    pub stop_id: Vec<u32>,
    
    // Control over BOS insertion
    #[arg(long)]
    pub no_bos: bool,
}
```

#### Template Resolution (`commands/inference.rs` lines 741-800+)

```rust
fn resolve_template_type(&self) -> Result<TemplateType> {
    self.resolve_template_type_with_default(TemplateType::Instruct)
}

fn resolve_template_type_with_default(
    &self,
    auto_default: TemplateType,
) -> Result<TemplateType> {
    if self.prompt_template.eq_ignore_ascii_case("auto") {
        // Auto-detect from GGUF + tokenizer hints
        let detected = self.auto_detect_template();
        if matches!(detected, TemplateType::Raw) {
            return Ok(auto_default); // Fallback if detection failed
        }
        Ok(detected)
    } else {
        self.prompt_template.parse()
            .context("Invalid prompt template")
    }
}
```

#### Template Application (`commands/inference.rs`)

```rust
fn apply_prompt_template(&self, user_text: &str) -> Result<String> {
    let template_type = self.resolve_template_type()?;
    Ok(template_type.apply(user_text, self.system_prompt.as_deref()))
}
```

### 3. Tokenization with Template Awareness

**Example CLI Usage** (from inference.rs docs, lines 94-112):

```bash
# Auto-detect template (recommended)
bitnet-cli run --model model.gguf --prompt "Capital of France?"

# Explicit Instruct (Q&A)
bitnet-cli run --model model.gguf --prompt-template instruct \
  --prompt "What is 2+2?" --max-tokens 16

# LLaMA-3 chat
bitnet-cli run --model model.gguf --prompt-template llama3-chat \
  --system-prompt "You are helpful" \
  --prompt "Explain photosynthesis" --max-tokens 128

# Raw (no formatting)
bitnet-cli run --model model.gguf --prompt-template raw \
  --prompt "2+2=" --max-tokens 16
```

**Encoding Process**:

1. Apply template: `user_text` → `formatted_text`
2. Determine tokenizer settings from template:
   - `add_bos = template.should_add_bos()` (true for Raw/Instruct, false for LLaMA-3)
   - `parse_special = template.parse_special()` (false for Raw/Instruct, true for LLaMA-3)
3. Encode: `tokenizer.encode(formatted_text, add_bos, parse_special)`
4. Resolve template stop sequences:
   - `stop_ids = template.resolve_stop_token_ids(&tokenizer)`
   - Merges with user-provided `--stop-id` flags

---

## BOS/EOS Token Handling

### 1. Tokenizer Level (HfTokenizer)

**Location**: `hf_tokenizer.rs` lines 113-142

```rust
fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
    // 1. Use HF tokenizer's built-in encoding (may include EOS via add_special)
    let enc = self.inner.encode(EncodeInput::Single(text.into()), add_special)?;
    let mut ids = enc.get_ids().to_vec();
    
    // 2. Add BOS if requested and not already present
    if add_bos && let Some(bos) = self.bos_id && 
       (ids.is_empty() || ids[0] != bos) {
        ids.insert(0, bos);
    }
    
    // 3. Add EOS if requested and not already present
    if add_special && let Some(eos) = self.eos_id && 
       (ids.is_empty() || ids[ids.len() - 1] != eos) {
        ids.push(eos);
    }
    
    Ok(ids)
}
```

**Key**: Avoids duplicate BOS/EOS insertion with presence checks.

### 2. Template Control

**Raw Template**:
```rust
pub fn should_add_bos(&self) -> bool {
    matches!(self, Self::Raw | Self::Instruct)  // true
}
```

**Instruct Template**:
```rust
pub fn should_add_bos(&self) -> bool {
    matches!(self, Self::Raw | Self::Instruct)  // true
}
```

**LLaMA-3 Chat Template**:
```rust
pub fn should_add_bos(&self) -> bool {
    matches!(self, Self::Llama3Chat)  // false - template includes BOS
}
```

### 3. Special Token Discovery

**HfTokenizer** (`hf_tokenizer.rs` lines 46-64):
```rust
let vocab = inner.get_vocab(true);
for (token, id) in vocab {
    // BOS patterns: <s>, <bos>, <|startoftext|>
    // EOS patterns: </s>, <eos>, <|endoftext|>
}
```

**GGUF RustTokenizer** (`gguf_loader.rs`):
Extracts BOS/EOS from GGUF metadata:
- `tokenizer.ggml.bos_token_id`
- `tokenizer.ggml.eos_token_id`
- `tokenizer.ggml.add_bos_token` (hint)

---

## Special Token Management

### 1. Token ID Resolution

**API**: `tokenizer.token_to_id(&str) -> Option<u32>`

**Implementation** (HfTokenizer, lines 160-164):
```rust
fn token_to_id(&self, token: &str) -> Option<u32> {
    let vocab = self.inner.get_vocab(true);
    vocab.get(token).copied()
}
```

**Mock Implementation** (for testing):
```rust
pub fn with_special_tokens(mappings: &[(&str, u32)]) -> Self {
    // Pre-populate token_to_id_map with special tokens
}
```

### 2. Template Stop Sequence Resolution

**Location**: `prompt_template.rs` lines 199-210

```rust
pub fn resolve_stop_token_ids(
    &self, 
    tokenizer: &dyn bitnet_tokenizers::Tokenizer
) -> Vec<u32> {
    let stop_sequences = self.default_stop_sequences();
    let mut stop_ids = Vec::new();
    
    for seq in &stop_sequences {
        if let Some(id) = tokenizer.token_to_id(seq) {
            stop_ids.push(id);
        }
    }
    
    stop_ids
}
```

**Example**: LLaMA-3 Chat
```rust
// template.default_stop_sequences() returns:
// ["<|eot_id|>", "<|end_of_text|>"]

// Resolve with real tokenizer:
// tokenizer.token_to_id("<|eot_id|>") → Some(128009)
// tokenizer.token_to_id("<|end_of_text|>") → Some(128010)

// Result: stop_ids = [128009, 128010]
```

### 3. Special Token Parsing Control

**Token Parsing** (whether to interpret special tokens like `<|eot_id|>`):

```rust
pub fn parse_special(&self) -> bool {
    matches!(self, Self::Llama3Chat)  // Only true for LLaMA-3
}
```

Passed to tokenizer encoding:
```rust
tokenizer.encode(formatted_text, add_bos=true, add_special=parse_special())
```

For LLaMA-3: `add_special=true` ensures `<|eot_id|>` tokens are recognized during encoding.

---

## Auto-Detection Flow

### 1. Complete Detection Pipeline

```
User Prompt
    ↓
[1] Check --prompt-template flag
    ├─ "raw" → use Raw
    ├─ "instruct" → use Instruct
    ├─ "llama3-chat" → use LLaMA-3
    └─ "auto" → detect below
[2] Auto-detect (if "auto")
    ├─ Read GGUF chat_template metadata
    │  ├─ Contains "<|start_header_id|>" + "<|eot_id|>" → LLaMA-3
    │  └─ Contains "for message in messages" → Instruct
    ├─ Read tokenizer family hints
    │  ├─ Contains "llama3" or "llama-3" → LLaMA-3
    │  └─ Contains "instruct" or "mistral" → Instruct
    └─ Fallback → Instruct (safer than Raw for most models)
[3] Apply template
    ├─ Format prompt with template rules
    └─ Determine BOS/EOS/special token parsing
[4] Tokenize
    └─ tokenizer.encode(formatted, add_bos, parse_special)
[5] Generate
    └─ Use template's default stop sequences
```

### 2. Detection Implementation

**Location**: `bitnet-inference/src/prompt_template.rs` lines 83-109

```rust
pub fn detect(
    tokenizer_name: Option<&str>,
    chat_template_jinja: Option<&str>
) -> Self {
    // Priority 1: GGUF chat_template
    if let Some(jinja) = chat_template_jinja {
        if jinja.contains("<|start_header_id|>") && jinja.contains("<|eot_id|>") {
            return Self::Llama3Chat;
        }
        if jinja.contains("{% for message in messages %}") {
            return Self::Instruct;
        }
    }
    
    // Priority 2: Tokenizer name heuristics
    if let Some(name) = tokenizer_name {
        let lower = name.to_ascii_lowercase();
        if lower.contains("llama3") || lower.contains("llama-3") {
            return Self::Llama3Chat;
        }
        if lower.contains("instruct") || lower.contains("mistral") {
            return Self::Instruct;
        }
    }
    
    // Priority 3: Fallback
    Self::Raw
}
```

---

## Cross-Validation Integration

### Current State: **NO TEMPLATE SUPPORT**

**Location**: `xtask/src/main.rs::crossval_per_token_cmd()` (lines ~1350)

```rust
fn crossval_per_token_cmd(
    model_path: &Path,
    tokenizer_path: &Path,
    prompt: &str,
    _max_tokens: usize,
    cos_tol: f32,
    format: &str,
) -> Result<()> {
    // ... setup ...
    
    // ISSUE: Direct tokenization without template application
    let tokenizer = bitnet_tokenizers::loader::load_tokenizer(tokenizer_path)?;
    let tokens = tokenizer.encode(prompt, false, false)?;  // ← Always Raw!
    
    // Template-applied prompt not tokenized:
    // ❌ template.apply(prompt, system_prompt)
    // ❌ template.should_add_bos()
    // ❌ template.parse_special()
```

**Problems**:
1. **No template application**: Prompt is tokenized raw, not formatted per template rules
2. **No BOS control**: Always calls `encode(..., false, false)` 
3. **No special token handling**: Doesn't set `parse_special=true` for LLaMA-3
4. **No stop sequence resolution**: Template stops not extracted
5. **Silent C++ mismatch**: If C++ uses template formatting, token sequences diverge at index 0

### Token Parity Pre-Gate

**Location**: `crossval/src/token_parity.rs` (lines 79-110)

```rust
pub fn validate_token_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[i32],
    prompt: &str,
) -> anyhow::Result<()> {
    let cpp_tokens_u32: Vec<u32> = cpp_tokens.iter().map(|&id| id as u32).collect();
    
    if rust_tokens != cpp_tokens_u32.as_slice() {
        // Find first diff and print diagnostic error
        let error = TokenParityError { ... };
        eprintln!("{}", format_token_mismatch_error(&error));
        anyhow::bail!("Token sequence mismatch at index {}", first_diff);
    }
    
    Ok(())
}
```

**Error Output Example**:
```
❌ Token Sequence Mismatch
Fix BOS/template before comparing logits

Rust tokens:
  [1229, 374]

C++ tokens:
  [128000, 1229, 374]

First diff at index: 0

Suggested fixes:
  • Use --prompt-template raw
  • Add --no-bos flag
  • Check GGUF chat_template metadata
  • Use --dump-ids to inspect token sequences
```

**Exit Code**: 2 (indicates usage error, not runtime failure)

---

## Integration Points with CLI

### 1. Chat Command (`commands/chat.rs`)

Uses template auto-detection and applies before encoding.

### 2. Inference Run Command (`commands/inference.rs`)

- Resolves template type
- Applies to user prompts
- Merges template stops with user-provided stops
- Controls BOS insertion via `no_bos` flag

### 3. Xtask Crossval Commands

- `crossval`: Uses templates indirectly (no direct integration)
- `crossval-per-token`: **DOES NOT USE TEMPLATES** ← GAP

---

## Configuration Options

### Tokenizer Loading

```bash
# Auto-discover tokenizer from model directory
bitnet-cli run --model model.gguf

# Explicit tokenizer path
bitnet-cli run --model model.gguf --tokenizer tokenizer.json

# GGUF-embedded tokenizer (no external file needed)
bitnet-cli run --model model.gguf  # Auto-uses embedded if present
```

### Template Control

```bash
# Auto-detect (recommended)
bitnet-cli run ... --prompt-template auto

# Explicit templates
bitnet-cli run ... --prompt-template raw
bitnet-cli run ... --prompt-template instruct
bitnet-cli run ... --prompt-template llama3-chat
```

### BOS/EOS Control

```bash
# Disable BOS (override template)
bitnet-cli run ... --no-bos

# System prompt (used by Instruct & LLaMA-3 templates)
bitnet-cli run ... --system-prompt "You are helpful"

# Manual stop sequences (merge with template defaults)
bitnet-cli run ... --stop "</s>" --stop "\n\n"

# Manual stop token IDs (merge with template defaults)
bitnet-cli run ... --stop-id 128009 --stop-id 128010
```

### Stop String Window

```bash
# Control tail buffer size for string-based stop detection (default: 64 bytes)
bitnet-cli run ... --stop-string-window 128
```

---

## API Surface: Key Exports

### From `bitnet-tokenizers`

```rust
pub trait Tokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>>;
    fn token_to_id(&self, token: &str) -> Option<u32>;
    fn bos_token_id(&self) -> Option<u32>;
    fn eos_token_id(&self) -> Option<u32>;
    fn vocab_size(&self) -> usize;
    fn real_vocab_size(&self) -> usize;  // Padding-aware
}

pub fn load_tokenizer(path: &Path) -> Result<Arc<dyn Tokenizer>>;
pub struct HfTokenizer { ... }
pub struct RustGgufTokenizer { ... }
```

### From `bitnet-inference`

```rust
pub enum TemplateType {
    Raw,
    Instruct,
    Llama3Chat,
}

impl TemplateType {
    pub fn detect(
        tokenizer_name: Option<&str>,
        chat_template_jinja: Option<&str>
    ) -> Self;
    
    pub fn apply(&self, user_text: &str, system_prompt: Option<&str>) -> String;
    
    pub fn default_stop_sequences(&self) -> Vec<String>;
    
    pub fn resolve_stop_token_ids(
        &self,
        tokenizer: &dyn bitnet_tokenizers::Tokenizer
    ) -> Vec<u32>;
    
    pub fn should_add_bos(&self) -> bool;
    pub fn parse_special(&self) -> bool;
}

pub struct PromptTemplate {
    template_type: TemplateType,
    system_prompt: Option<String>,
    conversation_history: Vec<(String, String)>,
}

impl PromptTemplate {
    pub fn new(template_type: TemplateType) -> Self;
    pub fn with_system_prompt(self, prompt: impl Into<String>) -> Self;
    pub fn format(&self, user_text: &str) -> String;
    pub fn stop_sequences(&self) -> Vec<String>;
}
```

---

## Gap Analysis: crossval-per-token

### Current Implementation Limitations

| Aspect | Current | Needed |
|--------|---------|--------|
| Template support | ❌ None | ✅ All (Raw, Instruct, LLaMA-3) |
| BOS control | ❌ Hardcoded `false` | ✅ Template-dependent |
| Special token parsing | ❌ Hardcoded `false` | ✅ Template-dependent |
| Stop sequence resolution | ❌ No | ✅ From template |
| System prompt | ❌ No | ✅ Optional via flag |
| Template auto-detection | ❌ No | ✅ From GGUF metadata |

### Problem Manifestation

**Scenario**: Cross-validating LLaMA-3 chat model

```bash
# User intention: cross-validate with LLaMA-3 chat formatting
cargo run -p xtask -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "What is 2+2?"

# Current behavior:
#   Tokenizes: "What is 2+2?" (raw, no template)
#   Rust tokens: [1229, 374]  (hypothetical)
#
# C++ reference (with template in llama.cpp):
#   Tokenizes with LLaMA-3 chat template
#   C++ tokens: [128000, 128001, ..., 1229, 374]
#
# Token parity check fails at index 0
# Exit code: 2 (usage error)
```

### Required Integration Points

**Needed in `xtask/src/main.rs::crossval_per_token_cmd()`**:

1. Add `--prompt-template` flag (default: "auto")
2. Add `--system-prompt` flag (optional)
3. Detect template from GGUF metadata if "auto"
4. Apply template to prompt before tokenization
5. Determine tokenizer encoding flags from template:
   ```rust
   let template_type = TemplateType::detect(...)?;
   let formatted_prompt = template_type.apply(&prompt, system_prompt.as_deref());
   let add_bos = template_type.should_add_bos();
   let parse_special = template_type.parse_special();
   let tokens = tokenizer.encode(&formatted_prompt, add_bos, parse_special)?;
   ```
6. Resolve template stop sequences:
   ```rust
   let stop_ids = template_type.resolve_stop_token_ids(&tokenizer);
   // Merge with user-provided --stop-id flags
   ```

---

## Mock Tokenizer for Testing

**Location**: `bitnet-tokenizers/src/mock.rs`

```rust
pub struct MockTokenizer {
    vocab_size: usize,
    token_to_id_map: HashMap<String, u32>,
}

impl MockTokenizer {
    pub fn new() -> Self { ... }
    
    pub fn with_special_tokens(mappings: &[(&str, u32)]) -> Self {
        // Pre-populate token-to-ID mappings for special tokens
    }
}
```

**Used in prompt_template tests** (lines 421-475):

```rust
#[test]
fn test_template_glue_with_real_token_ids() {
    let tokenizer = MockTokenizer::with_special_tokens(&[
        ("<|eot_id|>", 128009),
        ("<|end_of_text|>", 128010),
    ]);
    
    let template = TemplateType::Llama3Chat;
    let stop_ids = template.resolve_stop_token_ids(&tokenizer);
    
    assert!(stop_ids.contains(&128009));  // Resolved correctly
    assert!(stop_ids.contains(&128010));
}
```

---

## Model-Specific Wrapper Architecture

**Location**: `crates/bitnet-tokenizers/src/strategy.rs` (lines 295-608)

### LlamaTokenizerWrapper

Detects LLaMA variant (2, 3, CodeLlama) by vocabulary size:
- **32000**: LLaMA-2
- **32016**: CodeLlama
- **128256**: LLaMA-3

Sets BOS/EOS accordingly:
```rust
fn bos_token_id(&self) -> Option<u32> {
    Some(1)  // LLaMA BOS
}

fn eos_token_id(&self) -> Option<u32> {
    Some(2)  // LLaMA EOS
}
```

### Gpt2TokenizerWrapper

```rust
fn bos_token_id(&self) -> Option<u32> {
    None  // GPT-2 doesn't use BOS
}

fn eos_token_id(&self) -> Option<u32> {
    Some(50256)  // GPT-2 EOS
}
```

### BitNetTokenizerWrapper

Quantization-aware wrapper:
```rust
pub struct BitNetTokenizerWrapper {
    inner: Arc<dyn Tokenizer>,
    quantization_type: QuantizationType,
}

fn validate_quantization_compatibility(&self, tokens: &[u32]) -> Result<()> {
    // Validate token IDs are compatible with quantization format
}
```

---

## Recommended crossval-per-token Enhancement

### Proposed Changes

1. **Add flag to InferenceCommand / xtask**:
   ```rust
   #[arg(long, default_value = "auto")]
   pub prompt_template: String,
   
   #[arg(long)]
   pub system_prompt: Option<String>,
   ```

2. **Apply template before tokenization**:
   ```rust
   let template_type = TemplateType::detect(
       tokenizer_name,
       chat_template_jinja
   ).or_else(|| TemplateType::parse(&template_flag))?;
   
   let formatted_prompt = template_type.apply(
       &prompt,
       system_prompt.as_deref()
   );
   ```

3. **Determine tokenizer encoding flags**:
   ```rust
   let add_bos = template_type.should_add_bos();
   let parse_special = template_type.parse_special();
   ```

4. **Encode with template-aware flags**:
   ```rust
   let tokens = tokenizer.encode(
       &formatted_prompt,
       add_bos,
       parse_special
   )?;
   ```

5. **Resolve stop token IDs from template**:
   ```rust
   let mut stop_ids = template_type.resolve_stop_token_ids(&tokenizer);
   // Merge with user-provided --stop-id flags
   ```

### Backward Compatibility

Add `--prompt-template raw` as default equivalent to current behavior:
```bash
# Current (template-less):
cargo run -p xtask -- crossval-per-token --model m.gguf --prompt "..."

# After enhancement (preserves current behavior):
cargo run -p xtask -- crossval-per-token --model m.gguf \
  --prompt-template raw --prompt "..."

# New capability (template-aware):
cargo run -p xtask -- crossval-per-token --model m.gguf \
  --prompt-template auto --prompt "..."
```

---

## Test Infrastructure

### Template Tests (`prompt_template.rs` lines 352-611)

- `test_raw_template()`: Identity transformation
- `test_instruct_template()`: Q&A formatting
- `test_llama3_chat_template()`: Special tokens & formatting
- `test_template_from_str()`: Parsing
- `test_stop_sequences()`: Default stops per template
- `test_resolve_stop_token_ids()`: Stop token ID resolution with mock tokenizer
- `test_template_glue_with_real_token_ids()`: End-to-end template→stops→token_ids
- `test_bos_control()`: BOS insertion rules
- `test_parse_special_control()`: Special token parsing flags
- `test_conversation_history()`: Multi-turn chat support
- `test_render_chat_*()`: Conversation rendering

### Token Parity Tests (`crossval/src/token_parity.rs`)

- AC1: Mismatch detection before logits
- AC2: Both token sequences displayed
- AC3: First diff position identified
- AC4: Exit code 2 on mismatch
- AC5-AC8: Error message quality
- AC9: Silent success when tokens match

---

## Summary: Tokenizer API Surface

| Component | Module | Key Types | Entry Points |
|-----------|--------|-----------|--------------|
| Tokenizer trait | `bitnet-tokenizers` | `Tokenizer` | `load_tokenizer()` |
| HF tokenizer | `bitnet-tokenizers::hf_tokenizer` | `HfTokenizer` | `HfTokenizer::from_file()` |
| GGUF tokenizer | `bitnet-tokenizers::gguf_loader` | `RustGgufTokenizer` | `TokenizerBuilder::from_gguf_reader()` |
| Template types | `bitnet-inference::prompt_template` | `TemplateType`, `PromptTemplate` | `TemplateType::detect()`, `TemplateType::apply()` |
| Template stops | `bitnet-inference::prompt_template` | (in TemplateType) | `resolve_stop_token_ids()` |
| Special tokens | `bitnet-tokenizers` | (in Tokenizer trait) | `token_to_id()`, `bos_token_id()`, `eos_token_id()` |
| Token parity | `bitnet-crossval` | `TokenParityError` | `validate_token_parity()` |

---

## Conclusion

BitNet-rs implements a **sophisticated, production-ready tokenizer system** with:

✅ Multi-backend support (HF JSON, SentencePiece, GGUF-embedded)  
✅ Three prompt templates (Raw, Instruct, LLaMA-3 Chat)  
✅ Template auto-detection from GGUF metadata  
✅ BOS/EOS handling with deduplication  
✅ Special token resolution (e.g., `<|eot_id|>` → token ID)  
✅ Stop sequence management per template  
✅ Model-specific wrappers (LLaMA, GPT-2, BitNet)  
✅ Comprehensive testing infrastructure  

❌ **But crossval-per-token bypasses templates entirely**, creating silent token mismatch errors

**Recommended Fix**: Integrate template support into `crossval-per-token` command to ensure Rust and C++ token parity when comparing formatted prompts.

