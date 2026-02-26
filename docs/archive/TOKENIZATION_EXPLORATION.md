# BitNet CLI Tokenization System - Exploration Report

## Overview

This report documents the tokenization flow in BitNet-rs CLI with focus on template application, BOS handling, token output, and reusable components for xtask integration.

## 1. Template Application Flow

### 1.1 Template Type Detection (`crates/bitnet-inference/src/prompt_template.rs:76-109`)

**Function**: `TemplateType::detect()`
- **Signature**: `pub fn detect(tokenizer_name: Option<&str>, chat_template_jinja: Option<&str>) -> Self`
- **Location**: `/crates/bitnet-inference/src/prompt_template.rs:83`
- **Detection Priority**:
  1. GGUF chat_template metadata (if present)
  2. Tokenizer family name heuristics
  3. Fallback to Raw

**Key Logic**:
```rust
// Priority 1: Check for LLaMA-3 signature
if jinja.contains("<|start_header_id|>") && jinja.contains("<|eot_id|>") {
    return Self::Llama3Chat;
}

// Priority 2: Tokenizer name heuristics
if lower.contains("llama3") || lower.contains("llama-3") {
    return Self::Llama3Chat;
}
```

### 1.2 Template Application to Prompts (`crates/bitnet-inference/src/prompt_template.rs:111-167`)

**Function**: `TemplateType::apply()`
- **Signature**: `pub fn apply(&self, user_text: &str, system_prompt: Option<&str>) -> String`
- **Location**: `/crates/bitnet-inference/src/prompt_template.rs:112`

**Template Variants**:

#### Raw Template (No Formatting)
```rust
Self::Raw => user_text.to_string()
```
- Passes through user text unchanged

#### Instruct Template (Q&A Format)
```rust
// Optional system prompt
System: {system}

Q: {user_text}
A:
```
- Signals Q&A intent with Q:/A: markers

#### LLaMA-3 Chat Template
```rust
<|begin_of_text|>
[<|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>]
<|start_header_id|>user<|end_header_id|>
{user_text}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```
- Includes special tokens (`<|begin_of_text|>`, `<|eot_id|>`)
- Requires special token parsing during encoding

**CLI Integration Point** (`crates/bitnet-cli/src/main.rs:1078`):
```rust
let formatted_prompt = template_type.apply(&prompt, system_prompt.as_deref());
```

## 2. BOS Handling

### 2.1 BOS Policy Determination

**Location**: `crates/bitnet-cli/src/main.rs:1103-1108`
```rust
// Determine BOS policy (user flag wins, else template default)
let bos_policy = if bos {
    true // explicit --bos flag
} else {
    template_type.should_add_bos() // template default
};
```

**CLI Flags**:
- `--bos`: Explicitly insert BOS token (overrides template default)
- `--no-bos`: Disable BOS (only in `inference` subcommand)

### 2.2 BOS Template Defaults (`crates/bitnet-inference/src/prompt_template.rs:212-219`)

**Function**: `TemplateType::should_add_bos()`
- **Signature**: `pub fn should_add_bos(&self) -> bool`
- **Location**: `/crates/bitnet-inference/src/prompt_template.rs:214`

**Behavior by Template**:
```rust
Self::Raw | Self::Instruct => true,           // Add BOS (except if --no-bos)
Self::Llama3Chat => false,                     // Template includes <|begin_of_text|>
```

**Rationale**: LLaMA-3 chat template explicitly includes `<|begin_of_text|>` special token, so separate BOS insertion would duplicate it.

### 2.3 Inference Command BOS Logic (`crates/bitnet-cli/src/commands/inference.rs:1433-1444`)

**Function**: `InferenceCommand::should_add_bos()`
- **Signature**: `fn should_add_bos(&self) -> bool`
- **Location**: `/crates/bitnet-cli/src/commands/inference.rs:1433`

```rust
fn should_add_bos(&self) -> bool {
    if self.no_bos {
        return false;  // --no-bos flag takes precedence
    }

    // Check template preference with auto-detection
    if let Ok(template_type) = self.resolve_template_type() {
        template_type.should_add_bos()
    } else {
        true // Default to adding BOS
    }
}
```

## 3. Tokenization with Template/BOS

### 3.1 Tokenizer Trait Core API (`crates/bitnet-tokenizers/src/lib.rs:81-82`)

**Function Signature**:
```rust
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn token_to_piece(&self, token: u32) -> Option<String>;
    fn token_to_id(&self, _token: &str) -> Option<u32> { None }
    fn bos_token_id(&self) -> Option<u32> { None }
    fn eos_token_id(&self) -> Option<u32> { None }
}
```

**Parameters**:
- `add_bos: bool` - Whether to prepend BOS token to output
- `add_special: bool` - Whether to parse special tokens (e.g., `<|eot_id|>` in LLaMA-3)

### 3.2 Tokenization Entry Point (`crates/bitnet-cli/src/main.rs:1110-1112`)

**Location**: `/crates/bitnet-cli/src/main.rs:1110`
```rust
// Tokenize formatted prompt with proper BOS policy and special token parsing
let parse_special = template_type.parse_special();
let mut tokens = tokenizer.encode(&formatted_prompt, bos_policy, parse_special)?;
```

**Special Token Parsing** (`crates/bitnet-inference/src/prompt_template.rs:221-225`):
```rust
pub fn parse_special(&self) -> bool {
    matches!(self, Self::Llama3Chat)  // Only LLaMA-3 has special tokens to parse
}
```

## 4. Token Output (--dump-ids)

### 4.1 Token Printing Location (`crates/bitnet-cli/src/main.rs:1438-1441`)

**CLI Flag**: `--dump-ids` (boolean)
- **Location in args**: `/crates/bitnet-cli/src/main.rs:252-254`

**Output Implementation** (`crates/bitnet-cli/src/main.rs:1438-1441`):
```rust
// Dump IDs if requested
if dump_ids {
    println!("Token IDs: {:?}", generated_tokens);
}
```

**Current Behavior**:
- Prints generated tokens only (not including prompt tokens)
- Output format: `Token IDs: [1234, 5678, 9012, ...]`

### 4.2 Enhanced Tokenize Subcommand (`crates/bitnet-cli/src/main.rs:550-551`)

**Handler**: `handle_tokenize_command()`
- **Signature**: `async fn handle_tokenize_command(model_path, tokenizer_path, text, file, bos, json_out)`
- **Location**: `/crates/bitnet-cli/src/main.rs:645-720`

**Features**:
- Standalone tokenization without model inference
- BOS token control via `--bos` flag
- JSON output with token IDs and metadata
- Tokenizer source tracking (external vs. embedded GGUF)

**JSON Output Schema**:
```json
{
  "tokens": {
    "ids": [1234, 5678, 9012],
    "count": 3
  },
  "gen_policy": {
    "bos": true
  },
  "counts": {
    "n_kv": 256,
    "n_tensors": 512,
    "unmapped": 0
  },
  "tokenizer": {
    "type": "sentencepiece",
    "origin": "external" | "embedded",
    "bos": 1,
    "eos": 2
  }
}
```

## 5. Shared Reusable Code for xtask Integration

### 5.1 Core Reusable Components

#### 5.1.1 Tokenizer Loading (`crates/bitnet-tokenizers/src/lib.rs:423-435`)

**Public Function**: `TokenizerBuilder::from_file()`
- **Signature**: `pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Arc<dyn Tokenizer>>`
- **Location**: `/crates/bitnet-tokenizers/src/lib.rs:424`
- **Usage in CLI**: Load external tokenizer files (*.json, *.model)

**Example for xtask**:
```rust
use bitnet_tokenizers::TokenizerBuilder;
let tokenizer = TokenizerBuilder::from_file("models/tokenizer.json")?;
```

#### 5.1.2 GGUF Tokenizer Loading (`crates/bitnet-tokenizers/src/lib.rs:473-477`)

**Public Function**: `TokenizerBuilder::from_gguf_reader()`
- **Signature**: `pub fn from_gguf_reader(reader: &GgufReader) -> Result<Arc<dyn Tokenizer>>`
- **Location**: `/crates/bitnet-tokenizers/src/lib.rs:473`
- **Purpose**: Extract tokenizer embedded in GGUF models

**Example for xtask**:
```rust
use bitnet_models::GgufReader;
use bitnet_tokenizers::TokenizerBuilder;

let gguf_data = std::fs::read("model.gguf")?;
let reader = GgufReader::new(&gguf_data)?;
let tokenizer = TokenizerBuilder::from_gguf_reader(&reader)?;
```

### 5.2 Prompt Template (`crates/bitnet-inference/src/prompt_template.rs`)

**Public API**:

#### 5.2.1 Template Parsing
```rust
pub fn from_str(s: &str) -> Result<TemplateType>  // Parse --prompt-template args
```

#### 5.2.2 Template Application
```rust
pub fn apply(&self, user_text: &str, system_prompt: Option<&str>) -> String
```

#### 5.2.3 BOS Policy
```rust
pub fn should_add_bos(&self) -> bool
pub fn parse_special(&self) -> bool
```

#### 5.2.4 Stop Sequences
```rust
pub fn default_stop_sequences(&self) -> Vec<String>
pub fn resolve_stop_token_ids(&self, tokenizer: &dyn Tokenizer) -> Vec<u32>
```

**Example for xtask (e.g., per-token tracing)**:
```rust
use bitnet_inference::TemplateType;

let template = TemplateType::Instruct;
let formatted = template.apply("What is 2+2?", None);
let should_bos = template.should_add_bos();
let stop_ids = template.resolve_stop_token_ids(&tokenizer);
```

### 5.3 Complete Tokenization Pipeline for xtask

**Composable Pattern** (based on CLI main.rs:1110-1112):

```rust
// Step 1: Resolve template
let template = TemplateType::Instruct;  // or parse from CLI arg

// Step 2: Format prompt
let formatted = template.apply("What is 2+2?", None);

// Step 3: Determine BOS policy
let bos = template.should_add_bos();
let parse_special = template.parse_special();

// Step 4: Load tokenizer
let tokenizer = TokenizerBuilder::from_file("tokenizer.json")?;

// Step 5: Tokenize
let tokens = tokenizer.encode(&formatted, bos, parse_special)?;

// Step 6: Get stop sequences
let stop_ids = template.resolve_stop_token_ids(&tokenizer.as_ref());
```

## 6. Integration Points in xtask

### 6.1 Current xtask Tokenizer Usage

**Location**: `xtask/src/main.rs`
- **Tokenizer Module**: `xtask/src/tokenizers.rs` (line 39)
- **Subcommands**: 
  - `Tokenizer { output_dir: PathBuf }` (line 269) - Download tokenizer
  - Multiple inference-like commands accept `--tokenizer` paths

### 6.2 Recommended xtask Enhancements

#### 6.2.1 Per-Token Tracing (crossval-per-token)
**Requirement**: Tokenize prompts with template/BOS handling before passing to C++ reference
**Reusable Component**: 
- `TemplateType::apply()` + `tokenizer.encode()`
- Same BOS/special-token logic as CLI

#### 6.2.2 Tokenizer Diagnostics
**Potential Command**: `xtask tokenizer-info <model.gguf> --template instruct --prompt "test"`
**Reusable Code**:
```rust
let template = TemplateType::parse_str("instruct")?;
let bos = template.should_add_bos();
let formatted = template.apply("test", None);
let tokens = tokenizer.encode(&formatted, bos, template.parse_special())?;
println!("Formatted: {:?}", formatted);
println!("Tokens: {:?}", tokens);
```

#### 6.2.3 Batch Tokenization
**Potential Command**: `xtask batch-tokenize <prompts.txt> --tokenizer tok.json --template auto`
**Key Reusable Functions**:
- `TokenizerBuilder::from_file()` or `from_gguf_reader()`
- `TemplateType::detect()` (for auto-detection)
- `tokenizer.encode()` with proper BOS/parse_special handling

## 7. Key Design Principles

### 7.1 Template-Aware BOS
- Templates decide default BOS behavior (not tokenizer)
- `--bos` flag overrides template default in `run` command
- `--no-bos` flag available in `inference` command only

### 7.2 Special Token Parsing
- Only enabled for templates containing special tokens (LLaMA-3)
- Tokenizer implements via `add_special` parameter in `encode()`
- Example: `<|eot_id|>` tokenized as single token vs. character sequence

### 7.3 Tokenizer Abstraction
- Trait-based for different implementations (GGUF, HF, SPM)
- `token_to_id()` provides special token lookup for stop sequence resolution
- Default implementations avoid breaking changes

## 8. File References Summary

| Component | File | Key Functions |
|-----------|------|----------------|
| **Template System** | `crates/bitnet-inference/src/prompt_template.rs` | `TemplateType::apply()`, `should_add_bos()`, `parse_special()` |
| **Tokenizer Trait** | `crates/bitnet-tokenizers/src/lib.rs` | `Tokenizer::encode()`, `TokenizerBuilder::from_*()` |
| **CLI Run Command** | `crates/bitnet-cli/src/main.rs` | `run_simple_generation()` (lines 840-1444) |
| **CLI Inference** | `crates/bitnet-cli/src/commands/inference.rs` | `InferenceCommand::should_add_bos()` (line 1433) |
| **Tokenize Subcommand** | `crates/bitnet-cli/src/main.rs` | `handle_tokenize_command()` (lines 645-720) |
| **xtask Base** | `xtask/src/main.rs` | Token/Tokenizer commands (lines 269, 411, 485) |

## 9. Reuse in xtask - Action Items

### For Per-Token Tracing (crossval-per-token):
1. Import `TemplateType` from `bitnet_inference`
2. Load tokenizer using `TokenizerBuilder::from_file()` or `from_gguf_reader()`
3. Apply template with `template.apply()` → `tokenizer.encode()` pattern
4. Capture tokens before passing to C++ reference for comparison

### For Tokenizer Diagnostics:
1. Reuse `handle_tokenize_command()` logic as library function
2. Export JSON-serializable `TokenizerInfo` struct
3. Make `--template` flag available in xtask commands

### For Batch Operations:
1. Create helper function in common module:
   ```rust
   pub fn tokenize_with_template(
       prompt: &str,
       template: &str,
       tokenizer: &Arc<dyn Tokenizer>,
   ) -> Result<Vec<u32>>
   ```
2. Share between CLI and xtask

