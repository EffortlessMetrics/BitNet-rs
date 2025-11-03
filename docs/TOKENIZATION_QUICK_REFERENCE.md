# Tokenization Quick Reference - File Locations & APIs

## Core File Locations

```
Template System
├── crates/bitnet-inference/src/prompt_template.rs (TemplateType enum + methods)
│   ├── Line 44: pub enum TemplateType { Raw, Instruct, Llama3Chat }
│   ├── Line 83: pub fn detect(...) -> Self
│   ├── Line 112: pub fn apply(&self, user_text: &str, ...) -> String
│   ├── Line 214: pub fn should_add_bos(&self) -> bool
│   ├── Line 223: pub fn parse_special(&self) -> bool
│   └── Line 199: pub fn resolve_stop_token_ids(&self, tokenizer: &dyn Tokenizer) -> Vec<u32>

Tokenizer APIs
├── crates/bitnet-tokenizers/src/lib.rs
│   ├── Line 81-82: pub trait Tokenizer { fn encode(...), fn decode(...), ... }
│   ├── Line 82: fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>>
│   ├── Line 424: TokenizerBuilder::from_file(path) -> Result<Arc<dyn Tokenizer>>
│   └── Line 473: TokenizerBuilder::from_gguf_reader(reader) -> Result<Arc<dyn Tokenizer>>

CLI Integration (Run Command)
├── crates/bitnet-cli/src/main.rs
│   ├── Line 1078: let formatted_prompt = template_type.apply(...)
│   ├── Line 1103-1108: BOS policy determination
│   ├── Line 1110-1112: Tokenization entry point
│   ├── Line 1438-1441: --dump-ids output
│   └── Line 645-720: handle_tokenize_command() for --tokenize subcommand

CLI Integration (Inference Command)
├── crates/bitnet-cli/src/commands/inference.rs
│   ├── Line 248: pub no_bos: bool (--no-bos flag)
│   ├── Line 240: pub prompt_template: String (--prompt-template arg)
│   └── Line 1433: fn should_add_bos(&self) -> bool
```

## Essential Function Signatures

### Template Type Operations
```rust
// Parse template from string (e.g., CLI arg)
impl std::str::FromStr for TemplateType  // line 53
let template: TemplateType = "instruct".parse()?;

// Detect template from metadata
pub fn detect(tokenizer_name: Option<&str>, chat_template_jinja: Option<&str>) -> Self
// Priority: GGUF metadata → tokenizer hints → fallback

// Format prompt
pub fn apply(&self, user_text: &str, system_prompt: Option<&str>) -> String

// BOS/special token policy
pub fn should_add_bos(&self) -> bool
pub fn parse_special(&self) -> bool

// Stop sequences
pub fn default_stop_sequences(&self) -> Vec<String>
pub fn resolve_stop_token_ids(&self, tokenizer: &dyn Tokenizer) -> Vec<u32>
```

### Tokenizer Operations
```rust
// Load from file
pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Arc<dyn Tokenizer>>

// Load from GGUF
pub fn from_gguf_reader(reader: &GgufReader) -> Result<Arc<dyn Tokenizer>>

// Encode text to token IDs
fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>>
// add_bos: Whether to prepend BOS token
// add_special: Whether to parse special tokens (e.g., <|eot_id|>)

// Decode tokens to text
fn decode(&self, tokens: &[u32]) -> Result<String>

// Special token lookup
fn token_to_id(&self, token: &str) -> Option<u32>  // e.g., "<|eot_id|>"
fn bos_token_id(&self) -> Option<u32>
fn eos_token_id(&self) -> Option<u32>
```

## BOS Handling by Template

| Template | should_add_bos() | Reason |
|----------|------------------|--------|
| Raw | true | Standard BOS prefix |
| Instruct | true | Standard BOS prefix |
| Llama3Chat | false | Template includes `<|begin_of_text|>` |

## Template Examples

### Raw Template
```rust
Input: "What is 2+2?"
Output: "What is 2+2?" (unchanged)
```

### Instruct Template
```rust
Input: prompt="What is 2+2?", system=None
Output: "Q: What is 2+2?\nA:"

With system prompt:
Output: "System: {sys}\n\nQ: What is 2+2?\nA:"
```

### LLaMA-3 Chat Template
```rust
Input: prompt="What is 2+2?", system=None
Output: "<|begin_of_text|>\
         <|start_header_id|>user<|end_header_id|>\n\n\
         What is 2+2?<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n"

With system prompt:
Output: "<|begin_of_text|>\
         <|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n"
```

## Complete Tokenization Pipeline

```rust
// 1. Resolve template
let template = match template_arg.as_str() {
    "auto" => TemplateType::detect(tokenizer_name, chat_template_jinja),
    s => TemplateType::from_str(s)?,
};

// 2. Format prompt
let formatted = template.apply("What is 2+2?", None);

// 3. Load tokenizer
let tokenizer = TokenizerBuilder::from_file("tokenizer.json")?;

// 4. Encode with template policies
let tokens = tokenizer.encode(
    &formatted,
    template.should_add_bos(),  // BOS handling
    template.parse_special(),   // Special token parsing
)?;

// 5. Get stop sequences
let stop_ids = template.resolve_stop_token_ids(&tokenizer);
```

## CLI Flags Reference

### Run Command (bitnet run)
- `--prompt-template {auto|raw|instruct|llama3-chat}`: Choose template
- `--system-prompt TEXT`: Add system message for chat models
- `--bos`: Explicitly enable BOS (overrides template)
- `--dump-ids`: Print generated token IDs
- `--stop SEQ`: Add manual stop sequence
- `--stop-id ID`: Add manual stop token ID

### Tokenize Command (bitnet tokenize)
- `--model PATH`: Model path (for extracting GGUF tokenizer)
- `--tokenizer PATH`: External tokenizer file
- `--text TEXT`: Text to tokenize (inline)
- `--file PATH`: Read text from file
- `--bos`: Insert BOS token
- `--json-out PATH`: Write JSON output

### Inference Command (bitnet inference)
- `--prompt-template TEMPLATE`: Same as run command
- `--no-bos`: Explicitly disable BOS
- `--system-prompt TEXT`: System message
- `--stop SEQ`: Stop sequences
- `--stop-id ID`: Stop token IDs

## Integration Points for xtask

### Load Tokenizer from File
```rust
use bitnet_tokenizers::TokenizerBuilder;

let tokenizer = TokenizerBuilder::from_file("models/tokenizer.json")?;
```

### Load Tokenizer from GGUF
```rust
use bitnet_models::GgufReader;
use bitnet_tokenizers::TokenizerBuilder;

let gguf_data = std::fs::read("model.gguf")?;
let reader = GgufReader::new(&gguf_data)?;
let tokenizer = TokenizerBuilder::from_gguf_reader(&reader)?;
```

### Apply Template
```rust
use bitnet_inference::TemplateType;

let template = TemplateType::Instruct;
let formatted = template.apply("Your prompt", None);
let tokens = tokenizer.encode(&formatted, template.should_add_bos(), template.parse_special())?;
```

### Auto-Detect Template
```rust
let template = TemplateType::detect(tokenizer_name, chat_template_jinja);
// Uses GGUF metadata + tokenizer hints
```

### Resolve Stop Sequences
```rust
let stop_ids = template.resolve_stop_token_ids(&tokenizer);
// Converts "<|eot_id|>" → token ID 128009 (for example)
```

## Special Tokens

### Token ID Lookup
```rust
// Get token ID for special token string
if let Some(eot_id) = tokenizer.token_to_id("<|eot_id|>") {
    println!("EOT token ID: {}", eot_id);
}
```

### Special Token Parsing
- Enabled for **LLaMA-3 chat template only** via `parse_special()=true`
- Tokenizes `<|token_name|>` as single token vs. character sequence
- Essential for proper special token handling in generation

## Common Mistakes to Avoid

1. **Not checking `should_add_bos()`**
   - LLaMA-3 chat includes BOS in template, don't add it separately

2. **Forgetting `parse_special` flag**
   - LLaMA-3 special tokens must be parsed during encoding
   - Use `template.parse_special()` to determine automatically

3. **Manual stop sequence handling**
   - Use `template.resolve_stop_token_ids()` instead of hardcoded IDs
   - Enables cross-model compatibility

4. **Ignoring template auto-detection**
   - Use `TemplateType::detect()` with GGUF metadata when available
   - More reliable than guessing from file paths

## Output from --dump-ids

```bash
$ cargo run -p bitnet-cli -- run --model model.gguf --prompt "Test" --dump-ids
...
Generated 3 tokens in 100ms (30.0 tok/s)
Token IDs: [1234, 5678, 9012]
```

Note: Currently outputs **generated tokens only**, not including prompt tokens.

For full prompt + generated token IDs, use **bitnet tokenize** command:
```bash
$ cargo run -p bitnet-cli -- tokenize --model model.gguf --text "Test" --bos --json-out out.json
```

## See Also
- Full exploration: `docs/TOKENIZATION_EXPLORATION.md`
- Prompt template tests: `crates/bitnet-inference/tests/qa_template_formatting_fast.rs`
- Tokenizer tests: `crates/bitnet-tokenizers/src/`
- CLI tests: `crates/bitnet-cli/tests/`
