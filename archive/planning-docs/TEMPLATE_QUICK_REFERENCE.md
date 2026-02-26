# BitNet-rs Prompt Template System - Quick Reference

## Template Types

| Template | Apply Format | BOS | Use Case |
|----------|------|-----|----------|
| **Raw** | Plain text (no formatting) | YES | Completion-style models |
| **Instruct** | Q&A format (System, Q:, A:) | YES | Instruction-tuned (BitNet, Mistral, Phi) |
| **Llama3Chat** | Special tokens (<\|...\|>) | NO | LLaMA-3 models |

## Auto-Detection Priority

1. **GGUF metadata** (`chat_template` key)
   - LLaMA-3: `<|start_header_id|>` + `<|eot_id|>`
   - Instruct: `{% for message in messages %}`

2. **Tokenizer family name**
   - LLaMA-3: `"llama3"` or `"llama-3"`
   - Instruct: `"instruct"` or `"mistral"`

3. **Model/tokenizer path hints** (CLI only)
   - LLaMA-3: `path.contains("llama") && path.contains("3")`
   - BitNet: `path.contains("bitnet")` (→ Instruct for Q&A)

4. **Fallback** → Raw (or Instruct in CLI)

## CLI Flags

### Template Selection
```bash
--prompt-template auto|raw|instruct|llama3-chat     # (default: auto)
--system-prompt "Your system prompt here"           # (optional)
```

### BOS/EOS Control
```bash
--no-bos    # Disable BOS token insertion
--no-eos    # Disable EOS token insertion
```

### Stop Sequences
```bash
--stop "</s>"                   # String stop sequence(s)
--stop-id 128009                # Token ID stop(s)
--stop-string-window 64         # Tail buffer size for string matching
```

## Template Formatting Examples

### Instruct (Q&A)
```
System: You are a helpful assistant

Q: What is 2+2?
A:
```

### LLaMA-3 Chat
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

### Raw
```
2+2=
```

## Stop Sequences by Template

| Template | Default Stops |
|----------|---------------|
| Raw | (none) |
| Instruct | `\n\nQ:`, `\n\nHuman:` |
| Llama3Chat | `<\|eot_id\|>`, `<\|end_of_text\|>` |

## Code Usage Examples

### Library (bitnet-inference)
```rust
use bitnet_inference::TemplateType;

// Detect from metadata
let template = TemplateType::detect(Some("llama3"), None);

// Apply template
let formatted = template.apply("Hello", Some("Be helpful"));

// Get stops
let stops = template.default_stop_sequences();
let stop_ids = template.resolve_stop_token_ids(&tokenizer);

// Check BOS
let add_bos = template.should_add_bos();
```

### CLI (bitnet-cli)
```bash
# Auto-detect (recommended)
cargo run -p bitnet-cli -- run --model model.gguf --qa --prompt "Question?"

# Explicit template
cargo run -p bitnet-cli -- run \
  --model model.gguf \
  --prompt-template llama3-chat \
  --system-prompt "You are helpful" \
  --prompt "Question?" \
  --max-tokens 128

# Raw completion
cargo run -p bitnet-cli -- run \
  --model model.gguf \
  --prompt-template raw \
  --prompt "2+2=" \
  --max-tokens 4

# With custom stops
cargo run -p bitnet-cli -- run \
  --model model.gguf \
  --prompt "Test" \
  --stop "\n\n" \
  --stop-id 128009
```

### Cross-Validation (xtask)
```bash
# Current (no template support yet)
cargo run -p xtask -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "Test" \
  --max-tokens 4

# TODO: When template support added
cargo run -p xtask -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt-template llama3-chat \
  --prompt "Test" \
  --max-tokens 4 \
  --no-bos
```

## Key Files

| File | Purpose |
|------|---------|
| `crates/bitnet-inference/src/prompt_template.rs` | Core template types, detection, application |
| `crates/bitnet-cli/src/commands/inference.rs` | CLI integration, auto-detection logic |
| `crates/bitnet-inference/tests/template_detection.rs` | Auto-detection tests (20+ passing) |
| `crates/bitnet-cli/tests/template_auto_detect.rs` | CLI detection tests |

## BOS Token Handling

```rust
TemplateType::should_add_bos() -> bool {
    match self {
        Raw => true,        // Needs BOS
        Instruct => true,   // Needs BOS
        Llama3Chat => false // Has <|begin_of_text|> built-in
    }
}
```

Override with `--no-bos` flag to disable globally.

## Multi-Turn Chat (render_chat)

For interactive chat, use `render_chat()` instead of `apply()`:

```rust
let history = vec![
    ChatTurn::new(ChatRole::User, "Hello"),
    ChatTurn::new(ChatRole::Assistant, "Hi there!"),
    ChatTurn::new(ChatRole::User, "How are you?"),
];

let prompt = template.render_chat(&history, Some("Be helpful"))?;
```

Each template formats multi-turn history appropriately:
- **Instruct:** Q/A pairs with optional system
- **Llama3Chat:** Full role markers for each turn
- **Raw:** Concatenated with newlines

## Special Token Parsing

Only LLaMA-3 Chat needs `parse_special=true`:

```rust
template.parse_special() -> bool {
    matches!(self, Llama3Chat)
}
```

Pass to tokenizer: `tokenizer.encode(prompt, add_bos, parse_special)`

## Known Gaps

- [ ] `crossval-per-token` command doesn't yet expose template selection
  - Currently hardcoded: `tokenize(prompt, false, false)`
  - Needed: `--prompt-template`, `--system-prompt`, `--no-bos` flags

- [ ] Jinja template rendering not implemented
  - Jinja patterns detected in GGUF metadata but not rendered
  - Currently uses hard-coded templates instead
  - Future enhancement after MVP

- [ ] Tokenizer family name not yet exposed as trait method
  - CLI path detection works but not standardized
  - Future: Add `get_family_name()` to Tokenizer trait

## Next Steps for crossval-per-token

1. Add template CLI flags to `CrossvalPerToken` struct
2. Resolve template type with auto-detection
3. Apply template to prompt before tokenization
4. Pass `add_bos` and `parse_special` to tokenizer
5. Token parity pre-gate already catches mismatches
