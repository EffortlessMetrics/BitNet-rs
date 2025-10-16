# BitNet.rs Current Architecture Analysis

## Executive Summary

This document provides a comprehensive analysis of the BitNet.rs codebase implementation of prompt templates, chat functionality, tokenization, receipt generation, and streaming inference. It identifies the specific locations requiring changes for the proposed refactoring.

---

## 1. PROMPT TEMPLATE SYSTEM

### Location
- **Primary**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/prompt_template.rs`

### Current Architecture

#### Core Types (Already Exists)
```rust
pub enum ChatRole {
    System,
    User,
    Assistant,
}

pub struct ChatTurn {
    pub role: ChatRole,
    pub text: String,
}

pub enum TemplateType {
    Raw,
    Instruct,
    Llama3Chat,
}

pub struct PromptTemplate {
    template_type: TemplateType,
    system_prompt: Option<String>,
    conversation_history: Vec<(String, String)>,  // Currently (user, assistant) tuples
}
```

#### Key Methods
1. **Auto-Detection**: `TemplateType::detect(tokenizer_name, chat_template_jinja)` (lines 83-109)
   - Priority 1: GGUF `chat_template` metadata (checks for `<|start_header_id|>`, `<|eot_id|>`)
   - Priority 2: Tokenizer family name heuristics
   - Priority 3: Fallback to Raw

2. **Template Application**: 
   - `apply()` (lines 112-118): Single-turn formatting
   - `render_chat()` (lines 189-255): **Multi-turn chat rendering** ✓ Already implemented!
     - Returns properly formatted history as String
     - Supports system prompt
     - All three template types

3. **Stop Sequences**: `default_stop_sequences()` (lines 170-176)
4. **BOS Control**: `should_add_bos()` (lines 180-185)

#### Current Limitations
- `PromptTemplate::conversation_history` uses tuples `Vec<(String, String)>` not `ChatTurn` objects
- `PromptTemplate::format()` (line 289) doesn't use `render_chat()` - only does single-turn
- History management is separate from ChatTurn serialization

---

## 2. CHAT COMMAND IMPLEMENTATION

### Location
- **Primary**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/src/commands/chat.rs`
- **Supporting**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/src/commands/inference.rs` (contains InferenceCommand)

### Current Architecture

#### Main Chat Loop (lines 61-224)
```rust
pub async fn run_chat(&self, config: &CliConfig) -> Result<()> {
    // 1. Setup environment and logging
    // 2. Load model and tokenizer
    // 3. Resolve template type (line 74)
    // 4. REPL loop with commands: /help, /clear, /metrics, /exit
    // 5. Format each prompt with template (line 139)
    // 6. Run streaming inference (line 161)
    // 7. Emit receipts if specified (lines 176-188)
}
```

#### Chat State Management
```rust
// Local to run_chat (line 82)
let mut conversation_history: Vec<(String, String)> = Vec::new();
let mut metrics = ChatMetrics::default();
```

#### Template Formatting (lines 266-329) - DUPLICATE LOGIC ISSUE
```rust
fn format_chat_turn(
    &self,
    template: &TemplateType,
    history: &[(String, String)],
    current_input: &str,
) -> Result<String> {
    // Manual formatting logic DUPLICATED from TemplateType::apply
    // - Llama3Chat: manual token construction (lines 277-303)
    // - Instruct: manual format string building (lines 304-315)
    // - Raw: simple concatenation (lines 316-325)
}
```

#### Streaming Inference Path (lines 227-263)
```rust
async fn run_chat_inference(
    &self,
    engine: &mut InferenceEngine,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<(String, usize)> {
    // 1. Create streaming generator
    let mut stream = engine.generate_stream_with_config(prompt, &engine_config)?;
    
    // 2. Stream tokens to stdout and collect full_response
    while let Some(chunk) = stream.next().await {
        token_count += chunk.token_ids.len();
        full_response.push_str(&chunk.text);
        print!("{}", chunk.text);  // Direct stdout
    }
    
    // 3. Write receipt to ci/inference.json (line 258)
    if let Err(e) = self.write_receipt(engine, token_count).await { ... }
    
    Ok((full_response, token_count))
}
```

#### Receipt Copy (lines 28-41)
```rust
fn copy_receipt_if_present(dir: &Path) -> Result<Option<PathBuf>> {
    // Hardcoded source: Path::new("ci").join("inference.json")
    // Timestamped destination: dir.join(format!("chat-{}.json", ts))
}
```

#### Template Resolution (lines 74, 1423)
- Uses `self.resolve_template_type()` which parses `self.prompt_template` string
- No GGUF metadata reading for auto-detection in chat.rs

---

## 3. TOKENIZER INTEGRATION

### Location
- **Model Loading**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/src/commands/inference.rs` (lines 549-706)
- **Tokenizer Trait**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-tokenizers/src/lib.rs`

### Current Implementation

#### Tokenizer Loading (lines 696-706)
```rust
async fn load_tokenizer(&self, model_path: &Path) -> Result<Arc<dyn Tokenizer>> {
    let tokenizer = bitnet_tokenizers::auto::load_auto(model_path, self.tokenizer.as_deref())?;
    Ok(tokenizer)
}
```

#### BOS/EOS Handling
- **BOS Control** (line 1235): Template-driven decision via `should_add_bos()`
  - LLaMA-3 Chat: `false` (template includes `<|begin_of_text|>`)
  - Raw/Instruct: `true`

- **Applied in Tokenization** (line 982):
  ```rust
  let prompt_ids = tokenizer.encode(&formatted_prompt, self.should_add_bos(), false)?;
  ```

#### In Chat Context
- Tokenization happens AFTER template formatting (line 139 in chat.rs)
- Template already includes BOS tokens for LLaMA-3 (special `<|begin_of_text|>` marker)
- Raw encoding used (no special token handling beyond BOS)

---

## 4. STREAMING INFERENCE ARCHITECTURE

### Streaming Path Location
- **Engine**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/engine.rs` (lines 957-979)
- **Stream Implementation**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/streaming.rs`

### Current Flow

#### Engine Methods (engine.rs)
```rust
pub fn generate_stream(&self, prompt: &str) -> Result<GenerationStream> {
    let config = GenerationConfig::default();
    self.generate_stream_with_config(prompt, &config)
}

pub fn generate_stream_with_config(
    &self,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<GenerationStream> {
    GenerationStream::new(
        self.model.clone(),
        self.tokenizer.clone(),
        self.backend.clone_backend(),
        self.cache.clone(),
        prompt.to_string(),  // ALREADY STRING, NOT TOKENS
        config.clone(),
        streaming_config,
    )
}
```

#### Key Issue: **Tokenization happens INSIDE GenerationStream**
- Prompt is passed as String to GenerationStream::new
- GenerationStream internally handles tokenization
- This is a tight coupling issue

#### GenerationStream Structure (streaming.rs, lines 90-105)
```rust
pub struct GenerationStream {
    receiver: mpsc::Receiver<Result<StreamResponse>>,
    _handle: tokio::task::JoinHandle<()>,
    cancellation_token: Arc<AtomicBool>,
    generation_stats: Arc<GenerationStats>,
}

pub struct GenerationStats {
    pub tokens_generated: AtomicUsize,
    pub errors_encountered: AtomicUsize,
    pub retries_attempted: AtomicUsize,
    pub cancelled: AtomicBool,
}

impl Stream for GenerationStream {
    // Yields StreamResponse with token_ids and text
}
```

### Prefill Integration (engine.rs)

#### New Prefill Method (lines 1015-1066)
```rust
pub async fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
    // Validates token count and vocabulary
    // Runs forward_pass to populate cache
    // Returns timing metrics
}
```

#### Usage in Batch Processing (inference.rs, lines 985-1001)
```rust
// 2. Prefill Phase: Warm model cache with prompt tokens
let t1 = Instant::now();
engine.prefill(&prompt_ids).await?;
let t_prefill_ms = t1.elapsed().as_secs_f64() * 1e3;

// 3. Generation Phase: Generate new tokens using pre-warmed cache
let t2 = Instant::now();
let generated_ids = engine.generate_tokens(&prompt_ids, config).await?;
let t_decode_ms = t2.elapsed().as_secs_f64() * 1e3;
```

---

## 5. RECEIPT GENERATION

### Location
- **Receipt Struct**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/receipts.rs`
- **Receipt Writing**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/src/commands/inference.rs` (lines 832-883)

### Current Implementation

#### Receipt Schema (receipts.rs)
```rust
pub struct InferenceReceipt {
    pub schema_version: String,           // "1.0.0"
    pub timestamp: String,                // ISO 8601
    pub compute_path: String,             // "real" or "mock"
    pub backend: String,                  // "cpu", "cuda", "metal"
    pub kernels: Vec<String>,             // kernel IDs executed
    pub deterministic: bool,              // BITNET_DETERMINISTIC env var
    pub environment: HashMap<String, String>,  // env vars collected
    pub model_info: ModelInfo,            // model metadata
    pub test_results: TestResults,        // test execution summary
    pub performance_baseline: PerformanceBaseline,
    pub cross_validation: Option<CrossValidation>,
    pub corrections: Vec<CorrectionRecord>,
}
```

#### Receipt Generation (inference.rs, lines 832-883)
```rust
pub(super) async fn write_receipt(
    &self,
    _engine: &InferenceEngine,
    tokens_generated: usize,
) -> Result<()> {
    use chrono::Utc;
    use std::fs;

    let backend = self.device.as_deref().unwrap_or("cpu");
    
    // Build receipt JSON
    let kernels = vec![
        "embedding_lookup".to_string(),
        "prefill_forward".to_string(),
        "i2s_gemv".to_string(),
    ];
    
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "timestamp": Utc::now().to_rfc3339(),
        "compute_path": "real",
        "backend": backend,
        "deterministic": self.deterministic || self.greedy,
        "tokens_generated": tokens_generated,
        "kernels": kernels,
        // ...
    });
    
    // Hardcoded path: "ci/inference.json"
    fs::create_dir_all("ci")?;
    fs::write("ci/inference.json", serde_json::to_vec_pretty(&receipt)?)?;
    
    Ok(())
}
```

#### Hardcoded Receipt Path Issues
1. **Source**: `Path::new("ci").join("inference.json")` (chat.rs:31)
2. **Destination**: `emit_receipt_dir.join(format!("chat-{}.json", ts))` (chat.rs:38)
3. **Write Path**: `"ci/inference.json"` hardcoded (inference.rs:878)

#### Per-Turn Receipt Emission (chat.rs, lines 175-188)
```rust
if let Some(dir) = &self.emit_receipt_dir {
    match copy_receipt_if_present(dir) {
        Ok(Some(path)) => {
            debug!("Receipt saved: {}", path.display());
        }
        // ...
    }
}
```

---

## 6. AUTO-DETECTION & TEMPLATE RESOLUTION

### Current Detection Logic

#### In Engine (no current implementation for template reading from GGUF)
- GGUF metadata is available but not read by CLI

#### In PromptTemplate (lines 83-109)
```rust
pub fn detect(tokenizer_name: Option<&str>, chat_template_jinja: Option<&str>) -> Self {
    // Priority 1: GGUF chat_template metadata
    if let Some(jinja) = chat_template_jinja {
        if jinja.contains("<|start_header_id|>") && jinja.contains("<|eot_id|>") {
            return Self::Llama3Chat;
        }
        if jinja.contains("{% for message in messages %}") {
            return Self::Instruct;
        }
    }
    
    // Priority 2: Tokenizer family name heuristics
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

#### Missing: CLI Integration
- Chat command doesn't call `TemplateType::detect()`
- Uses `resolve_template_type()` which only parses CLI flag
- GGUF metadata not read for auto-detection

---

## 7. IDENTIFIED ISSUES & DUPLICATION

### Critical Issues

1. **Template Formatting Duplication**
   - `TemplateType::apply()` - single-turn
   - `TemplateType::render_chat()` - multi-turn ✓ (correct)
   - `InferenceCommand::format_chat_turn()` - manual duplication (WRONG)
   
   **Location**: chat.rs lines 266-329 duplicates TemplateType logic

2. **Conversation History Type Mismatch**
   - PromptTemplate uses `Vec<(String, String)>` tuples
   - Should use `Vec<ChatTurn>` for consistency and serialization
   
   **Locations**: 
   - prompt_template.rs:263
   - chat.rs:82, 173, 289, 311, 319

3. **Hardcoded Receipt Paths**
   - Source hardcoded: `"ci/inference.json"`
   - Write path hardcoded: `"ci/inference.json"`
   
   **Locations**:
   - chat.rs:31
   - inference.rs:878

4. **Missing GGUF Template Auto-Detection**
   - Detection logic exists but not used in CLI
   - No call to `TemplateType::detect()` from chat/inference commands
   
   **Location**: No integration in chat.rs or inference.rs

5. **Tokenization Coupling in Streaming**
   - Prompt passed as String to GenerationStream
   - Tokenization happens inside GenerationStream internals
   - Cannot pre-tokenize for per-turn receipts
   
   **Location**: engine.rs lines 963-978

### Minor Issues

6. **Metrics Struct Inconsistency**
   - ChatMetrics (chat.rs:20-26) - local
   - PerformanceMetrics (inference.rs:396-406) - module-level
   - Both track similar data

7. **Template Type Not Exported**
   - PromptTemplate builder doesn't use multi-turn rendering
   - `format()` method only does single-turn

---

## 8. CURRENT DATA FLOW FOR CHAT INFERENCE

```
User Input
    ↓
format_chat_turn() [WRONG: uses manual formatting]
    ↓
conversation_history (Vec<(String, String)>)
    ↓
template_type.apply() or manual formatting
    ↓
Formatted String Prompt
    ↓
engine.generate_stream_with_config(prompt_string)
    ↓
[INSIDE GenerationStream]
tokenizer.encode(prompt_string)  [PROBLEM: can't separate prefix/new]
    ↓
Token generation loop
    ↓
stream.next() → StreamResponse(token_ids, text)
    ↓
Chat outputs to stdout
    ↓
write_receipt() [HARDCODED: ci/inference.json]
    ↓
copy_receipt_if_present(emit_receipt_dir)
```

---

## 9. FILES REQUIRING CHANGES

### Primary Changes Required

1. **prompt_template.rs** (EXPAND)
   - Update `PromptTemplate::conversation_history` to use `Vec<ChatTurn>`
   - Add methods to work with ChatTurn objects
   - Expose `render_chat()` as primary formatting method
   
2. **chat.rs** (REFACTOR)
   - Replace `format_chat_turn()` with call to `template.render_chat()`
   - Update history storage from `Vec<(String, String)>` to `Vec<ChatTurn>`
   - Add GGUF template auto-detection
   - Fix hardcoded receipt paths
   - Add per-turn receipt metadata

3. **inference.rs** (MINOR)
   - Remove duplicate template formatting logic
   - Fix hardcoded receipt path via constructor argument
   - Export template detection to chat command

4. **receipts.rs** (EXTEND)
   - Add `ChatTurn` serialization fields if needed
   - Support per-turn metadata

### Supporting Files (Reference)

5. **engine.rs** (REFERENCE)
   - `generate_stream_with_config()` - understand tokenization coupling
   - `prefill()` method - already implemented

6. **streaming.rs** (REFERENCE)
   - Understand internal tokenization flow
   - May need tokenization decoupling in future

---

## 10. KEY ABSTRACTIONS ALREADY IN PLACE

✓ **ChatRole** enum (prompt_template.rs:9-16)
✓ **ChatTurn** struct (prompt_template.rs:28-38)
✓ **TemplateType** with auto-detection (prompt_template.rs:42-109)
✓ **render_chat()** multi-turn rendering (prompt_template.rs:189-255)
✓ **Prefill system** (engine.rs:1015-1066)
✓ **InferenceReceipt** struct (receipts.rs:150-189)
✓ **Stop sequences** per template (prompt_template.rs:170-176)
✓ **BOS control** per template (prompt_template.rs:180-185)

---

## Summary Table

| Component | File | Status | Issue |
|-----------|------|--------|-------|
| ChatRole | prompt_template.rs | ✓ Exists | - |
| ChatTurn | prompt_template.rs | ✓ Exists | Unused in PromptTemplate |
| TemplateType | prompt_template.rs | ✓ Exists | - |
| render_chat() | prompt_template.rs | ✓ Exists | Not used by CLI |
| format_chat_turn() | chat.rs | ✓ Works | Duplicates TemplateType logic |
| Chat history storage | chat.rs | ✓ Works | Uses tuples not ChatTurn |
| Auto-detection | prompt_template.rs | ✓ Exists | Not integrated in CLI |
| Receipt writing | inference.rs | ✓ Works | Hardcoded path |
| Per-turn receipts | chat.rs | ✓ Works | Hardcoded path |
| Streaming | engine.rs | ✓ Works | Tokenization internal |
| Prefill | engine.rs | ✓ Works | - |

---

## Conclusion

The BitNet.rs codebase has a solid foundation with:
- ✓ Complete template system with auto-detection
- ✓ ChatTurn and ChatRole abstractions
- ✓ Multi-turn rendering in `render_chat()`
- ✓ Prefill integration in engine

**Main work required**:
1. Eliminate template formatting duplication (use existing abstractions)
2. Update chat history to use ChatTurn instead of tuples
3. Integrate GGUF auto-detection in CLI
4. Refactor hardcoded receipt paths
5. Add per-turn receipt context

All the pieces exist - just need integration and cleanup!
