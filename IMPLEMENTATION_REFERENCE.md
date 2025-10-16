# Implementation Reference Guide

Quick lookup for all code locations referenced in the architecture analysis.

## File Structure

```
crates/
├── bitnet-inference/src/
│   ├── prompt_template.rs          ← CORE: Template system
│   ├── engine.rs                   ← CORE: Streaming & prefill
│   ├── streaming.rs                ← STREAMING: Token stream
│   ├── receipts.rs                 ← RECEIPTS: Schema & generation
│   └── lib.rs                      ← Exports
├── bitnet-cli/src/commands/
│   ├── chat.rs                     ← CHAT: REPL & history
│   ├── inference.rs                ← INFERENCE: Template application
│   └── mod.rs                      ← Exports
└── bitnet-tokenizers/src/
    └── lib.rs                      ← Tokenizer trait
```

---

## Key Code Locations

### 1. PROMPT TEMPLATE SYSTEM

**File**: `crates/bitnet-inference/src/prompt_template.rs`

| Component | Lines | Status |
|-----------|-------|--------|
| `ChatRole` enum | 9-16 | ✓ Exists |
| `ChatTurn` struct | 28-38 | ✓ Exists |
| `TemplateType` enum | 42-51 | ✓ Exists |
| `TemplateType::detect()` | 83-109 | ✓ Exists |
| `TemplateType::apply()` | 112-118 | ✓ Single-turn |
| `TemplateType::apply_instruct()` | 121-135 | ✓ Implementation |
| `TemplateType::apply_llama3_chat()` | 148-167 | ✓ Implementation |
| `TemplateType::default_stop_sequences()` | 170-176 | ✓ Implemented |
| `TemplateType::should_add_bos()` | 180-185 | ✓ Implemented |
| `TemplateType::render_chat()` | 189-255 | ✓ Multi-turn |
| `PromptTemplate` struct | 260-265 | ⚠ Uses tuples |
| `PromptTemplate::format()` | 289-293 | ⚠ Single-turn only |

**Tests**: Lines 311-506

### 2. CHAT COMMAND

**File**: `crates/bitnet-cli/src/commands/chat.rs`

| Component | Lines | Issue |
|-----------|-------|-------|
| `ChatMetrics` struct | 20-26 | Local metrics |
| `copy_receipt_if_present()` | 28-41 | ⚠ Hardcoded paths |
| `run_chat()` main | 61-224 | REPL loop |
| Template resolution | 74 | ✓ Works |
| History storage | 82 | ⚠ Uses tuples |
| User prompt loop | 91-135 | ✓ Works |
| Template formatting | 139 | ✓ Works |
| Streaming inference | 161-213 | ✓ Works |
| Receipt emission | 176-188 | ⚠ Hardcoded |
| `run_chat_inference()` | 227-263 | ✓ Works |
| Stream token collection | 238-255 | ✓ Works |
| Receipt writing call | 258 | ✓ Works |
| `format_chat_turn()` | 266-329 | ⚠ DUPLICATES logic |
| Chat help | 332-342 | ✓ Works |
| Chat metrics display | 345-359 | ✓ Works |

**Issues Summary**:
- Line 31: Hardcoded `"ci/inference.json"` as source
- Lines 82, 173, 289-325: Uses `Vec<(String, String)>` tuples
- Lines 266-329: Duplicates `TemplateType` formatting logic
- Line 74: No GGUF auto-detection integration

### 3. INFERENCE COMMAND

**File**: `crates/bitnet-cli/src/commands/inference.rs`

| Component | Lines | Status |
|-----------|-------|--------|
| `InferenceCommand` struct | 79-230 | ✓ Struct |
| Template flag | 187-188 | ✓ Flag |
| `resolve_template_type()` | 1423-1425 | ✓ Works |
| `load_model_and_tokenizer()` | 549-597 | ✓ Works |
| `load_tokenizer()` | 696-706 | ✓ Works |
| `apply_prompt_template()` | 1201-1218 | ✓ Works |
| `should_add_bos()` | 1235-1246 | ✓ Works |
| `get_stop_sequences()` | 1221-1232 | ✓ Works |
| `write_receipt()` | 832-883 | ⚠ Hardcoded |
| Receipt path | 878 | ⚠ `"ci/inference.json"` |
| `process_batch_sequential()` | 966-1042 | ✓ With prefill |
| Prefill call | 992 | ✓ Works |
| `PrefillEngine` trait | 321-358 | ✓ Works |

**Issues Summary**:
- Line 878: Hardcoded `"ci/inference.json"`
- Lines 874-875: Incomplete kernel recording
- No GGUF metadata reading for auto-detection

### 4. ENGINE & STREAMING

**File**: `crates/bitnet-inference/src/engine.rs`

| Component | Lines | Status |
|-----------|-------|--------|
| `generate_stream()` | 957-960 | ✓ Works |
| `generate_stream_with_config()` | 963-979 | ⚠ String prompt |
| `generate_tokens()` | 1069-1099+ | ✓ With prefill |
| `prefill()` | 1015-1066 | ✓ Implemented |

**Issues Summary**:
- Line 975: Prompt passed as String (tokenization inside)
- Prefill already working correctly

**File**: `crates/bitnet-inference/src/streaming.rs`

| Component | Lines | Status |
|-----------|-------|--------|
| `GenerationStream` struct | 90-95 | ✓ Works |
| `GenerationStats` struct | 98-123 | ✓ Works |
| `StreamingConfig` | 25-88 | ✓ Works |

**Issue**: Internal tokenization (line ~200) - tight coupling

### 5. RECEIPTS

**File**: `crates/bitnet-inference/src/receipts.rs`

| Component | Lines | Status |
|-----------|-------|--------|
| `InferenceReceipt` struct | 150-189 | ✓ Complete |
| `generate()` method | 211-230 | ✓ Works |
| `validate()` method | 306-362 | ✓ Works |
| `collect_env_vars()` | 233-267 | ✓ Works |
| `save()` method | 284-288 | ✓ Works |

**Status**: Complete - no changes needed for receipts.rs itself

---

## Duplication Locations

### Template Formatting Duplication

**Version 1** - Canonical (use this):
```
File: prompt_template.rs
- TemplateType::apply() [line 112-118]
- TemplateType::render_chat() [line 189-255]
- TemplateType::apply_llama3_chat() [line 148-167]
- TemplateType::apply_instruct() [line 121-135]
```

**Version 2** - Duplicate (remove):
```
File: chat.rs
- InferenceCommand::format_chat_turn() [line 266-329]
  - Llama3Chat logic [line 277-303]
  - Instruct logic [line 304-315]
  - Raw logic [line 316-325]
```

---

## History Storage Locations

**Current**: `Vec<(String, String)>` tuples

```
File: chat.rs
- Line 82:  let mut conversation_history: Vec<(String, String)> = Vec::new();
- Line 173: conversation_history.push((line.to_string(), response_text));
- Line 289: for (user_msg, assistant_msg) in history {
- Line 311: full_context.push_str(&format!("Q: {}\nA: {}\n\n", user_msg, assistant_msg));
- Line 319: full_context.push_str(user_msg);
```

**Target**: `Vec<ChatTurn>` objects

```
File: prompt_template.rs
- ChatTurn defined: Line 28-38
- ChatTurn::new() method: Line 35-38
- ChatRole enum: Line 9-16
```

---

## Hardcoded Receipt Paths

**Location 1** - Source path (chat.rs:31):
```rust
let src = Path::new("ci").join("inference.json");
```

**Location 2** - Write path (inference.rs:878):
```rust
fs::write("ci/inference.json", serde_json::to_vec_pretty(&receipt)?)?;
```

Both should be parameterized via command-line arguments.

---

## Auto-Detection Missing Integration

**Detection Logic** (implemented, not used):
```
File: prompt_template.rs
- TemplateType::detect() [line 83-109]
```

**Usage Locations** (where to integrate):

1. **chat.rs** (lines 74, 1423):
   ```rust
   // Current: self.resolve_template_type()
   // Should also try: TemplateType::detect(tokenizer_name, gguf_metadata)
   ```

2. **inference.rs** (lines 1423):
   ```rust
   // Current: only parses CLI flag
   // Should also try: read GGUF metadata and call detect()
   ```

---

## Tokenizer Integration Points

**Tokenizer Loading**:
```
File: inference.rs
- load_tokenizer() [line 696-706]
- Uses: bitnet_tokenizers::auto::load_auto()
```

**BOS/EOS Control**:
```
File: inference.rs
- should_add_bos() [line 1235-1246]
- Applied in: line 982 - tokenizer.encode(formatted_prompt, self.should_add_bos(), false)
```

**Template's BOS Decision**:
```
File: prompt_template.rs
- TemplateType::should_add_bos() [line 180-185]
```

---

## Prefill Integration Points

**Prefill Method**:
```
File: engine.rs
- pub async fn prefill(&mut self, tokens: &[u32]) [line 1015-1066]
- Validates token count and vocabulary [line 1024-1044]
- Runs forward_pass to populate cache [line 1049]
- Returns timing metrics
```

**Usage in Batch**:
```
File: inference.rs
- prefill(&prompt_ids).await? [line 992]
- Timing measurement [line 991-993]
- Followed by generate_tokens() [line 1000]
```

**NOT currently used in chat.rs** - could be improved for per-turn measurements

---

## Summary of Changes Required

### MUST CHANGE
1. **chat.rs**: Remove `format_chat_turn()` [lines 266-329]
   - Use `template.render_chat()` instead
   
2. **chat.rs**: Change history from tuples to `Vec<ChatTurn>` [lines 82, 173, 289-325]

3. **Both files**: Parameterize receipt paths
   - chat.rs:31 - source path
   - inference.rs:878 - write path

### SHOULD ADD
4. **chat.rs** (lines 74, 1423): Integrate GGUF auto-detection
   - Read GGUF metadata
   - Call `TemplateType::detect()`

### NICE TO HAVE
5. **chat.rs**: Add prefill measurements for per-turn timing
6. **Both files**: Unify metrics handling (ChatMetrics vs PerformanceMetrics)

---

## Testing Locations

**prompt_template.rs tests**:
- Lines 311-506: Comprehensive template tests
- Includes: render_chat(), detect(), stop sequences, BOS control

**inference.rs tests**:
- Lines 1428-1623: Prefill and mock engine tests
- Lines 1566-1622: Prefill timing tests

**Chat tests**:
- No current unit tests for format_chat_turn()
- Should add tests after refactoring

---

## File Modification Priority

1. **prompt_template.rs** (LOW impact)
   - Update PromptTemplate::conversation_history type
   - Add helper methods for ChatTurn operations

2. **chat.rs** (HIGH impact)
   - Remove format_chat_turn() duplication
   - Update history storage
   - Add auto-detection
   - Parameterize receipt paths

3. **inference.rs** (MEDIUM impact)
   - Parameterize receipt path
   - Add auto-detection integration

All files compile independently, so changes can be made incrementally.
