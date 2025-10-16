# BitNet.rs Architecture Diagrams

Visual representation of current architecture and data flows.

---

## Current Chat Inference Flow

```
User Types in REPL
        ↓
   [read_line()]
        ↓
   Check if command (/help, /clear, /metrics, /exit)
        ↓
   format_chat_turn()
   ├─ Match template type
   ├─ Manually reconstruct format (DUPLICATION!)
   │  ├─ Llama3Chat: manual token building (lines 277-303)
   │  ├─ Instruct: manual format strings (lines 304-315)
   │  └─ Raw: simple concat (lines 316-325)
   └─ Return formatted string
        ↓
   conversation_history.push((user_input, response))
   (stores as Vec<(String, String)> - not ChatTurn!)
        ↓
   engine.generate_stream_with_config(formatted_prompt)
        ├─ Passes String prompt
        └─ → GenerationStream::new()
             ├─ Tokenizes internally (COUPLING!)
             └─ Spawns async token generation task
        ↓
   Stream tokens to stdout
   (while let Some(chunk) = stream.next().await)
        ├─ Print chunk.text
        ├─ Accumulate full_response
        └─ Count tokens
        ↓
   write_receipt() 
   (writes to hardcoded: "ci/inference.json")
        ↓
   copy_receipt_if_present(emit_receipt_dir)
   (hardcoded source: "ci/inference.json")
        ↓
   Add to history and repeat
```

---

## Proposed Refactored Flow

```
User Types in REPL
        ↓
   [read_line()]
        ↓
   Check if command
        ↓
   Auto-detect template type
   ├─ Try GGUF chat_template metadata
   ├─ Try tokenizer hints
   └─ Fallback to CLI flag
        ↓
   Create ChatTurn objects
   └─ ChatTurn { role: ChatRole::User, text: input }
        ↓
   conversation_history.push(turn)
   (stores as Vec<ChatTurn>)
        ↓
   template.render_chat(history, system_prompt)
   (use existing TemplateType::render_chat() method)
   └─ Returns formatted string for current state
        ↓
   tokenizer.encode(formatted_prompt, add_bos)
        ↓
   engine.generate_stream_with_config(prompt_string)
        ↓
   Stream tokens to stdout
        ├─ Print token text
        ├─ Collect response text
        ├─ Count tokens
        └─ Collect per-turn telemetry
        ↓
   Create per-turn receipt
   ├─ Token count
   ├─ Latency
   ├─ Template used
   └─ System prompt (if any)
        ↓
   write_receipt(receipt, parameterized_path)
   (path from --emit-receipt-dir)
        ↓
   Store assistant response as ChatTurn
   └─ ChatTurn { role: ChatRole::Assistant, text: response }
        ↓
   Add to history and repeat
```

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    bitnet-cli (binary)                      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            InferenceCommand struct                   │   │
│  │  (args: prompt, model, template, emit-receipt-dir) │   │
│  └─────────────────────────────────────────────────────┘   │
│         │                           │                       │
│         ├──→ run_single_inference() │                       │
│         │                           │                       │
│         └──→ run_chat()      ← MAIN METHOD                  │
│                  ├─ load_model_and_tokenizer()             │
│                  ├─ resolve_template_type()    ← AUTO-DET! │
│                  ├─ REPL loop                              │
│                  ├─ format_chat_turn()         ← DUPLICATE │
│                  ├─ run_chat_inference()                   │
│                  ├─ write_receipt()            ← HARDCODED │
│                  └─ copy_receipt_if_present()  ← HARDCODED │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────┐
│           bitnet-inference (library)                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           TemplateType (CANONICAL SOURCE)            │  │
│  │                                                       │  │
│  │  - detect()              ← AUTO-DETECTION LOGIC      │  │
│  │  - apply()               ← Single-turn               │  │
│  │  - render_chat()         ← Multi-turn (NOT USED!)    │  │
│  │  - default_stop_seq()                                │  │
│  │  - should_add_bos()                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ChatRole + ChatTurn (ABSTRACTIONS)                  │  │
│  │                                                       │  │
│  │  - ChatRole::System/User/Assistant                   │  │
│  │  - ChatTurn { role, text }                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        InferenceEngine (INFERENCE)                   │  │
│  │                                                       │  │
│  │  - generate_stream_with_config() ← STREAMING         │  │
│  │  - prefill()                      ← NEW              │  │
│  │  - generate_tokens()              ← WITH PREFILL     │  │
│  │  - tokenizer()                    ← TRAIT ACCESS     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      InferenceReceipt (SCHEMA v1.0.0)               │  │
│  │                                                       │  │
│  │  - generate()         ← Create receipt               │  │
│  │  - validate()         ← Check compute_path           │  │
│  │  - save()             ← Write to file                │  │
│  │  - with_test_results() ← Builder pattern             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## History Storage Evolution

### Current (Problematic)

```
conversation_history: Vec<(String, String)>
[
  ("What is 2+2?", "4"),
  ("What about 3+3?", "6"),
]

Issues:
- No role information
- No semantic meaning
- Can't serialize with context
- Type mismatch with ChatTurn
```

### Target (Proper)

```
conversation_history: Vec<ChatTurn>
[
  ChatTurn { role: User,      text: "What is 2+2?" },
  ChatTurn { role: Assistant, text: "4" },
  ChatTurn { role: User,      text: "What about 3+3?" },
  ChatTurn { role: Assistant, text: "6" },
]

Benefits:
- Explicit role information
- System prompt can be separate
- Serializable with context
- Type-safe
- Matches prompt_template.rs abstractions
```

---

## Receipt Path Problem

### Current (Hardcoded)

```
write_receipt() [inference.rs:878]
    ↓
fs::write("ci/inference.json", ...)  ← HARDCODED
    ↓
Used by copy_receipt_if_present() [chat.rs:31]
    ↓
Path::new("ci").join("inference.json")  ← HARDCODED
    ↓
emit_receipt_dir.join(format!("chat-{}.json", ts))
```

### Target (Parameterized)

```
InferenceCommand struct
    ↓
--receipt-dir flag  ← NEW
    ↓
write_receipt(path) ← PARAMETERIZED
    ↓
emit_receipt_dir.join("inference.json")  ← USES FLAG
    ↓
copy_receipt_if_present(emit_receipt_dir)  ← CONSISTENT
```

---

## Template Auto-Detection Flow

### Current (Missing)

```
run_chat()
    ↓
self.resolve_template_type()
    ↓
Parse self.prompt_template string
    ↓
Return TemplateType
    │
    └─ NO GGUF LOOKUP!
       NO TOKENIZER HINTS!
```

### Target (Implemented)

```
run_chat()
    ↓
resolve_template_type() [TRY 1: CLI flag]
    ├─ Returns Some(type) → USE IT
    │
    └─ Returns None → AUTO-DETECT:
       ↓
       Read GGUF metadata [TRY 2: GGUF]
       ├─ Extract chat_template Jinja
       ├─ TemplateType::detect(None, Some(jinja))
       └─ Returns type or falls through
           ↓
           Get tokenizer name [TRY 3: Tokenizer]
           ├─ TemplateType::detect(Some(name), None)
           └─ Returns type or falls through
               ↓
               Fallback: TemplateType::Raw
```

---

## Duplication Problem

### What Exists (Template System)

```
prompt_template.rs:
├─ TemplateType::apply() [line 112-118]
│  └─ Formats single-turn prompt
│
├─ TemplateType::render_chat() [line 189-255]
│  └─ Formats multi-turn chat with history
│     ├─ Llama3Chat: special token formatting
│     ├─ Instruct: Q&A formatting
│     └─ Raw: simple text pass-through
│
├─ TemplateType::apply_llama3_chat() [line 148-167]
│  └─ Single implementation for Llama3 format
│
└─ TemplateType::apply_instruct() [line 121-135]
   └─ Single implementation for Instruct format
```

### What's Duplicated (Wrong!)

```
chat.rs:
└─ InferenceCommand::format_chat_turn() [line 266-329]
   ├─ Llama3Chat formatting [line 277-303] ← SAME AS render_chat!
   ├─ Instruct formatting [line 304-315]   ← SAME AS render_chat!
   └─ Raw formatting [line 316-325]        ← SAME AS render_chat!

PROBLEM: If template format changes, must update TWO places!
```

### Solution

```
Replace ALL of format_chat_turn() with:

template.render_chat(&history, self.system_prompt.as_deref())?

Single source of truth!
```

---

## Tokenization & Prefill Path

### Current

```
User Input
    ↓
format_chat_turn(template, history, input)
    ↓
Formatted String
    ↓
engine.generate_stream_with_config(prompt_string)
    │
    └─→ [INSIDE GenerationStream]
        ├─ tokenizer.encode(prompt_string)  ← CAN'T INTERCEPT
        ├─ Token generation loop
        └─ Stream tokens out
    ↓
Collect response
    ↓
write_receipt()
    └─ No per-turn tokenization context available
```

### Target (Future Improvement)

```
User Input
    ↓
format_chat_turn(template, history, input)
    ↓
tokenizer.encode(formatted_prompt, add_bos)
    ↓
Collect timing: t_tokenize
    ↓
engine.prefill(prompt_ids)  ← NEW
    ├─ Validates tokens
    ├─ Runs forward pass
    └─ Timing: t_prefill
    ↓
engine.generate_tokens(prompt_ids, config)
    ├─ Incremental generation
    └─ Timing: t_decode
    ↓
collect_response()
    ↓
Create per-turn receipt with:
├─ Tokenization time
├─ Prefill time
├─ Decode time
└─ Template used
    ↓
write_receipt(receipt, path)
```

---

## File Dependency Graph

```
┌──────────────────┐
│  chat.rs (CLI)   │
├──────────────────┤
│ Depends on:      │
│ - TemplateType   │──→ ISSUE: duplicates logic
│ - InferenceEngine│
│ - receipts.rs    │──→ ISSUE: hardcoded paths
└──────────────────┘
        │
        ↓
┌─────────────────────────┐
│  inference.rs (CLI)     │
├─────────────────────────┤
│ Depends on:             │
│ - TemplateType          │
│ - InferenceEngine       │
│ - Tokenizer             │
│ - receipts.rs           │──→ ISSUE: hardcoded path
└─────────────────────────┘
        │
        ↓
┌─────────────────────────────┐
│  prompt_template.rs (LIB)   │
├─────────────────────────────┤
│ Core types:                 │
│ - ChatRole enum             │
│ - ChatTurn struct           │
│ - TemplateType enum         │
│ - PromptTemplate struct     │
│                             │
│ Key methods:                │
│ - detect()                  │──→ NOT INTEGRATED
│ - render_chat()             │──→ NOT USED
│ - apply() / apply_*()       │
│ - stop_sequences()          │
│ - should_add_bos()          │
└─────────────────────────────┘
        │
        ↓
┌─────────────────────────────┐
│  engine.rs (LIB)            │
├─────────────────────────────┤
│ Depends on:                 │
│ - streaming.rs              │
│ - GenerationStream          │
│ - prefill()                 │ ✓ READY
│ - generate_tokens()         │ ✓ READY
└─────────────────────────────┘
        │
        ↓
┌─────────────────────────────┐
│  streaming.rs (LIB)         │
├─────────────────────────────┤
│ - GenerationStream struct   │
│ - GenerationStats           │
│ - TokenResponse             │
└─────────────────────────────┘

┌─────────────────────────────┐
│  receipts.rs (LIB)          │
├─────────────────────────────┤
│ - InferenceReceipt struct   │
│ - Receipt schema v1.0.0     │
│ - generate() method         │ ✓ READY
│ - validate() method         │ ✓ READY
│ - save() method             │ ✓ READY
└─────────────────────────────┘
```

---

## Prefill Architecture

```
Prefill Phase
┌─────────────────────────────────────────┐
│ engine.prefill(&prompt_ids)             │
├─────────────────────────────────────────┤
│ 1. Validate token count                 │
│    └─ tokens.len() ≤ max_context        │
│                                         │
│ 2. Validate token values                │
│    └─ all token < vocab_size            │
│                                         │
│ 3. Forward pass (full sequence)         │
│    └─ forward_pass(tokens).await?       │
│       → Populates KV cache              │
│       → Discards logits                 │
│       → Returns timing                  │
│                                         │
│ 4. Optional: Timing measurement         │
│    └─ Records prefill throughput        │
└─────────────────────────────────────────┘
           ↓
Generation Phase
┌─────────────────────────────────────────┐
│ engine.generate_tokens(&prompt_ids)     │
├─────────────────────────────────────────┤
│ 1. First step (step==0):                │
│    └─ forward_pass(full_sequence)       │
│       → Uses pre-warmed cache           │
│       → Fast because KV already there   │
│                                         │
│ 2. Subsequent steps (step>0):           │
│    └─ forward_pass(&[last_token])       │
│       → Only process new token          │
│       → Cache adds new KV entries       │
│       → MUCH faster per-token           │
│                                         │
│ 3. Sampling:                            │
│    └─ Sample next token from logits     │
│    └─ Add to sequence                   │
│    └─ Continue until max_tokens         │
└─────────────────────────────────────────┘

Benefits:
├─ Accurate prefill vs decode timing
├─ Clear separation of concerns
├─ Explicit cache warming
├─ Better metrics for analysis
└─ Enables advanced optimization
```

---

## Summary of Issues & Locations

```
DUPLICATION
├─ format_chat_turn() [chat.rs:266-329]
│  └─ Duplicates: TemplateType::apply/render_chat()
│
STORAGE MISMATCH
├─ conversation_history [chat.rs:82, 173, 289-325]
│  └─ Uses: Vec<(String, String)>
│  └─ Should use: Vec<ChatTurn>
│
HARDCODED PATHS
├─ Source: "ci/inference.json" [chat.rs:31]
├─ Write: "ci/inference.json" [inference.rs:878]
│  └─ Should be: parameterized via CLI
│
MISSING AUTO-DETECTION
├─ Detection exists: TemplateType::detect() [prompt_template.rs:83-109]
├─ Not integrated: chat.rs, inference.rs
│  └─ Should call: with GGUF metadata
│
TOKENIZATION COUPLING
├─ Prompt passed as String to GenerationStream
├─ Tokenization happens internally (can't intercept)
│  └─ Future work: tokenize before passing to stream
```

