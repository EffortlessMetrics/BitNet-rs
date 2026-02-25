# CLI UX Improvements Specification

## Context

The BitNet.rs CLI currently provides functional inference capabilities but lacks the polish and familiarity expected from modern LLM tools. Users familiar with OpenAI, Anthropic, llama.cpp, and Ollama interfaces encounter unnecessary friction with non-standard flag names, missing subcommand aliases, and manual template selection. This specification defines comprehensive CLI UX improvements to transform the interface from "works" to "pleasant and familiar" while maintaining BitNet's strict receipts-first posture and neural network inference pipeline integrity.

**Current State:**
- Primary token limit flag is `--max-tokens` (correct but lacks backward-compatible aliases)
- `run` subcommand exists but no `generate` alias for clarity
- No interactive `chat` subcommand for REPL-style inference
- Template selection is manual via `--prompt-template` (no auto-detection)
- `--stop` flag lacks common aliases like `--stop-sequence`
- Help text is functional but lacks canonical usage examples

**Target State:**
- Primary flag `--max-tokens` with visible aliases `--max-new-tokens` and `--n-predict` for compatibility
- `generate` alias for `run` subcommand (OpenAI/Anthropic convention)
- Interactive `chat` subcommand with streaming inference and llama3-chat default
- Automatic template detection from GGUF metadata and tokenizer family hints
- `--stop-sequence` and `--stop_sequences` aliases for `--stop`
- Enhanced help text with three canonical examples (deterministic, creative, interactive)
- Full backward compatibility with existing CLI workflows

**Affected Components:**
- CLI: crates/bitnet-cli/src/commands/inference.rs, main.rs
- Inference: crates/bitnet-inference/src/prompt_template.rs
- Templates: crates/bitnet-inference/src/lib.rs (TemplateType exports)
- Documentation: CLAUDE.md, tests/prompts/README.md

## User Stories

### US1: Familiar Flag Names
**As a** user migrating from OpenAI/Anthropic/llama.cpp
**I want** `--max-tokens` to be the primary flag with `--max-new-tokens` and `--n-predict` as visible aliases
**So that** I can use familiar flag names from my existing workflows without learning BitNet-specific conventions

**Business Value:** Reduced onboarding friction for users from other LLM frameworks (OpenAI API, llama.cpp, Ollama)

### US2: Intuitive Subcommand Aliases
**As a** CLI user
**I want** `bitnet generate` to work as an alias for `bitnet run`
**So that** the command name clearly indicates the text generation action

**Business Value:** Improved discoverability and semantic clarity for text generation workflows

### US3: Interactive Chat Mode
**As a** developer experimenting with models
**I want** `bitnet chat` to launch an interactive REPL with streaming inference
**So that** I can have natural back-and-forth conversations without manually formatting prompts

**Business Value:** Enables rapid prototyping and model experimentation with real-time token streaming

### US4: Automatic Template Detection
**As a** user loading a LLaMA-3 model
**I want** the CLI to automatically detect and apply the llama3-chat template
**So that** I don't need to manually specify `--prompt-template llama3-chat` for every invocation

**Business Value:** Eliminates manual template selection for standard model families (LLaMA, Mistral, etc.)

### US5: Enhanced Help Documentation
**As a** first-time user
**I want** `bitnet --help` to show three canonical examples (deterministic Q&A, creative completion, interactive chat)
**So that** I can quickly understand common usage patterns without reading full documentation

**Business Value:** Accelerated learning curve with practical copy-paste examples

## Acceptance Criteria

### AC1: Flag Renaming and Aliasing
**Given** the CLI is invoked with token limit flags
**When** user specifies `--max-tokens`, `--max-new-tokens`, or `--n-predict`
**Then** all three flags work identically with the same behavior
**And** help text shows `--max-tokens` as primary with aliases documented

```bash
# AC1:primary - Primary flag works
bitnet run --model model.gguf --prompt "Test" --max-tokens 16

# AC1:alias1 - Backward-compatible alias works
bitnet run --model model.gguf --prompt "Test" --max-new-tokens 16

# AC1:alias2 - llama.cpp-style alias works
bitnet run --model model.gguf --prompt "Test" --n-predict 16
```

### AC2: Subcommand Alias
**Given** the CLI supports text generation
**When** user invokes `bitnet generate`
**Then** it behaves identically to `bitnet run`
**And** help text shows `generate` as a visible alias

```bash
# AC2:primary - run subcommand works
bitnet run --model model.gguf --prompt "Test" --max-tokens 16

# AC2:alias - generate alias works identically
bitnet generate --model model.gguf --prompt "Test" --max-tokens 16
```

### AC3: Interactive Chat Subcommand
**Given** a model is loaded
**When** user invokes `bitnet chat --model model.gguf`
**Then** an interactive REPL launches with streaming inference
**And** default template is `llama3-chat` unless overridden
**And** special commands work: `/help`, `/clear`, `/exit`, `/metrics`

```bash
# AC3:basic - Chat mode launches
bitnet chat --model model.gguf

# AC3:template - Default template is llama3-chat
# Expected: REPL uses <|start_header_id|>...<|end_header_id|> formatting

# AC3:streaming - Tokens stream in real-time
# Expected: Tokens appear incrementally, not all at once

# AC3:commands - Special commands work
> /help      # Shows available commands
> /clear     # Clears conversation history
> /metrics   # Shows performance stats
> /exit      # Exits REPL
```

### AC4: Automatic Template Detection
**Given** a GGUF model contains `chat_template` metadata or tokenizer JSON has family hints
**When** user runs inference without `--prompt-template`
**Then** CLI automatically detects and applies the correct template
**And** logs the auto-selected template in debug mode

```bash
# AC4:llama3 - Detects llama3-chat from metadata
bitnet run --model llama3-model.gguf --prompt "Test" --max-tokens 16
# Expected debug log: "Auto-detected template: llama3-chat from chat_template metadata"

# AC4:instruct - Detects instruct from family hints
bitnet run --model mistral-instruct.gguf --prompt "Test" --max-tokens 16
# Expected debug log: "Auto-detected template: instruct from tokenizer family"

# AC4:fallback - Falls back to raw if no hints
bitnet run --model unknown-model.gguf --prompt "Test" --max-tokens 16
# Expected debug log: "No template hints found, using raw template"
```

### AC5: Stop Sequence Aliases
**Given** the CLI supports stop sequences
**When** user specifies `--stop`, `--stop-sequence`, or `--stop_sequences`
**Then** all three flags work identically
**And** help text shows `--stop` as primary with aliases documented

```bash
# AC5:primary - Primary flag works
bitnet run --model model.gguf --prompt "Test" --stop "</s>" --stop "\n\n"

# AC5:alias1 - Singular alias works
bitnet run --model model.gguf --prompt "Test" --stop-sequence "</s>"

# AC5:alias2 - Plural alias works
bitnet run --model model.gguf --prompt "Test" --stop_sequences "</s>"
```

### AC6: Enhanced Help Text
**Given** the CLI is invoked with `--help`
**When** user views the help output
**Then** three canonical examples are shown:
1. Deterministic Q&A with greedy decoding
2. Creative completion with nucleus sampling
3. Interactive chat with streaming

```bash
# AC6:help - Help shows canonical examples
bitnet --help
# Expected output includes:
# Examples:
#   # Deterministic Q&A (greedy decoding)
#   bitnet run --model model.gguf --prompt "What is 2+2?" --greedy --max-tokens 16
#
#   # Creative completion (nucleus sampling)
#   bitnet run --model model.gguf --prompt "Once upon a time" --temperature 0.8 --top-p 0.95 --max-tokens 128
#
#   # Interactive chat (streaming)
#   bitnet chat --model model.gguf --stream
```

### AC7: Template Detection Tests
**Given** the `TemplateType::detect()` method is implemented
**When** tests exercise detection with various metadata patterns
**Then** all detection scenarios pass:
- LLaMA-3 models detected from `chat_template` metadata
- Instruct models detected from tokenizer family name
- Unknown models fall back to Raw template

```rust
// AC7:llama3 - Test LLaMA-3 detection
#[test]
fn test_detect_llama3_from_chat_template() {
    let metadata = json!({"chat_template": "<|start_header_id|>"});
    assert_eq!(TemplateType::detect(&metadata), TemplateType::Llama3Chat);
}

// AC7:instruct - Test instruct detection
#[test]
fn test_detect_instruct_from_family() {
    let metadata = json!({"tokenizer": {"family": "mistral-instruct"}});
    assert_eq!(TemplateType::detect(&metadata), TemplateType::Instruct);
}

// AC7:fallback - Test fallback to raw
#[test]
fn test_detect_fallback_to_raw() {
    let metadata = json!({});
    assert_eq!(TemplateType::detect(&metadata), TemplateType::Raw);
}
```

### AC8: CLI Argument Alias Tests
**Given** flag aliases are implemented
**When** tests verify all alias variants
**Then** CLI parses all variants identically

```rust
// AC8:max_tokens - Test --max-tokens aliases
#[test]
fn test_max_tokens_aliases() {
    let cmd1 = parse_args(&["bitnet", "run", "--max-tokens", "16"]);
    let cmd2 = parse_args(&["bitnet", "run", "--max-new-tokens", "16"]);
    let cmd3 = parse_args(&["bitnet", "run", "--n-predict", "16"]);
    assert_eq!(cmd1.max_tokens, 16);
    assert_eq!(cmd2.max_tokens, 16);
    assert_eq!(cmd3.max_tokens, 16);
}

// AC8:stop - Test --stop aliases
#[test]
fn test_stop_aliases() {
    let cmd1 = parse_args(&["bitnet", "run", "--stop", "</s>"]);
    let cmd2 = parse_args(&["bitnet", "run", "--stop-sequence", "</s>"]);
    let cmd3 = parse_args(&["bitnet", "run", "--stop_sequences", "</s>"]);
    assert_eq!(cmd1.stop, vec!["</s>"]);
    assert_eq!(cmd2.stop, vec!["</s>"]);
    assert_eq!(cmd3.stop, vec!["</s>"]);
}
```

### AC9: Documentation Updates
**Given** CLI UX improvements are implemented
**When** CLAUDE.md and tests/prompts/README.md are updated
**Then** all examples use new primary flags and subcommand aliases
**And** chat subcommand is documented with examples
**And** auto-detection behavior is explained

### AC10: Backward Compatibility
**Given** existing CLI workflows use current flags
**When** new aliases are introduced
**Then** all existing commands continue to work without modification
**And** no breaking changes are introduced

## Technical Requirements

### Scope

**Affected Workspace Crates:**
- `bitnet-cli`: Main CLI implementation with flag parsing and subcommand routing
- `bitnet-inference`: TemplateType detection logic and prompt template application

**Pipeline Stages:**
- CLI Argument Parsing: Flag aliases and subcommand routing (pre-model-loading)
- Template Detection: Metadata reading and heuristic matching (model-loading stage)
- Inference Execution: Prompt formatting and generation (inference stage)
- Output Rendering: Streaming tokens and metrics display (post-inference)

### Constraints

**Performance Targets:**
- Template detection overhead: <5ms (one-time metadata read)
- Chat REPL responsiveness: <50ms latency for token display
- Flag parsing overhead: negligible (<1ms)

**Quantization Accuracy:**
- No impact on quantization pipeline (CLI changes only)
- Maintains strict receipt validation (no mock inference)

**GPU/CPU Compatibility:**
- Template detection works identically on CPU and GPU
- Chat streaming uses existing inference pipeline (no device-specific code)

**Device-Aware Error Handling:**
- Template detection failures fall back to Raw template with warning
- Chat mode gracefully handles model loading errors

**Production Reliability:**
- All flag aliases are visible in help text (no hidden flags)
- Template auto-detection can be overridden with `--prompt-template`
- Chat mode respects `--deterministic` and `--seed` flags

### Public Contracts

**CLI Flag Interface:**
```bash
# Primary flag (recommended)
--max-tokens <N>          Maximum number of tokens to generate

# Aliases (backward compatibility)
--max-new-tokens <N>      Alias for --max-tokens (OpenAI-style)
--n-predict <N>           Alias for --max-tokens (llama.cpp-style)

# Stop sequences
--stop <SEQ>              Stop sequence (repeatable)
--stop-sequence <SEQ>     Alias for --stop (singular)
--stop_sequences <SEQ>    Alias for --stop (plural, underscore)
```

**Subcommand Interface:**
```bash
# Run inference (existing)
bitnet run [OPTIONS] --model <PATH> --prompt <TEXT>

# Generate alias (new)
bitnet generate [OPTIONS] --model <PATH> --prompt <TEXT>

# Interactive chat (new)
bitnet chat [OPTIONS] --model <PATH>
  --stream                Enable streaming inference (default: true)
  --prompt-template <T>   Override auto-detected template
  --system-prompt <TEXT>  System prompt for chat context
```

**Template Detection API:**
```rust
impl TemplateType {
    /// Detect template type from GGUF metadata or tokenizer hints
    ///
    /// Detection strategy:
    /// 1. Check `chat_template` GGUF metadata for known patterns
    /// 2. Check tokenizer family name (e.g., "llama3" → Llama3Chat)
    /// 3. Fall back to Raw if no hints available
    pub fn detect(
        gguf_metadata: Option<&GgufMetadata>,
        tokenizer: Option<&dyn Tokenizer>
    ) -> Self;
}
```

### Risks

**Performance Impact:**
- Risk: Template detection metadata read adds latency
- Mitigation: Cache detection result after first inference
- Measurement: Benchmark with/without auto-detection (<5ms target)

**Quantization Accuracy:**
- Risk: Template formatting affects prompt token distribution
- Mitigation: Validate deterministic inference with fixed seed
- Testing: Cross-validation against C++ reference with known templates

**Compatibility:**
- Risk: Flag aliases conflict with future features
- Mitigation: Reserve `--max-*` and `--stop-*` namespace in design docs
- Documentation: Clearly mark aliases as backward-compatible (not primary)

**Inference Pipeline:**
- Risk: Chat mode streaming introduces state management bugs
- Mitigation: Reuse existing `generate_stream_with_config` path (no new code)
- Testing: TDD with `// AC:ID` tags for REPL command handling

## Domain Schemas

### TemplateDetectionMetadata
```rust
/// Metadata extracted for template detection
pub struct TemplateDetectionMetadata {
    /// GGUF chat_template metadata (HuggingFace convention)
    pub chat_template: Option<String>,

    /// Tokenizer family name (e.g., "llama3", "mistral")
    pub tokenizer_family: Option<String>,

    /// Model architecture hint (e.g., "llama", "mistral")
    pub architecture: Option<String>,
}

impl TemplateDetectionMetadata {
    /// Extract metadata from GGUF and tokenizer
    pub fn from_model(
        gguf_metadata: &GgufMetadata,
        tokenizer: &dyn Tokenizer
    ) -> Self;
}
```

### ChatReplState
```rust
/// State management for interactive chat REPL
pub struct ChatReplState {
    /// Conversation history (user message, assistant response)
    pub conversation_history: Vec<(String, String)>,

    /// Template type (auto-detected or user-specified)
    pub template_type: TemplateType,

    /// System prompt for chat context
    pub system_prompt: Option<String>,

    /// Streaming enabled
    pub stream: bool,

    /// Metrics collection enabled
    pub metrics: bool,
}

impl ChatReplState {
    /// Process user input and generate response
    pub async fn process_input(
        &mut self,
        engine: &mut InferenceEngine,
        input: &str,
        config: &GenerationConfig
    ) -> Result<String>;

    /// Handle special commands (/help, /clear, /exit, /metrics)
    pub fn handle_command(&mut self, command: &str) -> Result<Option<String>>;
}
```

## Implementation Roadmap

### Phase 1: Flag Renaming and Aliasing (Foundational)
**Duration:** 1 day
**Dependencies:** None

**Tasks:**
1. Update `InferenceCommand` struct with `#[arg(visible_alias)]` attributes
2. Add `--max-new-tokens` and `--n-predict` as visible aliases for `--max-tokens`
3. Add `--stop-sequence` and `--stop_sequences` as visible aliases for `--stop`
4. Update help text to show primary flag with aliases
5. Write CLI argument alias tests (AC8)

**Deliverables:**
- Updated `crates/bitnet-cli/src/commands/inference.rs`
- Unit tests in `tests/cli_args.rs`
- Help text verification

**Validation:**
```bash
# Test all flag variants
cargo test --no-default-features --features cpu,full-cli -- test_max_tokens_aliases
cargo test --no-default-features --features cpu,full-cli -- test_stop_aliases

# Verify help text
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run --help
```

### Phase 2: Subcommand Alias (Quick Win)
**Duration:** 0.5 days
**Dependencies:** Phase 1 complete

**Tasks:**
1. Add `Commands::Generate` as alias for `Commands::Inference`
2. Update `main.rs` to route `generate` to inference handler
3. Update help text with subcommand examples

**Deliverables:**
- Updated `crates/bitnet-cli/src/main.rs`
- Subcommand routing test

**Validation:**
```bash
# Test generate alias
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- generate \
  --model models/model.gguf --prompt "Test" --max-tokens 16
```

### Phase 3: Template Detection (Core Feature)
**Duration:** 2 days
**Dependencies:** Phase 1 complete

**Tasks:**
1. Implement `TemplateType::detect()` method in `prompt_template.rs`
2. Add GGUF `chat_template` metadata reader
3. Add tokenizer family name heuristics
4. Integrate detection into CLI inference path
5. Write template detection tests (AC7)
6. Add debug logging for auto-detection

**Deliverables:**
- Updated `crates/bitnet-inference/src/prompt_template.rs`
- Detection tests in `crates/bitnet-inference/tests/template_detection.rs`
- Integration with CLI

**Validation:**
```bash
# Test auto-detection with llama3 model
RUST_LOG=debug cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/llama3-model.gguf --prompt "Test" --max-tokens 16
# Expected: "Auto-detected template: llama3-chat"

# Test fallback with unknown model
RUST_LOG=debug cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/unknown-model.gguf --prompt "Test" --max-tokens 16
# Expected: "No template hints found, using raw template"
```

### Phase 4: Interactive Chat Subcommand (High Impact)
**Duration:** 2 days
**Dependencies:** Phase 2, Phase 3 complete

**Tasks:**
1. Create `commands/chat.rs` with `ChatCommand` struct
2. Implement `ChatReplState` for conversation history management
3. Add REPL loop with streaming inference integration
4. Implement special commands (`/help`, `/clear`, `/exit`, `/metrics`)
5. Default to `llama3-chat` template in chat mode
6. Wire chat subcommand into `main.rs`
7. Write chat mode integration tests (AC3)

**Deliverables:**
- New file: `crates/bitnet-cli/src/commands/chat.rs`
- Updated `crates/bitnet-cli/src/main.rs`
- Integration tests for REPL commands

**Validation:**
```bash
# Test chat mode
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/model.gguf --stream

# Expected:
# > Hello
# [streaming response]
# > /help
# [shows available commands]
# > /exit
```

### Phase 5: Enhanced Help Text (Polish)
**Duration:** 0.5 days
**Dependencies:** Phases 1-4 complete

**Tasks:**
1. Update `Cli` struct `long_about` with three canonical examples
2. Update `InferenceCommand` help text with flag descriptions
3. Add `ChatCommand` help text with REPL usage

**Deliverables:**
- Updated help text in `crates/bitnet-cli/src/main.rs`
- Help text verification test

**Validation:**
```bash
# Verify help text
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- --help
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run --help
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat --help
```

### Phase 6: Documentation Updates (Essential)
**Duration:** 1 day
**Dependencies:** Phases 1-5 complete

**Tasks:**
1. Update CLAUDE.md with new primary flags and aliases
2. Add chat subcommand examples to CLAUDE.md
3. Document template auto-detection behavior
4. Update tests/prompts/README.md with new examples
5. Add troubleshooting section for template detection

**Deliverables:**
- Updated CLAUDE.md
- Updated tests/prompts/README.md
- Documentation verification

**Validation:**
```bash
# Verify all documented examples work
# (Manual verification of each example in CLAUDE.md)
```

## Integration Points

### Neural Network Inference Pipeline
**Stage:** Prompt Formatting (Pre-Tokenization)
**Integration:** Template detection runs after model loading, before tokenization
**Data Flow:**
```
Model Loading → Metadata Extraction → Template Detection → Prompt Formatting → Tokenization → Inference
```

**Implementation:**
```rust
// In InferenceCommand::execute()
async fn load_model_and_tokenizer(&self) -> Result<(InferenceEngine, TemplateType)> {
    // 1. Load model (existing)
    let model = loader.load(model_path)?;
    let tokenizer = self.load_tokenizer(model_path)?;

    // 2. Detect template if not specified
    let template_type = if self.prompt_template == "auto" {
        let metadata = extract_metadata(&model, &tokenizer);
        TemplateType::detect(Some(&metadata), Some(&*tokenizer))
    } else {
        self.prompt_template.parse()?
    };

    // 3. Create engine with detected template
    let engine = InferenceEngine::new(model, tokenizer, device)?;
    Ok((engine, template_type))
}
```

### External Dependencies
**GGUF Metadata Reader:**
- Dependency: `bitnet-models::GgufReader`
- Integration: Read `chat_template` and `general.architecture` metadata
- Fallback: If metadata unavailable, use tokenizer hints

**Tokenizer Family Detection:**
- Dependency: `bitnet-tokenizers::Tokenizer`
- Integration: Check tokenizer type and special tokens for family hints
- Fallback: If no hints, use Raw template

**Streaming Inference:**
- Dependency: `bitnet-inference::InferenceEngine::generate_stream_with_config`
- Integration: Chat mode uses existing streaming path
- No new code: Reuse validated streaming implementation

### BitNet.rs Workspace Architecture
**Affected Crates:**
- `bitnet-cli` (primary): Flag parsing, subcommand routing, REPL implementation
- `bitnet-inference` (secondary): Template detection, prompt formatting
- `bitnet-models` (indirect): Metadata extraction from GGUF
- `bitnet-tokenizers` (indirect): Family name hints for detection

**Feature Flags:**
- `full-cli`: Required for inference and chat subcommands
- `cpu` or `gpu`: Required for inference execution
- No new feature flags introduced

## Testing Strategy

### TDD Approach
All tests use `// AC:ID` tags mapping to acceptance criteria:

```rust
// AC1:primary - Primary --max-tokens flag works
#[test]
fn test_max_tokens_primary_flag() {
    let args = parse_args(&["bitnet", "run", "--max-tokens", "16", "--prompt", "Test"]);
    assert_eq!(args.max_tokens, 16);
}

// AC1:alias1 - Backward-compatible --max-new-tokens alias works
#[test]
fn test_max_tokens_alias1() {
    let args = parse_args(&["bitnet", "run", "--max-new-tokens", "16", "--prompt", "Test"]);
    assert_eq!(args.max_tokens, 16);
}

// AC1:alias2 - llama.cpp-style --n-predict alias works
#[test]
fn test_max_tokens_alias2() {
    let args = parse_args(&["bitnet", "run", "--n-predict", "16", "--prompt", "Test"]);
    assert_eq!(args.max_tokens, 16);
}
```

### Unit Tests
**Template Detection:**
```rust
// AC7:llama3 - Test LLaMA-3 detection from chat_template
#[test]
fn test_detect_llama3_from_chat_template() {
    let gguf_metadata = GgufMetadata {
        chat_template: Some("<|start_header_id|>user<|end_header_id|>".into()),
        ..Default::default()
    };
    assert_eq!(TemplateType::detect(Some(&gguf_metadata), None), TemplateType::Llama3Chat);
}

// AC7:instruct - Test instruct detection from tokenizer family
#[test]
fn test_detect_instruct_from_family() {
    let tokenizer = MockTokenizer::new("mistral-instruct");
    assert_eq!(TemplateType::detect(None, Some(&tokenizer)), TemplateType::Instruct);
}

// AC7:fallback - Test fallback to Raw when no hints
#[test]
fn test_detect_fallback_to_raw() {
    assert_eq!(TemplateType::detect(None, None), TemplateType::Raw);
}
```

### Integration Tests
**Chat Mode REPL:**
```rust
// AC3:basic - Chat mode launches and accepts input
#[tokio::test]
async fn test_chat_mode_basic() {
    let mut chat = ChatReplState::new(TemplateType::Llama3Chat);
    let response = chat.process_input(&mut mock_engine(), "Hello", &config).await?;
    assert!(!response.is_empty());
}

// AC3:commands - Special commands work
#[tokio::test]
async fn test_chat_special_commands() {
    let mut chat = ChatReplState::new(TemplateType::Llama3Chat);

    // /help command
    assert!(chat.handle_command("/help")?.is_some());

    // /clear command
    chat.conversation_history.push(("test".into(), "response".into()));
    chat.handle_command("/clear")?;
    assert_eq!(chat.conversation_history.len(), 0);
}
```

### Smoke Tests
**End-to-End CLI Validation:**
```bash
# AC2:alias - Generate alias works identically to run
cargo test --no-default-features --features cpu,full-cli -- test_generate_alias_e2e

# AC4:llama3 - Auto-detection works for llama3 models
cargo test --no-default-features --features cpu,full-cli -- test_auto_detect_llama3_e2e

# AC6:help - Enhanced help text shows canonical examples
cargo test --no-default-features --features cpu,full-cli -- test_help_shows_canonical_examples
```

### Cross-Validation Tests
**Template Output Validation:**
```rust
// Validate template formatting matches expected output
#[test]
fn test_llama3_template_formatting() {
    let template = TemplateType::Llama3Chat;
    let formatted = template.apply("Hello", Some("You are helpful"));

    assert!(formatted.starts_with("<|begin_of_text|>"));
    assert!(formatted.contains("<|start_header_id|>system<|end_header_id|>"));
    assert!(formatted.contains("You are helpful"));
    assert!(formatted.contains("<|start_header_id|>user<|end_header_id|>"));
    assert!(formatted.contains("Hello"));
}
```

### Performance Benchmarks
**Template Detection Overhead:**
```bash
# Benchmark with auto-detection
cargo bench --no-default-features --features cpu,full-cli -- bench_template_detection

# Expected: <5ms overhead for metadata read + heuristic matching
```

**Chat Streaming Latency:**
```bash
# Benchmark REPL responsiveness
cargo bench --no-default-features --features cpu,full-cli -- bench_chat_streaming_latency

# Expected: <50ms token display latency
```

## Success Metrics

**Functional Completeness:**
- ✅ All 10 acceptance criteria pass with `// AC:ID` tagged tests
- ✅ Flag aliases work identically (AC1, AC5)
- ✅ Subcommand aliases route correctly (AC2)
- ✅ Chat mode REPL functional with special commands (AC3)
- ✅ Template auto-detection covers LLaMA-3, instruct, fallback cases (AC4, AC7)
- ✅ Enhanced help text shows three canonical examples (AC6)
- ✅ Documentation updated with new patterns (AC9)
- ✅ Backward compatibility maintained (AC10)

**Performance Targets:**
- ✅ Template detection overhead <5ms (metadata read + heuristics)
- ✅ Chat streaming latency <50ms (token display responsiveness)
- ✅ Flag parsing overhead negligible (<1ms)

**Quality Gates:**
- ✅ All unit tests pass: `cargo test --workspace --no-default-features --features cpu`
- ✅ Clippy clean: `cargo clippy --all-targets --no-default-features --features cpu,full-cli -- -D warnings`
- ✅ Integration tests pass: `cargo test --no-default-features --features cpu,full-cli -- test_chat_mode`
- ✅ Receipt validation unchanged: `cargo run -p xtask -- verify-receipt ci/inference.json`

**User Experience:**
- ✅ Users can copy-paste OpenAI/Anthropic-style `--max-tokens` commands
- ✅ llama.cpp users can use familiar `--n-predict` flag
- ✅ Chat mode provides natural REPL experience with streaming
- ✅ Template auto-detection eliminates manual `--prompt-template` selection
- ✅ Help text provides actionable examples for common use cases

## Dependencies

**External Crates:**
- `clap`: CLI parsing with `visible_alias` attribute support (already used)
- `tokio`: Async runtime for streaming inference (already used)
- `serde_json`: GGUF metadata parsing (already used)

**Internal Crates:**
- `bitnet-models`: GGUF metadata reader for `chat_template` extraction
- `bitnet-tokenizers`: Tokenizer trait for family name detection
- `bitnet-inference`: Streaming inference for chat mode

**Test Models:**
- `tests/models/tiny.gguf`: For deterministic testing
- LLaMA-3 model with `chat_template` metadata: For auto-detection validation
- Mistral-instruct model: For family name heuristics

**CI/CD:**
- No new CI checks required (existing test suite covers all changes)
- Receipt validation continues to enforce honest compute

## Documentation Requirements

### CLAUDE.md Updates
**Section: Inference Usage**
```markdown
### Primary Flags (Recommended)

```bash
# Use --max-tokens for token limits (OpenAI/Anthropic convention)
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --prompt "What is 2+2?" \
  --max-tokens 16 \
  --temperature 0.0

# Aliases for backward compatibility
--max-new-tokens <N>    # OpenAI-style alias
--n-predict <N>         # llama.cpp-style alias
```

### Subcommand Aliases

```bash
# run (existing)
bitnet run --model model.gguf --prompt "Test"

# generate (alias for clarity)
bitnet generate --model model.gguf --prompt "Test"
```

### Interactive Chat Mode

```bash
# Launch interactive REPL with streaming
bitnet chat --model models/model.gguf --stream

# Special commands
> /help      # Show available commands
> /clear     # Clear conversation history
> /metrics   # Show performance stats
> /exit      # Exit REPL
```

### Automatic Template Detection

BitNet.rs automatically detects prompt templates from model metadata:

```bash
# LLaMA-3 models: auto-detects llama3-chat from chat_template metadata
bitnet run --model llama3-model.gguf --prompt "Test"

# Instruct models: auto-detects instruct from tokenizer family hints
bitnet run --model mistral-instruct.gguf --prompt "Test"

# Unknown models: falls back to raw template
bitnet run --model unknown-model.gguf --prompt "Test"

# Manual override
bitnet run --model model.gguf --prompt-template llama3-chat --prompt "Test"
```
```

### tests/prompts/README.md Updates
```markdown
## Usage with New Flags

```bash
# Deterministic testing with primary flags
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 16 \
  --greedy

# Backward-compatible aliases
--max-new-tokens 16    # OpenAI-style
--n-predict 16         # llama.cpp-style

# Interactive chat mode
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/model.gguf \
  --stream
```

## Template Auto-Detection

Template is automatically detected from model metadata:
- LLaMA-3 models → llama3-chat
- Mistral-instruct → instruct
- Unknown → raw

Override with `--prompt-template <type>` if needed.
```

## Future Enhancements (Out of Scope)

**Not Included in This Specification:**
- Multi-turn conversation history serialization (save/load chat sessions)
- Custom template definitions via YAML configuration files
- Template detection from HuggingFace Hub metadata
- Interactive template selection wizard (`bitnet template --interactive`)
- Streaming inference progress bars with ETA
- Token-by-token latency histograms for performance debugging

**Rationale:** These enhancements require additional design decisions and user research. This specification focuses on high-impact UX improvements with clear implementation paths that align with existing BitNet.rs patterns.

## Conclusion

This specification defines comprehensive CLI UX improvements that transform BitNet.rs from a functional tool into a pleasant, familiar interface aligned with industry conventions (OpenAI, Anthropic, llama.cpp, Ollama). The changes maintain strict backward compatibility, preserve receipt-first validation, and integrate seamlessly with the existing neural network inference pipeline.

**Key Deliverables:**
1. Flag aliases (`--max-tokens`, `--max-new-tokens`, `--n-predict`, `--stop-sequence`)
2. Subcommand alias (`generate` for `run`)
3. Interactive chat subcommand with streaming and REPL commands
4. Automatic template detection from GGUF metadata and tokenizer hints
5. Enhanced help text with three canonical examples
6. Comprehensive tests with `// AC:ID` tags for traceability
7. Updated documentation (CLAUDE.md, tests/prompts/README.md)

**Success Criteria:** All 10 acceptance criteria pass, performance targets met (<5ms template detection, <50ms streaming latency), backward compatibility maintained, and documentation updated.

**Implementation Timeline:** 6.5 days across 6 phases (flag aliases → subcommand alias → template detection → chat mode → help text → documentation).

**Impact:** Reduced onboarding friction for users from other LLM frameworks, improved discoverability with intuitive command names, and enhanced developer experience with interactive chat mode and automatic template selection.
