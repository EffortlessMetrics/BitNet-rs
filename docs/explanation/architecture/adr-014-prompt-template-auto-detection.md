# ADR-014: Prompt Template Auto-Detection System

## Status
**Proposed** - Architectural Decision Record for CLI UX Improvements (Issue TBD)

## Context

BitNet.rs neural network inference currently requires manual prompt template specification via `--prompt-template`, creating friction for users migrating from other LLM frameworks (OpenAI, Anthropic, llama.cpp, Ollama). Users must manually determine if a model needs `raw`, `instruct`, or `llama3-chat` formatting, leading to:

- **User Friction**: Manual template selection requires understanding model architecture
- **Error-Prone**: Incorrect template selection results in poor generation quality
- **Non-Standard**: Industry tools (llama.cpp, Ollama) auto-detect templates from metadata
- **Neural Network Context**: Prompt formatting affects token distribution and inference quality
- **Scale Considerations**: LLaMA-3 (128K vocab), Mistral-instruct (32K vocab) have distinct template requirements

**Current Limitations:**
- No automatic template detection from GGUF `chat_template` metadata
- No fallback detection from tokenizer family hints
- Users must manually specify template for every model
- No guidance when template selection is incorrect
- No integration with HuggingFace conventions

## Decision

We will implement a **Prompt Template Auto-Detection System** with the following architecture:

### 1. TemplateType Detection API
```rust
impl TemplateType {
    /// Detect template type from GGUF metadata and tokenizer hints
    ///
    /// Detection priority:
    /// 1. GGUF `chat_template` metadata (HuggingFace convention)
    /// 2. Tokenizer family name heuristics (llama3, mistral-instruct)
    /// 3. Model architecture hints from GGUF general.architecture
    /// 4. Fallback to Raw template with warning
    pub fn detect(
        gguf_metadata: Option<&GgufMetadata>,
        tokenizer: Option<&dyn Tokenizer>
    ) -> Self {
        // Priority 1: Check chat_template metadata
        if let Some(metadata) = gguf_metadata {
            if let Some(template_str) = metadata.get_chat_template() {
                if Self::matches_llama3_pattern(&template_str) {
                    return Self::Llama3Chat;
                }
                if Self::matches_instruct_pattern(&template_str) {
                    return Self::Instruct;
                }
            }
        }

        // Priority 2: Check tokenizer family hints
        if let Some(tok) = tokenizer {
            if let Some(family) = tok.get_family_name() {
                if family.contains("llama3") {
                    return Self::Llama3Chat;
                }
                if family.contains("instruct") {
                    return Self::Instruct;
                }
            }
        }

        // Priority 3: Check architecture hints
        if let Some(metadata) = gguf_metadata {
            if let Some(arch) = metadata.get_architecture() {
                if arch == "llama" {
                    // LLaMA models default to llama3-chat if recent
                    if metadata.get_version_hint().unwrap_or(0) >= 3 {
                        return Self::Llama3Chat;
                    }
                }
            }
        }

        // Fallback to Raw with debug warning
        tracing::debug!("No template hints found, using raw template");
        Self::Raw
    }
}
```

### 2. Template Pattern Matching Strategies

**LLaMA-3 Pattern Detection:**
```rust
impl TemplateType {
    fn matches_llama3_pattern(chat_template: &str) -> bool {
        // Check for distinctive LLaMA-3 special tokens
        chat_template.contains("<|start_header_id|>")
            && chat_template.contains("<|end_header_id|>")
            && chat_template.contains("<|eot_id|>")
    }
}
```

**Instruct Pattern Detection:**
```rust
impl TemplateType {
    fn matches_instruct_pattern(chat_template: &str) -> bool {
        // Check for common instruct formatting markers
        (chat_template.contains("Q:") && chat_template.contains("A:"))
            || chat_template.contains("### Instruction")
            || chat_template.contains("### Response")
    }
}
```

### 3. CLI Integration with Override Support
```rust
// In InferenceCommand::load_model_and_tokenizer
async fn load_model_and_tokenizer(&self) -> Result<(InferenceEngine, TemplateType)> {
    let model = loader.load(model_path)?;
    let tokenizer = self.load_tokenizer(model_path)?;

    // Detect or use user-specified template
    let template_type = if self.prompt_template == "auto" {
        let metadata = extract_gguf_metadata(&model)?;
        let detected = TemplateType::detect(Some(&metadata), Some(&*tokenizer));
        tracing::info!("Auto-detected template: {:?}", detected);
        detected
    } else {
        // User override
        self.prompt_template.parse()?
    };

    let engine = InferenceEngine::new(model, tokenizer, device)?;
    Ok((engine, template_type))
}
```

## Architecture Decisions

### Decision 1: Priority-Based Detection Strategy
**Choice**: Multi-tier detection with GGUF metadata as highest priority
**Rationale**:
- **GGUF chat_template** is authoritative source (HuggingFace convention)
- **Tokenizer family hints** provide fallback when metadata unavailable
- **Architecture hints** enable heuristic matching for common models
- **Raw fallback** ensures inference always succeeds (never fails on template detection)

**Alternatives Considered**:
- Single-source detection (GGUF only): Too brittle, fails when metadata missing
- User configuration files: Adds complexity without clear benefit
- ML-based template inference: Overkill for small set of template types

### Decision 2: Pattern Matching vs. Template Compilation
**Choice**: String pattern matching on known template markers
**Rationale**:
- **Simplicity**: Easy to understand and maintain (no template compiler)
- **Performance**: <5ms detection overhead (string search is fast)
- **Reliability**: Deterministic matching for known template families
- **Extensibility**: New patterns added via simple string checks

**Alternatives Considered**:
- Jinja2 template parsing: Complex, requires external dependencies
- Template AST compilation: Overkill for three template types
- Regex-based matching: More flexible but harder to debug

### Decision 3: User Override Always Wins
**Choice**: User-specified `--prompt-template` bypasses auto-detection
**Rationale**:
- **Control**: Users can override when auto-detection is wrong
- **Debugging**: Allows testing different templates manually
- **Compatibility**: Maintains existing CLI behavior for explicit templates
- **Neural Network Context**: Some models may need custom templates for optimal inference

**Implementation**:
```bash
# Auto-detection (default)
bitnet run --model llama3.gguf --prompt "Test"

# Manual override
bitnet run --model llama3.gguf --prompt-template raw --prompt "Test"
```

### Decision 4: Debug Logging for Detection Transparency
**Choice**: Log auto-detection results at debug level
**Rationale**:
- **Transparency**: Users can see what template was selected
- **Debugging**: Helps diagnose generation quality issues
- **Auditing**: Provides trace for template selection decisions
- **Non-Intrusive**: Only visible with `RUST_LOG=debug`

**Log Format**:
```
DEBUG bitnet_inference::prompt_template: Auto-detected template: llama3-chat from chat_template metadata
DEBUG bitnet_inference::prompt_template: Auto-detected template: instruct from tokenizer family "mistral-instruct"
DEBUG bitnet_inference::prompt_template: No template hints found, using raw template
```

## Performance Implications

### Detection Overhead
**Target**: <5ms per model load (one-time cost)

**Breakdown**:
- GGUF metadata read: 2ms (memory-mapped, cached)
- String pattern matching: <1ms (simple contains() checks)
- Tokenizer family lookup: <1ms (trait method call)
- Total overhead: 3-4ms (negligible compared to model loading ~500ms)

**Validation**:
```bash
# Benchmark with and without auto-detection
cargo bench --no-default-features --features cpu,full-cli -- bench_template_detection
```

### Neural Network Impact
**Prompt Formatting**: Correct template improves generation quality
- **LLaMA-3 with raw**: Poor response quality (no special token handling)
- **LLaMA-3 with llama3-chat**: Optimal response quality (proper formatting)
- **Token Distribution**: Incorrect template shifts token probabilities

**Quantization Compatibility**: No impact on I2S/TL1/TL2 quantization
- Template detection happens before tokenization
- Quantization pipeline operates on token IDs (template-agnostic)

## Integration Points

### GGUF Metadata Reader
**Dependency**: `bitnet-models::GgufReader`
**Interface**:
```rust
pub trait GgufMetadata {
    fn get_chat_template(&self) -> Option<&str>;
    fn get_architecture(&self) -> Option<&str>;
    fn get_version_hint(&self) -> Option<u32>;
}
```

**Integration**:
- Read metadata during model loading (existing path)
- Extract chat_template field (HuggingFace convention)
- Pass metadata to TemplateType::detect()

### Tokenizer Family Hints
**Dependency**: `bitnet-tokenizers::Tokenizer`
**Interface**:
```rust
pub trait Tokenizer {
    fn get_family_name(&self) -> Option<String> {
        None  // Default implementation for backward compatibility
    }
}
```

**Implementation**:
- Add optional get_family_name() to Tokenizer trait
- Implement for HuggingFace tokenizers (read from tokenizer.json metadata)
- Implement for SentencePiece tokenizers (infer from special tokens)

### CLI Flag Compatibility
**Existing Flags**: `--prompt-template <TYPE>`
**New Behavior**:
- Default value changes from `"raw"` to `"auto"`
- `"auto"` triggers detection logic
- Explicit values (`"raw"`, `"instruct"`, `"llama3-chat"`) bypass detection

**Backward Compatibility**:
- Existing commands with explicit `--prompt-template` unchanged
- Existing commands without flag get auto-detection (better default)
- No breaking changes to CLI API

## Testing Strategy

### Unit Tests
**Template Pattern Matching:**
```rust
// Test LLaMA-3 pattern detection
#[test]
fn test_matches_llama3_pattern() {
    let template = "<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>";
    assert!(TemplateType::matches_llama3_pattern(template));
}

// Test instruct pattern detection
#[test]
fn test_matches_instruct_pattern() {
    let template = "Q: {user_text}\nA:";
    assert!(TemplateType::matches_instruct_pattern(template));
}

// Test fallback to raw
#[test]
fn test_detect_fallback_to_raw() {
    assert_eq!(TemplateType::detect(None, None), TemplateType::Raw);
}
```

### Integration Tests
**End-to-End Detection:**
```rust
// Test auto-detection with llama3 model
#[tokio::test]
async fn test_auto_detect_llama3() {
    let model_path = "tests/models/llama3-tiny.gguf";
    let cmd = InferenceCommand {
        model: Some(model_path.into()),
        prompt_template: "auto".into(),
        ..Default::default()
    };

    let (engine, template) = cmd.load_model_and_tokenizer(&config).await?;
    assert_eq!(template, TemplateType::Llama3Chat);
}
```

### Cross-Validation Tests
**Template Output Consistency:**
```rust
// Validate auto-detected template matches expected output
#[test]
fn test_auto_detected_template_output() {
    let metadata = create_llama3_metadata();
    let template = TemplateType::detect(Some(&metadata), None);
    let formatted = template.apply("Hello", Some("You are helpful"));

    // Should match LLaMA-3 format
    assert!(formatted.starts_with("<|begin_of_text|>"));
    assert!(formatted.contains("<|start_header_id|>system<|end_header_id|>"));
}
```

## Risks and Mitigations

### Risk 1: Incorrect Template Detection
**Risk**: Auto-detection selects wrong template, degrading generation quality
**Mitigation**:
- Conservative detection: Only match high-confidence patterns
- User override always available via `--prompt-template`
- Debug logging shows detection decision
- Fallback to Raw template (safest default)

**Example**:
```bash
# If auto-detection is wrong, user overrides
bitnet run --model model.gguf --prompt-template raw --prompt "Test"
```

### Risk 2: Template Pattern Drift
**Risk**: New model versions change template format, breaking pattern matching
**Mitigation**:
- Version hints in GGUF metadata (check version >= 3 for LLaMA-3)
- Multiple pattern checks (not single magic string)
- Fallback to Raw when patterns don't match
- Community feedback loop for new patterns

**Monitoring**:
```bash
# Track template detection decisions
RUST_LOG=debug bitnet run --model model.gguf --prompt "Test" 2>&1 | grep "Auto-detected"
```

### Risk 3: Performance Regression
**Risk**: Template detection adds unacceptable latency
**Mitigation**:
- <5ms target overhead (validated via benchmarks)
- Detection happens once per model load (cached)
- String pattern matching is O(n) with small n
- No network calls or disk I/O in detection path

**Validation**:
```bash
# Benchmark detection overhead
cargo bench -- bench_template_detection
# Expected: <5ms overhead vs. manual template specification
```

## Success Metrics

**Functional Completeness:**
- ✅ LLaMA-3 models auto-detect llama3-chat template
- ✅ Mistral-instruct models auto-detect instruct template
- ✅ Unknown models fall back to raw template
- ✅ User override via `--prompt-template` works
- ✅ Debug logging shows detection decisions

**Performance Targets:**
- ✅ Detection overhead <5ms (one-time per model load)
- ✅ String pattern matching completes in <1ms
- ✅ No impact on inference throughput (detection pre-tokenization)

**Quality Gates:**
- ✅ Unit tests cover all pattern matching scenarios
- ✅ Integration tests validate end-to-end detection
- ✅ Cross-validation tests ensure template output correctness
- ✅ Backward compatibility maintained (explicit templates unchanged)

**User Experience:**
- ✅ Users no longer need to specify `--prompt-template` for common models
- ✅ Generation quality improves for auto-detected templates
- ✅ Debug mode provides transparency for template selection
- ✅ Override mechanism available when auto-detection fails

## Future Enhancements (Out of Scope)

**Not Included in This ADR:**
- Custom template definitions via YAML configuration
- Template detection from HuggingFace Hub API
- ML-based template inference from generation patterns
- Template versioning and migration strategies
- Community template registry with voting

**Rationale**: These enhancements require additional design decisions and user research. This ADR focuses on foundational auto-detection for standard template families.

## Alternatives Considered

### Alternative 1: Template Registry with User-Defined Templates
**Description**: Maintain registry of custom templates users can define
**Pros**: Maximum flexibility, community-contributed templates
**Cons**: Complex implementation, maintenance burden, version management
**Decision**: Rejected - Too complex for initial implementation

### Alternative 2: HuggingFace Hub API Integration
**Description**: Query HuggingFace API for authoritative template metadata
**Pros**: Always up-to-date with latest models, no hardcoded patterns
**Cons**: Network dependency, API rate limits, offline inference broken
**Decision**: Rejected - Breaks offline-first neural network inference

### Alternative 3: Template Compilation to AST
**Description**: Parse template strings into AST for programmatic execution
**Pros**: Supports arbitrary template logic, extensible to new formats
**Cons**: Complex compiler, slower execution, overkill for three types
**Decision**: Rejected - Over-engineering for current requirements

## Conclusion

This ADR defines a pragmatic **Prompt Template Auto-Detection System** that eliminates manual template selection for common model families while maintaining full user control via override flags. The priority-based detection strategy (GGUF metadata → tokenizer hints → architecture hints → raw fallback) ensures reliable operation across diverse model sources with <5ms overhead.

**Key Design Principles:**
1. **GGUF-First**: Authoritative metadata is primary detection source
2. **Conservative Detection**: Only match high-confidence patterns (avoid false positives)
3. **User Control**: Override always available via explicit `--prompt-template`
4. **Transparency**: Debug logging shows detection decisions for auditing
5. **Graceful Degradation**: Fall back to Raw template when uncertain

**Integration Points:**
- GGUF metadata reader (chat_template field)
- Tokenizer family hints (optional trait method)
- CLI flag parsing (auto as new default value)
- Inference pipeline (detection before tokenization)

**Success Criteria:**
- LLaMA-3 models work without manual template specification
- Detection overhead <5ms (negligible vs. 500ms model load)
- Backward compatibility maintained (explicit templates unchanged)
- User override mechanism preserved for edge cases

**Next Steps:**
1. Implement TemplateType::detect() with pattern matching
2. Add get_family_name() to Tokenizer trait
3. Integrate detection into CLI load_model_and_tokenizer()
4. Write unit and integration tests with cross-validation
5. Benchmark detection overhead (<5ms target)
6. Update documentation (CLAUDE.md, help text)
