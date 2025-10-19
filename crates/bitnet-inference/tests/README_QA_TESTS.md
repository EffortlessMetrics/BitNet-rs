# Q&A Template Fast Tests

Fast, lightweight regression tests for Q&A template formatting and greedy inference.

## Test Files

### 1. `qa_template_formatting_fast.rs` (bitnet-inference)

**Fast template formatting unit tests** - no model loading required.

**Purpose**: Catch regressions in template behavior and ensure correct formatting.

**Performance**: < 1 second total execution

**Run tests**:
```bash
# Run all template formatting tests
cargo test -p bitnet-inference --test qa_template_formatting_fast

# Run with verbose output
cargo test -p bitnet-inference --test qa_template_formatting_fast -- --nocapture
```

**Test coverage**:
- ✅ Raw template preserves input exactly
- ✅ Instruct template adds Q&A formatting (Q: / A:)
- ✅ LLaMA-3 template adds special tokens (<|begin_of_text|>, <|eot_id|>)
- ✅ System prompt handling for all templates
- ✅ Stop sequences (Raw: none, Instruct: "\n\nQ:", LLaMA-3: "<|eot_id|>")
- ✅ BOS token control (Raw/Instruct: yes, LLaMA-3: no)
- ✅ Special token parsing control (LLaMA-3: yes, others: no)
- ✅ Template type parsing from strings
- ✅ Template auto-detection from GGUF/tokenizer metadata
- ✅ Edge cases: empty input, special characters, multiline input

**Results** (as of 2025-10-18):
```
test result: ok. 25 passed; 0 failed; 0 ignored; 0 measured
```

### 2. `qa_greedy_math_confidence.rs` (bitnet-cli)

**Greedy math end-to-end tests** - requires model loading.

**Purpose**: Validate greedy inference produces correct math answers.

**Performance**: 3-10 seconds per test (depending on model size)

**Status**: Tests marked with `#[ignore]` - run manually or in CI with model.

**Setup**:
```bash
# 1. Download test model
cargo run -p xtask -- download-model

# 2. Build CLI
cargo build -p bitnet-cli --no-default-features --features cpu,full-cli
```

**Run tests**:
```bash
# Run all greedy math tests (requires model)
BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
  cargo test -p bitnet-cli --test qa_greedy_math_confidence -- --ignored

# Run specific test
BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
  cargo test -p bitnet-cli --test qa_greedy_math_confidence test_greedy_math_simple_2plus2 -- --ignored
```

**Manual testing** (if tests are skipped):
```bash
# Simple math (raw template)
BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
  cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  run --model models/model.gguf \
  --prompt-template raw \
  --prompt "2+2=" \
  --max-tokens 4 \
  --greedy

# Q&A format (instruct template)
BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
  cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  run --model models/model.gguf \
  --prompt-template instruct \
  --prompt "What is 2+2?" \
  --max-tokens 16 \
  --greedy
```

**Expected outputs**:
- Simple math: Output contains "4"
- Q&A format: Output contains "4" or "four"
- Determinism: Two runs produce identical output

**Test coverage**:
- ✅ Greedy math simple (2+2=)
- ✅ Greedy Q&A format (What is 2+2?)
- ✅ Deterministic reproducibility
- ✅ Stop sequence behavior

## CI Integration

### Fast tests (always run)
```bash
# Include in CI pipeline
cargo test --workspace --no-default-features --features cpu
# This includes qa_template_formatting_fast.rs automatically
```

### Slow tests (optional - requires model)
```bash
# Run in dedicated CI job with model cache
export BITNET_GGUF=/path/to/cached/model.gguf
cargo test -p bitnet-cli --test qa_greedy_math_confidence -- --ignored
```

## Troubleshooting

### Fast tests fail
```bash
# Check template implementation
cargo test -p bitnet-inference --lib prompt_template

# Run with verbose output
cargo test -p bitnet-inference --test qa_template_formatting_fast -- --nocapture
```

### Greedy tests skipped
```bash
# No model file found
# Solution: Download model or set BITNET_GGUF
cargo run -p xtask -- download-model
# OR
export BITNET_GGUF=/path/to/model.gguf
```

### Greedy tests fail
```bash
# Check CLI binary is built
cargo build -p bitnet-cli --no-default-features --features cpu,full-cli

# Verify determinism settings
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

# Run test with verbose output
cargo test -p bitnet-cli --test qa_greedy_math_confidence test_greedy_math_simple_2plus2 -- --ignored --nocapture
```

### Non-deterministic results
Ensure environment variables are set:
```bash
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
```

## Development Workflow

### Adding new template tests
1. Add test function to `qa_template_formatting_fast.rs`
2. Run: `cargo test -p bitnet-inference --test qa_template_formatting_fast`
3. Verify test passes in < 1 second

### Adding new greedy inference tests
1. Add test function to `qa_greedy_math_confidence.rs`
2. Mark with `#[ignore]` if it requires model
3. Test manually: `cargo test -p bitnet-cli --test qa_greedy_math_confidence <test_name> -- --ignored`
4. Update documentation with expected behavior

## Performance Benchmarks

| Test Suite | Tests | Time | Model Required |
|------------|-------|------|----------------|
| qa_template_formatting_fast | 25 | < 1s | No |
| qa_greedy_math_confidence | 4 | 3-10s | Yes |

## References

- **Spec**: `docs/explanation/cli-ux-improvements-spec.md`
- **Architecture**: `docs/reference/prompt-template-architecture.md`
- **Inference**: `docs/reference/inference-engine-architecture.md`
- **Template module**: `crates/bitnet-inference/src/prompt_template.rs`
- **CLI inference**: `crates/bitnet-cli/src/commands/inference.rs`
