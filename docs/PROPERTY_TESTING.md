# Property-Based Testing Framework

## Overview

BitNet.rs includes a comprehensive property-based testing framework designed to ensure correctness and consistency across different execution modes and compared to reference implementations. The framework uses adversarial prompt generation and hard-to-game metrics to catch regressions and ensure behavioral parity.

## Key Features

### 1. Deterministic Greedy Decoding

The CLI now supports deterministic execution for reproducible testing:

```bash
# Force greedy decoding (temperature=0, top_p=1, top_k=0)
bitnet run --model model.gguf --tokenizer tokenizer.json \
  --prompt "Hello" --greedy --seed 42

# Full deterministic mode (single-threaded, deterministic ops)
bitnet run --model model.gguf --tokenizer tokenizer.json \
  --prompt "Hello" --greedy --deterministic --seed 42

# Control thread count explicitly
bitnet run --model model.gguf --tokenizer tokenizer.json \
  --prompt "Hello" --threads 1 --seed 42
```

### 2. Property-Based Tests

Located in `crossval/props/`, the framework includes:

- **Adversarial prompt generation** using Hypothesis
- **Cross-system runners** for BitNet, llama.cpp, and HuggingFace
- **Hard-to-game metrics** including:
  - Token-level Levenshtein distance
  - Prefix match length
  - N-gram F1 scores
  - Combined similarity scoring

### 3. Test Categories

#### Determinism Tests
Verify that BitNet produces identical outputs for the same seed:

```python
# Runs automatically in CI
pytest crossval/props/test_greedy_parity.py::TestGreedyDeterminism
```

#### Cross-System Parity
Compare BitNet against reference implementations:

```bash
# With llama.cpp
LLAMA_BIN=/path/to/main LLAMA_MODEL=model.gguf \
  scripts/prop-greedy-parity.sh

# With HuggingFace
HF_MODEL_ID=1bitLLM/bitnet_b1_58-3B \
  scripts/prop-greedy-parity.sh
```

#### Edge Case Testing
Test handling of adversarial inputs:
- Empty prompts
- Unicode edge cases (ligatures, RTL marks, emoji)
- Very long prompts
- Special characters and escape sequences

#### Regression Testing
Fixed prompts for stability tracking:

```python
REGRESSION_PROMPTS = [
    "What is 2+2?",
    "Complete this: The quick brown",
    "def fibonacci(n):",
    '{"name": "test", "value":',
    "Translate to French: Hello",
    "List three colors:",
]
```

## Running Tests

### Quick Local Test

```bash
# Test determinism only
MODEL_PATH=model.gguf TOKENIZER=tokenizer.json \
  scripts/test-determinism.sh
```

### Full Property Testing

```bash
# Configure thresholds
export PROP_EXAMPLES=50           # Number of test examples
export PROP_MAX_NEW_TOKENS=128    # Max generation length
export PROP_PREFIX_MIN=10         # Min shared prefix tokens
export PROP_BIGRAM_F1_MIN=0.55    # Min bigram F1 score
export PROP_LEV_MAX=60            # Max edit distance

# Run tests
MODEL_PATH=model.gguf TOKENIZER=tokenizer.json \
  scripts/prop-greedy-parity.sh
```

### CI Integration

The framework includes GitHub Actions workflows:

```yaml
# .github/workflows/property-tests.yml
- Runs on every PR affecting inference code
- Tests determinism with fixed seeds
- Runs regression suite
- Uploads failure artifacts for debugging
```

## Metrics Explained

### Levenshtein Distance
Token-level edit distance between outputs. Lower is better.
- Catches word substitutions, insertions, deletions
- Normalized by max sequence length for comparison

### Prefix Match Length
Number of identical tokens at the start of sequences.
- Critical for instruction following
- Early divergence indicates fundamental differences

### N-gram F1 Score
Measures local structure similarity:
- Bigram F1: Adjacent token pairs
- Trigram F1: Three-token sequences
- Robust to minor reordering while catching semantic drift

### Combined Score
Weighted combination of all metrics:
```python
weights = {
    "prefix_match_norm": 0.25,  # Early agreement
    "bigram_f1": 0.20,          # Local structure
    "levenshtein_norm": 0.20,   # Overall distance
    "lcs_norm": 0.15,           # Global structure
    "jaccard": 0.10,            # Vocabulary overlap
    "trigram_f1": 0.10,         # Longer patterns
}
```

## Configuring Thresholds

Thresholds can be tuned based on model and requirements:

```bash
# Strict (for identical models)
export PROP_PREFIX_MIN=20
export PROP_BIGRAM_F1_MIN=0.75
export PROP_LEV_MAX=30

# Relaxed (for different implementations)
export PROP_PREFIX_MIN=5
export PROP_BIGRAM_F1_MIN=0.40
export PROP_LEV_MAX=100
```

## Debugging Failures

When tests fail, artifacts are saved to `test-artifacts/`:

```json
{
  "prompt": "def fibonacci(n):",
  "seed": 42,
  "bitnet": "def fibonacci(n):\n    if n <= 1:\n        return n",
  "llama.cpp": "def fibonacci(n):\n    if n < 2:\n        return n",
  "metrics": {
    "levenshtein": 3,
    "prefix_match": 8,
    "bigram_f1": 0.85
  },
  "failures": ["prefix too short: 8 < 10"]
}
```

To reproduce a failure:

```bash
# Use exact seed and prompt from artifact
bitnet run --model model.gguf --tokenizer tokenizer.json \
  --prompt "def fibonacci(n):" --seed 42 --greedy --deterministic
```

## Adding New Tests

### Custom Prompt Strategies

Add to `crossval/props/strategies.py`:

```python
def s_custom_task():
    """Custom task generator."""
    return st.sampled_from([
        "Your prompt 1",
        "Your prompt 2",
    ])

# Add to main strategy
def prompt_strategy():
    base_task = st.one_of(
        s_json_task(),
        s_code_task(),
        s_custom_task(),  # Add here
    )
```

### Custom Metrics

Add to `crossval/props/metrics.py`:

```python
def custom_metric(a: str, b: str) -> float:
    """Your custom similarity metric."""
    # Implementation
    return score

# Add to basic_text_metrics
def basic_text_metrics(a: str, b: str) -> Dict[str, float]:
    metrics = {
        # ... existing metrics ...
        "custom": custom_metric(a, b),
    }
```

## Performance Considerations

- **Deterministic mode** is slower (single-threaded)
- **Property tests** generate many examples - tune `PROP_EXAMPLES`
- **Cross-system tests** require both systems loaded - use sparingly
- **CI tests** use reduced examples to stay under timeout

## Future Enhancements

Planned improvements:

1. **Logit-level comparison**: Compare top-k token probabilities
2. **Perplexity parity**: Match model perplexity on reference texts
3. **Semantic similarity**: Use embeddings for semantic comparison
4. **Fuzzing strategies**: More adversarial prompt generation
5. **Performance regression**: Track inference speed over time

## Troubleshooting

### "BitNet not deterministic"
- Ensure `--deterministic` flag is used
- Check environment variables are set
- Verify single-threaded execution

### "Prefix too short"
- Model genuinely diverges early
- Check tokenization matches
- Verify model weights loaded correctly

### "Tests timeout"
- Reduce `PROP_EXAMPLES`
- Decrease `PROP_MAX_NEW_TOKENS`
- Use `--threads` to limit parallelism

### "Module not found"
- Install Python dependencies: `pip install hypothesis pytest numpy scipy`
- Check Python path includes project root
