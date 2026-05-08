# Python to Rust Migration Example

This example demonstrates converting a Python BitNet-style inference script into a native Rust implementation.

## Overview

The migration shows how to replace common Python patterns with Rust equivalents:

- Dynamic dictionaries become typed Rust structs.
- Runtime exceptions become `Result<T, E>` with an explicit error enum.
- A Python decode loop becomes an owned Rust inference type.
- Ad-hoc timing fields become typed `Duration` and throughput metrics.
- A Python benchmark script becomes a dependency-free Rust benchmark smoke test.

## Layout

```text
python-to-rust/
├── before/
│   ├── inference.py   # Legacy Python inference wrapper
│   └── benchmark.py   # Legacy Python benchmark harness
├── after/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs      # Converted Rust inference wrapper
│       └── benchmark.rs # Converted Rust benchmark harness
└── README.md
```

## Before: Python Implementation

The original Python code stores model state in a dynamic class and returns dictionary-shaped generation results:

```python
class BitNetInference:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def generate(self, prompt: str, max_tokens: int = 100) -> dict[str, Any]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        start_time = time.perf_counter()
        tokens = self.model.tokenize(prompt)
        output_tokens: list[int] = []

        for _ in range(max_tokens):
            logits = self.model.forward(tokens)
            next_token = max(range(len(logits[-1])), key=logits[-1].__getitem__)
            output_tokens.append(next_token)
            tokens.append(next_token)

            if next_token == self.model.eos_token:
                break

        inference_time = time.perf_counter() - start_time
        token_count = len(output_tokens)

        return {
            "text": self.model.detokenize(output_tokens),
            "tokens": token_count,
            "time": inference_time,
            "tokens_per_second": token_count / inference_time if inference_time else 0.0,
        }
```

See `before/inference.py` for the complete legacy example.

## After: Rust Implementation

The Rust version makes model loading, generation configuration, output metrics, and errors explicit:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationConfig {
    pub max_tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationStats {
    pub text: String,
    pub tokens: usize,
    pub elapsed: Duration,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceError {
    EmptyModelPath,
    EmptyPrompt,
    InvalidMaxTokens,
}

impl BitNetInference {
    pub fn generate(
        &self,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<GenerationStats, InferenceError> {
        if prompt.trim().is_empty() {
            return Err(InferenceError::EmptyPrompt);
        }
        if config.max_tokens == 0 {
            return Err(InferenceError::InvalidMaxTokens);
        }

        // Decode loop omitted here; see after/src/main.rs.
        // The full implementation returns GenerationStats on success.
    }
}
```

See `after/src/main.rs` for the complete converted implementation.

## Migration Map

| Python pattern | Rust replacement |
| --- | --- |
| `dict[str, Any]` generation result | `GenerationStats` struct |
| `ValueError` for validation | `InferenceError` enum implementing `std::error::Error` |
| `max_tokens: int = 100` | `GenerationConfig::default()` |
| Mutable Python list token buffers | Owned `Vec<u32>` buffers |
| `time.perf_counter()` float seconds | `Instant` and `Duration` |
| Script-level benchmark | `src/benchmark.rs` binary with tests |

## Running the Examples

Run the legacy Python example:

```bash
python examples/migration/python-to-rust/before/inference.py
```

Run the converted Rust inference example:

```bash
cargo run --manifest-path examples/migration/python-to-rust/after/Cargo.toml --bin inference
```

Run the converted Rust benchmark smoke test:

```bash
cargo run --manifest-path examples/migration/python-to-rust/after/Cargo.toml --bin benchmark
```

Run the Rust tests:

```bash
cargo test --manifest-path examples/migration/python-to-rust/after/Cargo.toml
```

## Key Benefits

- **Type safety:** Callers receive `GenerationStats` instead of a dictionary with string keys.
- **Explicit failures:** Validation errors are exhaustively represented by `InferenceError`.
- **Ownership:** Model state is owned by `BitNetInference`, so cleanup happens automatically.
- **No runtime dependencies:** The converted sample uses only the Rust standard library.
- **Migration validation:** Unit tests cover model loading validation, prompt validation, generation limits, and benchmark output.

## Next Steps for a Real Model

1. Replace the placeholder `BitNetModel::tokenize`, `forward_next_token`, and `detokenize` methods with the production bitnet-rs APIs.
2. Extend `GenerationConfig` with sampling parameters such as temperature, top-p, and top-k.
3. Add integration tests that compare Python and Rust output on fixed prompts.
4. Promote the smoke benchmark to Criterion once the production inference path is wired in.
