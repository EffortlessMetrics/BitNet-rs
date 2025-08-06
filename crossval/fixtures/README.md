# Cross-Validation Test Fixtures

This directory contains small test models and datasets used for cross-validation between the Rust and C++ BitNet implementations.

## Fixture Format

Test fixtures are JSON files that describe:
- Model file location
- Test prompts to use
- Expected outputs (optional)

Example fixture format:

```json
{
  "name": "minimal_test",
  "model_path": "minimal_model.gguf",
  "test_prompts": [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog."
  ],
  "expected_tokens": null
}
```

## Available Fixtures

Currently, no fixtures are included in the repository to keep it lightweight. Fixtures can be generated using:

```bash
cargo xtask gen-fixtures
```

Or created manually following the format above.

## Model Requirements

Test models should be:
- Small (< 100KB preferred, < 1MB maximum)
- In GGUF format
- Compatible with both Rust and C++ implementations
- Deterministic (same input produces same output)

## Usage

Fixtures are automatically discovered by the cross-validation framework:

```rust
use bitnet_crossval::fixtures::TestFixture;

// List all available fixtures
let fixtures = TestFixture::list_available()?;

// Load a specific fixture
let fixture = TestFixture::load("minimal_test")?;
```

## Creating New Fixtures

1. Create a small test model in GGUF format
2. Place it in this directory
3. Create a corresponding JSON file describing the test
4. Add test prompts that exercise different model behaviors
5. Optionally include expected token outputs for validation

## Security Note

Test fixtures should only contain safe, non-sensitive content suitable for automated testing. Do not include:
- Personal information
- Copyrighted content
- Large models that would bloat the repository
- Models with unknown provenance or licensing