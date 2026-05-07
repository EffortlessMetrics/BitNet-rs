# Tokenizer Test Fixtures

This directory contains tokenizer fixtures for TokenizerAuthority E2E integration tests.

## Fixtures

- `valid_tokenizer_a.json`: Reference tokenizer copied from `models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json`.
- `valid_tokenizer_b.json`: Byte-identical clone of `valid_tokenizer_a.json`.
- `different_vocab_size.json`: Reference tokenizer with one extra special token in `added_tokens`, used to verify config-hash divergence.
- `corrupted.json`: Malformed JSON produced by truncating `valid_tokenizer_a.json`, used for error handling tests.

## Regeneration

```bash
cargo run -p xtask -- download-model
./scripts/generate_tokenizer_fixtures.sh
```

The script requires `jq`. Tests copy these fixtures to temporary directories before use.
