# VERIFY

## Toolchain

- Rust `1.90.0` pinned via `rust-toolchain.toml`
- Components: rustfmt, clippy, rust-analyzer

```bash
rustup toolchain install 1.90.0 -c rustfmt -c clippy -c rust-analyzer
cargo +1.90.0 fmt --all -- --check
cargo +1.90.0 clippy -q
```

## Quick Accept

```bash
./scripts/bitnet_accept.sh \
  models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  models/llama3-tokenizer/tokenizer.json
```

Expected:

- I2-S probes: sane magnitudes (no 1e9 explosions)
- `DEBUG_ATTN=1` shows stable layer-0 norms
- Benchmark one-liner prints: `prefill: … ms, decode: … s → X tok/s`
