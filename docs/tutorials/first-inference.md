# Your First Inference

This tutorial walks you through loading a BitNet GGUF model and generating your first tokens. It assumes you have completed the [Getting Started guide](../getting-started.md) and have a model downloaded.

**Time:** ~5 minutes  
**Prerequisites:** Rust toolchain, model downloaded to `models/`

---

## 1. Verify the model is ready

```bash
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  compat-check models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
```

Expected output includes `✓ format`, `✓ quantization`, `✓ vocab_size`.

---

## 2. Run a simple inference

```bash
RUST_LOG=warn \
  cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 8 \
  --temperature 0.0 --greedy
```

`RUST_LOG=warn` suppresses verbose debug output so you only see the generated text.

---

## 3. What happened

The CLI:

1. **Loaded the GGUF** — memory-mapped the model file; parsed metadata and tensor data.
2. **Auto-detected the tokenizer** — found `tokenizer.json` alongside the model.
3. **Selected the backend** — chose CPU with SIMD acceleration based on available features.
4. **Applied the prompt template** — wrapped your prompt with the instruct template (`Q: ... A:`).
5. **Generated tokens** — ran the autoregressive decode loop with greedy sampling.

The startup log (visible with `RUST_LOG=info`) shows:

```
INFO backend selected  backend_selection=requested=auto detected=[cpu-rust] selected=cpu-rust
INFO loaded model       tensors=xxx vocab=32000 layers=18
```

---

## 4. Adjust sampling

```bash
# Creative with temperature (stochastic):
RUST_LOG=warn cargo run -p bitnet-cli ... run \
  --prompt "Write a haiku about Rust" \
  --max-tokens 32 \
  --temperature 0.8 --top-p 0.95

# Deterministic with seed (reproducible):
RUST_LOG=warn cargo run -p bitnet-cli ... run \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --temperature 0.0 --greedy --seed 42
```

---

## 5. Try interactive chat

```bash
RUST_LOG=warn \
  cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json
```

Type your question, press Enter. Commands: `/help`, `/clear`, `/metrics`, `/exit`.

---

## 6. Performance notes

The QK256 scalar kernels run at ~0.1 tok/s on 2B models. For quick validation, use `--max-tokens 4-16`. For faster inference, use the BitNet32-F16 format:

```bash
# Check if BitNet32 format is available:
ls models/microsoft-bitnet-b1.58-2B-4T-gguf/
# Look for a non-i2_s.gguf file
```

See [How-to: use-qk256-models](../howto/use-qk256-models.md) for details on format differences.

---

## Next steps

- [How-to: validate-models](../howto/validate-models.md) — inspect and validate model quality
- [How-to: cross-validate](../howto/parity-playbook.md) — compare Rust vs C++ numeric parity
- [Explanation: quantization formats](../explanation/i2s-dual-flavor.md) — understand I2_S/QK256
- [Reference: CLI](../reference/api-reference.md) — full flag reference
