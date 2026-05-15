# Apple M4 BitNet Eval And Benchmark

This is the next BitNet proof layer after the Apple M4 one-shot `bitnet mac ask`
route and fixed-prompt `bitnet mac bitnet-warm` proof. The goal is to make
BitNet measurable enough for operators before any broader product surface is
enabled.

## Current Boundary

The accepted BitNet artifact is:

- Model id: `microsoft-bitnet-b1.58-2B-4T-i2s`
- Repo: `microsoft/bitnet-b1.58-2B-4T-gguf`
- Revision: `a1f2f1c765812aa8af3f6eda4a313707064bba15`
- File: `ggml-model-i2_s.gguf`
- SHA256: `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`
- Tokenizer repo: `microsoft/bitnet-b1.58-2B-4T`
- Tokenizer revision: `04c3b9ad9361b824064a1f25ea60a8be9599b127`
- Tokenizer file: `tokenizer.json`
- Tokenizer SHA256: `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7`
- Pre-tokenizer authority: `llama-bpe`
- Prompt authority: `bitnetcpp-answer`

The current product surface remains narrow:

- `bitnet mac ask` supports explicit one-shot BitNet asks for the accepted
  artifact and tokenizer.
- `bitnet mac bitnet-warm` supports a fixed-prompt warm proof route.
- BitNet chat and BitNet serve remain disabled.

## Campaign Shape

`M4-BITNET-EVAL-001` adds a deterministic BitNet-specific corpus and dry-runs it
through the existing answer-corpus parser/scoring path. This is a fixture and
tracking PR only. It does not run the model and does not claim runtime accuracy
or performance.

Later work items add:

- BitNet eval/report schema fields for reference-vs-Rust comparison.
- M4 eval receipts for the accepted I2_S artifact.
- One-shot and fixed-warm benchmark receipts.
- Advisory/nightly regression dashboards for BitNet quality and performance.

## Claim Policy

Allowed after the first slice:

```text
A 100-case deterministic BitNet eval corpus exists and parser/scoring dry-run
validation passes.
```

Not allowed after the first slice:

```text
Runtime BitNet accuracy has been measured for this corpus.
BitNet performance has been benchmarked.
BitNet chat works.
BitNet serve works.
Full apple-m4-metal inference works.
QK256, Neural Engine, MPSGraph, MacBook, or broad Apple Silicon claims.
```

Dense SLM eval reports stay separate. They can prove dense Qwen behavior only;
they are not BitNet quality or performance evidence.
