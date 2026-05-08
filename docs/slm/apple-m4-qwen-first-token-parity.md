# Apple M4 Qwen First-Token Parity

Campaign: `apple-m4-slm-answer`

Work item: `SLM-M4-003`

Status: `first-token parity restored`

## Summary

`SLM-M4-003` localized the Qwen divergence to dense GGUF tensor layout handling. The Qwen ChatML prompt tokenizes to the same IDs as llama.cpp, the Rust CLI prefills the prompt prefix before decoding, Q8_0 token embeddings and projections now match the reference trace, and the final `output.weight` vocabulary projection is loaded as token-major `[vocab, hidden]` instead of being treated as an already-transposed `[hidden, vocab]` matrix.

After that fix, the Rust CLI first generated token matches the llama.cpp reference newline token `198`, and the 16-token Apple M4 CPU/NEON smoke answer is coherent:

```text
2+2 equals 4.<|im_end|>
```

This evidence uses the Q8_0 companion artifact only as a diagnostic target because the accepted Q4_K_M artifact from `SLM-M4-002` is still unsupported by strict Rust GGUF loading. The Q8_0 companion is not accepted for operator use by this note.

## Fixed Surface

The exact Qwen2.5 ChatML prompt ends with the assistant role token and no trailing newline:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2? Answer briefly.<|im_end|>
<|im_start|>assistant
```

Both Rust and `llama-tokenize` produce:

```text
[151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 21806, 26753, 13, 151645, 198, 151644, 77091]
```

## Reference

```bash
/Users/steven/.cache/bitnet_cpp/build/bin/llama-cli \
  -m target/apple-m4-slm-answer/SLM-M4-003/candidates/qwen2_5_0_5b_q8_0/qwen2.5-0.5b-instruct-q8_0.gguf \
  -f target/apple-m4-slm-answer/SLM-M4-003/qwen_chatml_prompt.txt \
  -n 4 \
  --no-display-prompt \
  --temp 0 \
  --top-k 1 \
  -ngl 0
```

Reference continuation:

```text
2+2
```

The first generated token is newline token `198`.

## Rust

```bash
RUST_LOG=warn cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- --device apple-m4-cpu-neon \
  run \
  --model target/apple-m4-slm-answer/SLM-M4-003/candidates/qwen2_5_0_5b_q8_0/qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "What is 2+2? Answer briefly." \
  --max-new-tokens 1 \
  --temperature 0 \
  --prompt-template qwen2.5 \
  --greedy \
  --deterministic \
  --strict-loader \
  --strict-tokenizer \
  --dump-logit-steps 1 \
  --logits-topk 10 \
  --assert-greedy \
  --json-out target/apple-m4-slm-answer/SLM-M4-003/qwen-rust-trace-answer.json
```

Rust selected `apple-m4-cpu-neon` with `fallback_used=false`, and the first generated token is now `198`.

Top Rust logits:

| Rank | Token ID | Logit |
|---:|---:|---:|
| 1 | 198 | 23.761173248291016 |
| 2 | 271 | 16.088563919067383 |
| 3 | 715 | 14.38894271850586 |
| 4 | 576 | 13.5435791015625 |
| 5 | 2303 | 13.521736145019531 |

## Rust Answer Smoke

```bash
RUST_LOG=warn cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- --device apple-m4-cpu-neon \
  run \
  --model target/apple-m4-slm-answer/SLM-M4-003/candidates/qwen2_5_0_5b_q8_0/qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "What is 2+2? Answer briefly." \
  --max-new-tokens 16 \
  --temperature 0 \
  --top-k 1 \
  --prompt-template qwen2.5 \
  --greedy \
  --deterministic \
  --strict-loader \
  --strict-tokenizer \
  --assert-greedy \
  --json-out target/apple-m4-slm-answer/SLM-M4-003/qwen-rust-answer.json
```

Receipt excerpt:

```json
{
  "requested_backend": "apple-m4-cpu-neon",
  "selected_backend": "apple-m4-cpu-neon",
  "fallback_used": false,
  "generated_ids": [198, 17, 10, 17, 16819, 220, 19, 13, 151645],
  "text": "\\n2+2 equals 4.<|im_end|>"
}
```

## Next Debug Targets

The debug trace localized these issues without adding speculative architecture changes:

- Qwen2.5 ChatML must not append a newline after `<|im_start|>assistant`; the extra newline token changes the decode boundary.
- The Rust generation path must prefill the prompt prefix before decoding the first new token.
- GGUF Q8_0 token embeddings and `output.weight` use `[hidden, vocab]` dims but token-major storage; Rust must reshape those tensors to `[vocab, hidden]` without transposing values.
- GGUF Q8_0 attention and MLP projections use `[in, out]` dims with out-major storage; Rust reshapes those tensors to `[out, in]` without transposing values.

This note still does not claim BitNet local-answer quality, full `apple-m4-metal` inference, QK256 support, Neural Engine execution, or general performance.
