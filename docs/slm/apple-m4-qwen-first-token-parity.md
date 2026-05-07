# Apple M4 Qwen First-Token Parity

Campaign: `apple-m4-slm-answer`

Work item: `SLM-M4-003`

Status: `diverged`

## Summary

`SLM-M4-003` is not passing yet. The Qwen ChatML prompt now tokenizes to the same IDs as llama.cpp, and the Rust CLI now prefills the prompt prefix before decoding. The first generated token still diverges, so Rust-native Apple M4 SLM answers must not be claimed.

This evidence uses the Q8_0 companion artifact only as a diagnostic target because the accepted Q4_K_M artifact from `SLM-M4-002` is still unsupported by strict Rust GGUF loading. The Q8_0 companion is not accepted for operator use by this note.

## Fixed Surface

The exact ChatML prompt:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2? Answer briefly.<|im_end|>
<|im_start|>assistant
```

Both Rust and `llama-tokenize` produce:

```text
[151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 21806, 26753, 13, 151645, 198, 151644, 77091, 198]
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
  --json-out target/apple-m4-slm-answer/SLM-M4-003/qwen2_5_0_5b_q8_0_rust_math_chatml_first_token.json
```

Rust selected `apple-m4-cpu-neon` with `fallback_used=false`, but the first generated token was `24723`, decoded as `eway`.

Top Rust logits:

| Rank | Token ID | Logit |
|---:|---:|---:|
| 1 | 24723 | 15.079877853393555 |
| 2 | 105322 | 14.901577949523926 |
| 3 | 84709 | 14.533596992492676 |
| 4 | 139370 | 14.124734878540039 |
| 5 | 70378 | 13.377963066101074 |

## Next Debug Targets

The next slice should localize the first runtime divergence before adding more architecture fixes. Start by adding debug-only tensor trace points before changing model math:

- normalized tensor shapes and transpose metadata;
- configured norm type, norm epsilon, and layer-0 norm RMS samples;
- `lm_head` orientation and whether transposed output weights are used directly;
- layer-0 embedding, norm, Q/K/V projection, RoPE, attention, and MLP RMS samples;
- final norm RMS samples and top-k logits.

If the trace diverges immediately after normalization, fix Qwen RMSNorm selection first. If it diverges after Q/K/V projection, investigate tensor orientation, especially square projection tensors where shape cannot reveal a bad transpose. If hidden states remain close until logits, fix the LM-head path.

Do not claim Rust-native Apple M4 SLM answers work until this first-token divergence is resolved and coherent output passes the campaign gate.
