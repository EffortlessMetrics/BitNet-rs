# M4-SLM-EX-006 Q4_K_M Second Dense Model

Campaign: `apple-m4-slm-excellence`

Work item: `M4-SLM-EX-006`

Date: `2026-05-09`

## Result

Accepted `qwen2.5-0.5b-instruct-q4_k_m` as the second supported dense SLM model
for the Apple M4 Mac mini lane. The default remains
`qwen2.5-0.5b-instruct-q8_0`.

This proves only the Rust-native Apple M4 CPU/NEON dense Qwen answer path for
this artifact. It does not prove BitNet local-answer quality, QK256, Neural
Engine execution, MPSGraph model inference, full Apple Metal inference, CUDA,
x86, or broad Apple Silicon performance.

## Artifact

| Field | Value |
|---|---|
| Model ID | `qwen2.5-0.5b-instruct-q4_k_m` |
| Repo | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` |
| Repo revision | `9217f5db79a29953eb74d5343926648285ec7e67` |
| File | `qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| SHA256 | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` |
| Bytes | `491400032` |
| GGUF architecture | `qwen2` |
| Quantization | `Q4_K_M` |
| Tokenizer | `tokenizer.ggml.model = gpt2`, `tokenizer.ggml.pre = qwen2` |
| Prompt template | `qwen2.5` |
| Cache path | user model cache, not committed |

## Reference Sanity

The existing `SLM-M4-002` reference-runner report accepted this exact artifact.
For this promotion slice, the same cached artifact was also checked with the
local reference runner:

```bash
/Users/steven/.cache/bitnet_cpp/build/bin/llama-cli \
  -m ~/Library/Caches/bitnet-rs/models/qwen2.5-0.5b-instruct-q4_k_m/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -p "<prompt>" \
  -n 16 \
  --no-display-prompt \
  --temp 0 \
  --top-k 1 \
  -ngl 0
```

Observed outputs:

| Prompt | Output sanity |
|---|---|
| `What is 2+2? Answer briefly.` | Contains `4` |
| `Name the capital of France.` | Contains `Paris` |
| `Write one short sentence about Rust.` | Mentions Rust as a programming language |

## Rust M4 Evidence

The candidate first failed the strict Rust path because the artifact uses
standard GGUF quantized tensor types `Q5_0`, `Q4_K`, and `Q6_K`. The support
slice adds eager F32 dequantization for those types and leaves unrelated
standard GGUF quantizations fail-closed.

Strict one-shot probe:

```bash
bitnet mac ask "Answer with a single digit: 2+2=" \
  --model-id qwen2.5-0.5b-instruct-q4_k_m \
  --max-new-tokens 8 \
  --json-out target/apple-m4-slm-excellence/M4-SLM-EX-006/q4-ask-supported.json
```

Result: `: 4<|im_end|>` with `fallback_used = false`.

Quality corpus:

```bash
bitnet mac validate \
  --model-id qwen2.5-0.5b-instruct-q4_k_m \
  --json-out target/apple-m4-slm-excellence/M4-SLM-EX-006/q4-mac-validate.json \
  --quiet
```

Receipt check:

```bash
bitnet mac receipts-check \
  target/apple-m4-slm-excellence/M4-SLM-EX-006/q4-mac-validate.json \
  --json
```

Result:

- `artifact_kind = slm_apple_m4_warm_session`
- `requested_backend = apple-m4-cpu-neon`
- `selected_backend = apple-m4-cpu-neon`
- `runtime_api = cpu`
- `fallback_used = false`
- `prompt_count = 10`
- `generated_tokens = 108`
- `quality_summary.passed = true`
- `determinism.passed = true`
- `repeated_prompt_groups = 5`
