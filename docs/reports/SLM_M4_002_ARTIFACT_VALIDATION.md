# SLM-M4-002 Artifact Validation

Campaign: `apple-m4-slm-answer`

Work item: `SLM-M4-002`

Date: `2026-05-07`

## Result

Accepted `Qwen/Qwen2.5-0.5B-Instruct-GGUF` `qwen2.5-0.5b-instruct-q4_k_m.gguf` as the first dense SLM reference-good artifact for the Apple M4 SLM answer lane.

This only proves reference-runner answer-readiness for the dense SLM artifact. It does not prove Rust-native Apple M4 SLM answers, BitNet local-answer quality, full `apple-m4-metal` inference, Neural Engine execution, QK256 support, or performance.

## Artifact

| Field | Value |
|---|---|
| Repo | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` |
| Repo revision | `9217f5db79a29953eb74d5343926648285ec7e67` |
| File | `qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| SHA256 | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` |
| Bytes | `491400032` |
| Local path | `target/apple-m4-slm-answer/SLM-M4-002/candidates/qwen2_5_0_5b/qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| Storage policy | Under preferred `<= 500 MiB` artifact budget; binary is ignored and not committed |

## Metadata

Reference runner metadata reports:

- `general.architecture = qwen2`
- `general.name = qwen2.5-0.5b-instruct`
- `general.size_label = 630M`
- `general.file_type = 15`
- `tokenizer.ggml.model = gpt2`
- `tokenizer.ggml.pre = qwen2`
- `tokenizer.chat_template` is present
- `tokenizer.ggml.eos_token_id = 151645`
- `tokenizer.ggml.bos_token_id = 151643`
- `model ftype = Q4_K - Medium`
- `model size = 462.96 MiB`
- `offloaded 0/25 layers to GPU`

## Reference Command

```bash
/Users/steven/.cache/bitnet_cpp/build/bin/llama-cli \
  -m target/apple-m4-slm-answer/SLM-M4-002/candidates/qwen2_5_0_5b/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -p "<prompt>" \
  -n 16 \
  --no-display-prompt \
  --temp 0 \
  --top-k 1 \
  -ngl 0
```

The prompt suite is recorded in `ci/quality/apple-m4-slm-answer-corpus.yaml`. The reference command uses a raw deterministic prompt shape. The artifact has `tokenizer.chat_template` metadata, but Rust CLI chat-template integration remains `SLM-M4-003+` work.

## Prompt Results

| Case | Gate | Continuation | Pass |
|---|---|---|---|
| `math_2_plus_2` | contains `4` or `four` | `4. Answer: 4. To explain it simply: 1.` | yes |
| `capital_france` | contains `Paris` | `The capital of France is Paris. It is located in the south of the country` | yes |
| `rust_sentence` | contains Rust/programming/language/safety/efficiency | `Rust is a programming language that is known for its safety, efficiency, and ability` | yes |

## Follow-Up

`SLM-M4-003` may now attempt Rust-native Apple M4 CPU/NEON SLM answer receipts against this artifact. It must still record tokenizer authority, backend routing, fallback status, generated text, token IDs, and timing before claiming Rust-native Apple M4 SLM answers work.
