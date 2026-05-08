# CUDA-ANSWER-010 Strict RTX 5070 Ti Corpus Proof

## Summary

`CUDA-ANSWER-010` records the first strict RTX 5070 Ti CUDA answer-corpus proof
for the official Microsoft BitNet I2_S artifact after the QK256 CUDA GEMV layout
alignment in PR #4024.

The proof uses the answer-ready artifact authority from `MODEL-ARTIFACT-007`
and the strict Rust CPU correctness from `CPU-ANSWER-007`. It does not make a
speedup claim or a broad chat-quality claim.

## Source PR

| Field | Value |
|---|---|
| PR | `#4024` |
| Title | `fix(cuda): align QK256 I2_S GEMV layout` |
| Merge SHA | `1d0efa60e1d501fe03b5289de756e71b976fb8de` |
| Merged at | `2026-05-08T10:20:26Z` |

## Artifact and Authority

| Field | Value |
|---|---|
| Model artifact | `microsoft_bitnet_b158_2b_4t_gguf_i2s_current` |
| Repo | `microsoft/bitnet-b1.58-2B-4T-gguf` |
| File | `ggml-model-i2_s.gguf` |
| SHA256 | `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162` |
| Tokenizer authority | external Microsoft tokenizer, `tokenizer.ggml.pre=llama-bpe` |
| Prompt template | `bitnetcpp-answer` |
| Prompt envelope | `User: <question><\|eot_id\|>Assistant:` |

## Committed Evidence

| Receipt | Purpose |
|---|---|
| `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/strict-cuda-ask-math.json` | One strict CUDA `ask` run for `math_2_plus_2`. |
| `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-answer-corpus.json` | Five-case deterministic CUDA answer corpus with `--fail-on-quality`. |

Receipt SHA256 values from the local closeout run:

| Receipt | SHA256 |
|---|---|
| `strict-cuda-ask-math.json` | `b9aabc28b865e067b9c33705280cf5213fdd007b04298e66ef7cbcaf1748dd0b` |
| `cuda-answer-corpus.json` | `13a7e4a6a8ebde4bd0342949485935cf56168397a39623157d11683a8ab29b3c` |

## Strict Ask Result

The strict CUDA ask command generated the visible answer:

```text
4
```

The receipt records:

| Field | Value |
|---|---|
| `artifact_kind` | `bitnet_cuda_answer` |
| `backend.requested_backend` | `nvidia-rtx-5070-ti-cuda` |
| `backend.selected_backend` | `nvidia-rtx-5070-ti-cuda` |
| `backend.runtime_api` | `cuda` |
| `backend.fallback_used` | `false` |
| `bitnet.kernel_id` | `qk256_gemv_cuda` |
| `bitnet.weights_uploaded_once` | `true` |
| `bitnet.per_token_weight_upload` | `false` |
| `execution_coverage.bitnet_linear_layers_cpu_fallback` | `0` |
| `execution_coverage.bitnet_linear_layers_on_cuda` | `4410` |
| `prompt_prefill.exercised` | `true` |
| `prompt_prefill.tokens` | `18` |
| `quality.garbage_filter_passed` | `true` |
| `speedup_claim` | `false` |

## Corpus Result

The CUDA answer corpus passed all committed deterministic cases.

| Case | Answer | Gate result |
|---|---|---|
| `math_2_plus_2` | `4` | pass |
| `capital_france` | `Paris` | pass |
| `repeat_colors` | `red blue green` | pass |
| `say_ok` | `OK` | pass |
| `yes_no_water` | `Yes.` | pass |

The corpus receipt records `quality_summary.passed = 5`,
`quality_summary.failed = 0`, `quality_summary.timeout = 0`, and
`quality_summary.not_run = 0`.

Every case records:

- `selected_backend = nvidia-rtx-5070-ti-cuda`
- `runtime_api = cuda`
- `fallback_used = false`
- `selected_kernel = qk256_gemv_cuda`
- `quality.passed = true`

The aggregate answer-corpus artifact still uses the diagnostic corpus artifact
kind emitted by the harness. This report is the governed CUDA-ANSWER-010 proof
that the diagnostic corpus passed under an answer-ready artifact and strict CUDA
backend conditions. It does not broaden the claim beyond the committed corpus.

## Validation Commands

The local closeout branch reran:

```powershell
cargo run --locked --release -p bitnet-cli --no-default-features `
  --features cpu,cuda,full-cli -- ask `
  --device nvidia-rtx-5070-ti-cuda `
  --model D:\Code\Rust\BitNet\models\microsoft-bitnet-b1.58-2B-4T-gguf\ggml-model-i2_s.gguf `
  --tokenizer D:\Code\Rust\BitNet\models\microsoft-bitnet-b1.58-2B-4T\tokenizer.json `
  --question "What is 2+2? Answer with only the number." `
  --max-new-tokens 8 `
  --temperature 0 `
  --strict-cuda `
  --receipt-out target\bitnet\receipts\cuda-answer-readiness\strict-cuda-ask-math-closeout.json
```

```powershell
cargo run --locked --release -p bitnet-cli --no-default-features `
  --features cpu,cuda,full-cli -- answer-corpus `
  --device nvidia-rtx-5070-ti-cuda `
  --model D:\Code\Rust\BitNet\models\microsoft-bitnet-b1.58-2B-4T-gguf\ggml-model-i2_s.gguf `
  --tokenizer D:\Code\Rust\BitNet\models\microsoft-bitnet-b1.58-2B-4T\tokenizer.json `
  --corpus ci\quality\bitnet-answer-corpus.yaml `
  --per-prompt-timeout-seconds 300 `
  --dump-logit-steps 1 `
  --logits-topk 20 `
  --fail-on-quality `
  --json-out target\bitnet\receipts\cuda-answer-readiness\cuda-answer-corpus-closeout.json
```

## Claim Boundary

Allowed after this proof:

- The official Microsoft I2_S artifact passes the committed deterministic answer
  corpus through the strict RTX 5070 Ti CUDA answer path.
- The strict answer path selected `nvidia-rtx-5070-ti-cuda`, used CUDA runtime
  execution, routed BitNet linear work through `qk256_gemv_cuda`, rejected CPU
  fallback, exercised prompt prefill, and passed answer quality gates for the
  committed corpus.

Not claimed:

- Broad chat quality.
- Production server readiness.
- CUDA speedup.
- Full CUDA residency for every non-linear transformer phase.
- Dense CUDA, generic GPU, WGPU, OpenCL, Metal, NPU, or SLM equivalence to this
  BitNet packed I2_S CUDA proof.

## Next Step

The next CUDA answer-readiness PR should record CPU/CUDA answer parity for the
same artifact, tokenizer, prompt template, and deterministic corpus before warm
session or throughput work.
