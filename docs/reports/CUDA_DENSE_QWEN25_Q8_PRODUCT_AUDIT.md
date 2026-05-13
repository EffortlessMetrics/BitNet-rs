# CUDA Dense Qwen2.5 Q8 Product Audit

Date: 2026-05-13
Campaign item: CUDA-DENSE-050
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen2.5-0.5b-instruct-q8_0

## Summary

Dense Qwen2.5 0.5B Q8_0 has committed real RTX 5070 Ti hardware receipts for
one-token, short-decode, warm-session, benchmark-baseline, repeated-comparator,
and benchmark-qualification proof. These are not validator-only artifacts. They
record `selected_backend =
nvidia-rtx-5070-ti-cuda`, `runtime_api = cuda`, `fallback_used = false`, the
`dense_regular_llm_cuda` route, CPU/CUDA token or top-k parity evidence, CUDA
kernel stats, timing, and claim-boundary fields.

The benchmark qualification receipts explicitly keep `speedup_claim = false`
and `benchmark_qualified_speedup = false`: the reviewed CUDA profiles are
slower than same-artifact CPU means, and pure host-to-device copy timing remains
blocked even after the model-load envelope was recorded.

The user-facing dense Qwen ask/chat paths are implemented and campaign-merged,
but this audit did not find a committed hardware receipt named for a direct
`bitnet ask` or `bitnet chat` user invocation under
`ci/hardware/windows-9950x3d-rtx5070ti`. The product claim should therefore say
that the ask/chat UX is backed by the strict runtime receipts and receipt
validators, while the next proof should commit direct user-path ask/chat
receipts before promoting any broader product statement.

## Receipt Classification

| Surface | Current state | Evidence | Classification | Claim allowed | Must not claim |
| --- | --- | --- | --- | --- | --- |
| One-token strict CUDA | Real hardware receipt exists | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json` | hardware/user-path runtime proof command | one deterministic greedy token through `dense_regular_llm_cuda` on RTX 5070 Ti, CPU/CUDA selected-token and top-k rank match | BitNet QK256 proof, chat, speedup, server readiness, full residency |
| Short decode strict CUDA | Real hardware receipt exists | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json` | hardware/user-path runtime proof command | bounded 8-token deterministic decode through `dense_regular_llm_cuda`, CPU/CUDA generated-token match | broad chat quality, server readiness, speedup, full residency |
| Warm-session strict CUDA | Real hardware receipt exists | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json` | hardware/user-path runtime proof command | three-turn bounded warm session, model/tokenizer/context loaded once, weights uploaded once, fallback false | broad chat quality, server readiness, global persistence/full residency, speedup |
| Benchmark baseline | Real baseline receipt exists | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-cuda-benchmark-baseline.json` | measured existing-receipt baseline | one-token, short-decode, and warm-session profile measurements with speedup false | benchmark-qualified speedup |
| Repeated comparator | Real repeated comparator exists | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-gguf-qwen-repeated-comparator.json` | repeated CPU/CUDA comparator | three runs per backend for one-token, short-decode, and warm-session profiles | benchmark-qualified speedup |
| Benchmark qualification | Real qualification reviews exist; speedup rejected | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-gguf-qwen-benchmark-qualification.json`; `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-006-h2d-envelope-qualification/dense-gguf-qwen-benchmark-qualification-h2d-envelope.json` | governed profile review | profile-specific review exists and keeps `benchmark_qualified_speedup = false` | accepted speedup, global CUDA speedup |
| `bitnet ask --device cuda --model qwen2.5-0.5b-instruct-q8_0` | UX path merged | `CUDA-UX-003` event and code path | implemented UX backed by short-decode source receipt/validator | bounded ask UX may emit validated `dense_gguf_qwen_ask_strict_cuda_proof` receipts | committed hardware ask receipt exists under `ci/hardware` |
| `bitnet chat --device cuda --model qwen2.5-0.5b-instruct-q8_0` | UX path merged | `CUDA-UX-004` event and code path | implemented UX backed by warm-session source receipt/validator | bounded chat UX may emit validated `dense_gguf_qwen_chat_strict_cuda_proof` receipts | committed hardware chat receipt exists under `ci/hardware` |

## Answers To Audit Questions

Is one-token strict CUDA real hardware execution or validator-only?

Real hardware execution. The one-token receipt is committed at
`ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json`.
It records `artifact_kind = dense_gguf_qwen_one_token_strict_cuda_proof`,
`selected_backend = nvidia-rtx-5070-ti-cuda`, `runtime_api = cuda`,
`fallback_used = false`, `execution_plan.selected_route =
dense_regular_llm_cuda`, three kernel-stat entries, CPU selected token `576`,
CUDA selected token `576`, and matching top-k rank evidence.

Is short decode real?

Real hardware execution. The short-decode receipt is committed at
`ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json`.
It records `artifact_kind = dense_gguf_qwen_short_decode_strict_cuda_proof`,
`fallback_used = false`, eight generated tokens, matching CPU/CUDA generated
token IDs, and decoded text `The answer is 4. What is`.

Is warm session real?

Real hardware execution. The warm-session receipt is committed at
`ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json`.
It records three turns, `model_loaded_once = true`, `tokenizer_loaded_once =
true`, `cuda_context_initialized_once = true`, `weights_uploaded_once = true`,
`runtime_buffers_reused = true`, and `fallback_used = false`.

Is chat path real?

The chat UX path is implemented and merged in `CUDA-UX-004`. It emits and
validates `dense_gguf_qwen_chat_strict_cuda_proof` receipts backed by the
warm-session proof source receipt. This audit did not find a committed direct
`bitnet chat` hardware receipt under `ci/hardware/windows-9950x3d-rtx5070ti`,
so the durable proof should remain "chat UX exists and is validator-backed by
warm-session proof" until a direct user-path chat receipt is committed.

Are CPU and CUDA generated token IDs compared?

Yes. The one-token receipt compares CPU and CUDA selected token IDs and top-k
rank evidence. The short-decode receipt compares CPU and CUDA generated token
IDs across eight tokens. The warm-session receipt compares generated token IDs
and top-k evidence across three turns.

Which receipts are committed?

- `dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json`
- `dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json`
- `dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json`
- `dense-gguf-qwen-cuda-benchmark-baseline.json`
- `dense-gguf-qwen-repeated-comparator.json`
- `dense-gguf-qwen-benchmark-qualification.json`
- `dense-gguf-qwen-benchmark-qualification-h2d-envelope.json`
- Prerequisite dense Qwen route, boundary, KV, sampling, and op-parity receipts
  under `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/`

Which receipts are only synthetic validator tests?

The receipt validators and synthetic rejection tests live in Rust test/code
surfaces, not as committed hardware receipts in this audit set. The committed
one-token, short-decode, warm-session, and benchmark-baseline JSON files listed
above are hardware/user-path runtime proof receipts or measured baseline
receipts, not synthetic-only validator fixtures.

Which benchmark profiles exist?

The benchmark baseline records these profiles, all with `speedup_claim = false`
and `benchmark_qualified_speedup = false`. The repeated comparator and
qualification reviews consume the same profile set:

- `one_token`
- `short_decode_8`
- `warm_session_3_turns`

The committed qualification reviews reject all reviewed profiles for speedup.
The latest review consumes the H2D model-load envelope and still records
`benchmark_qualified_speedup = false` because CUDA means are slower than CPU
means and pure host-to-device copy timing remains unmeasured.

Which gaps block product CLI readiness?

The model coverage matrix already marks Qwen2.5 0.5B Q8_0 as
`product_cli_ready`, `accelerator_answer_ready = true`, and
`dense_regular_llm_cuda_proof = true`. The remaining gaps are narrower:

- Commit direct `bitnet ask` and `bitnet chat` user-path hardware receipts if
  the product claim needs direct command evidence rather than validator-backed
  runtime proof.
- Keep speedup false unless a later governed benchmark review accepts a
  profile-specific speedup claim from newer evidence.
- Keep `server_ready = false` until a strict server smoke receipt lands.
- Keep `full_residency_claim = false` until every relevant phase is proven
  resident.

## Claim Boundary

May claim:

- Qwen2.5 0.5B Q8_0 has real RTX 5070 Ti strict CUDA dense runtime receipts for
  one-token, short-decode, and warm-session proof.
- The route is `dense_regular_llm_cuda`, not BitNet QK256.
- The receipts are fallback-free and selected-backend specific.
- CPU/CUDA generated-token or selected-token evidence is recorded and passing
  for the committed proof scopes.

Must not claim:

- Dense Qwen proof is BitNet packed I2_S/QK256 proof.
- Dense Qwen has accepted benchmark-qualified speedup.
- Dense Qwen has server readiness.
- Dense Qwen has global full-residency proof.
- A direct committed `bitnet ask` or `bitnet chat` hardware receipt exists under
  `ci/hardware/windows-9950x3d-rtx5070ti`.

## Validation

```powershell
rtk python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json
rtk python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json
rtk python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json
rtk python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-cuda-benchmark-baseline.json
rtk python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-gguf-qwen-repeated-comparator.json
rtk python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-gguf-qwen-benchmark-qualification.json
rtk python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-006-h2d-envelope-qualification/dense-gguf-qwen-benchmark-qualification-h2d-envelope.json
```
