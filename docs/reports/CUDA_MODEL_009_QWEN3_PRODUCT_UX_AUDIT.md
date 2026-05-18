# CUDA-MODEL-009 Qwen3 0.6B Product UX Audit

Date: 2026-05-18
Campaign item: CUDA-MODEL-009
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen3-0.6b-instruct-q8_0

## Summary

CUDA-MODEL-009 audits whether the Qwen3 0.6B Q8_0 candidate is ready to move
from `accelerator_answer_ready` to `product_cli_ready`.

It is not ready for promotion yet. The existing receipts prove the pinned
artifact, CPU answer sanity, all-layer planning, one-token CUDA, short-decode
CUDA, warm-session CUDA, fallback rejection, and governed benchmark review for
the Qwen3-specific `dense_regular_llm_cuda` route. They do not prove the normal
user-facing `bitnet ask` or `bitnet chat` product paths for Qwen3.

The model coverage row stays at:

```text
model_coverage_row = dense_qwen3_06b_q8_candidate
current_tier = accelerator_answer_ready
selected_route = dense_regular_llm_cuda
product_cli_ready = false
server_ready = false
speedup_claim = false
full_residency_claim = false
bitnet_packed_i2s_qk256_proof = false
dense_regular_llm_cuda_proof = true
```

## Evidence Inspected

| Surface | Evidence | Audit result | Promotion effect |
| --- | --- | --- | --- |
| Model verify | `verifier_surface = "bitnet model verify qwen3-0.6b-instruct-q8_0"` in `ci/model-artifacts/model-coverage-matrix.toml` | The model has a named verifier surface and exact artifact contract. | Prerequisite only. |
| Model status | `dense_qwen3_06b_q8_candidate` is listed as `accelerator_answer_ready` for `nvidia-rtx-5070-ti-cuda`. | The cockpit shows Qwen3 as a candidate with one-token, short-decode, warm-session, and benchmark review ready. | Keep current tier. |
| Normal ask path | No committed Qwen3 receipt from normal `bitnet ask --device cuda --model qwen3-0.6b-instruct-q8_0 ...` was found. | Product ask UX is still unproven. | Blocks `product_cli_ready`. |
| Chat or warm user path | `qwen3-0_6b-warm-session-cuda.json` proves a bounded strict CUDA warm-session harness with load-once fields. | The receipt is useful accelerator evidence, but it is not a normal `bitnet chat` or user ask-session product receipt. | Blocks `product_cli_ready`. |
| Receipt explain | Existing Qwen3 receipts are JSON receipts under `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/`. | The support surface can explain the proof family and rejected claims. | No promotion by itself. |
| Fallback rejection | One-token, short-decode, warm-session, and benchmark review receipts select `nvidia-rtx-5070-ti-cuda`, route `dense_regular_llm_cuda`, and record `fallback_used = false`. | Strict fallback rejection is proven for the committed proof receipts. | Supports candidate state only. |
| Quality gate | CPU answer sanity, generated-token equality, top-k equality, and valid decoded text are recorded across the Qwen3 proof ladder. | Quality evidence is sufficient for `accelerator_answer_ready`. | Still needs user-path receipts. |
| Benchmark review | `qwen3-0_6b-benchmark-qualification.json` records `qualification_decision = not_accepted`. | Speedup remains rejected for all reviewed profiles. | Keep `speedup_claim = false`. |
| Claim booleans | Coverage row keeps product, server, speedup, full-residency, and BitNet QK256 claims false. | Proof-family separation is intact. | No claim changes. |

## Existing Receipt Ladder

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-cpu-answer-corpus.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-cuda-all-layer-plan.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-model-boundary-fixtures.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-kv-cache-policy.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-sampling-policy.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-one-token-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-short-decode-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-warm-session-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-benchmark-qualification.json
```

Recorded Qwen3 CUDA facts:

```text
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
route = dense_regular_llm_cuda
fallback_used = false
speedup_claim = false
server_ready_claimed = false
full_cuda_residency_claimed = false
bitnet_packed_i2s_qk256_proof = false
```

## Blockers

1. `CUDA-MODEL-010` must record a normal user-path Qwen3 ask receipt.
   The receipt must come from `bitnet ask --device cuda --model
   qwen3-0.6b-instruct-q8_0 ...`, select `nvidia-rtx-5070-ti-cuda`, select
   `dense_regular_llm_cuda`, reject fallback, produce valid decoded text, and
   be explainable by `bitnet receipts explain`.
2. `CUDA-MODEL-011` must record a normal Qwen3 chat or ask-session receipt.
   The receipt must show one model load, one tokenizer load, one CUDA context
   initialization, upload-once weights, multiple prompts, fallback rejection,
   and a session summary receipt.
3. `CUDA-MODEL-012` must review those user-path receipts before any
   `product_cli_ready` promotion.
4. Any future speed claim still needs repeated same-artifact CPU and CUDA
   comparator evidence by exact profile. Product UX receipts do not satisfy the
   performance contract by themselves.

## Claim Boundary

This audit may claim:

- Qwen3 0.6B Q8_0 is an accelerator-answer-ready candidate;
- Qwen3 has exact artifact, CPU answer, all-layer plan, one-token,
  short-decode, warm-session, and benchmark review receipts;
- the existing Qwen3 CUDA proof receipts use `dense_regular_llm_cuda` on
  `nvidia-rtx-5070-ti-cuda`;
- fallback was rejected in the committed Qwen3 CUDA proof receipts;
- benchmark qualification was reviewed and speedup was not accepted;
- Qwen3 product CLI promotion is blocked on normal ask and chat user-path
  receipts.

It must not claim:

- Qwen3 product CLI readiness;
- Qwen3 server readiness;
- Qwen3 speedup;
- Qwen3 benchmark-qualified speed;
- Qwen3 full CUDA residency;
- Qwen3 broad dense GGUF readiness;
- Qwen3 proof inherited from Qwen2.5;
- dense regular LLM CUDA proof is BitNet packed I2_S/QK256 proof.

## Validation

This report is generated from committed docs, model coverage, and receipt
artifacts. No runtime, kernel, tokenizer, loader, server, or model coverage
claim was changed.

```powershell
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-cpu-answer-corpus.json
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-one-token-cuda.json
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-short-decode-cuda.json
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-warm-session-cuda.json
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-benchmark-qualification.json
rtk cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli model_status_dashboard_lists_qwen3_as_accelerator_ready_candidate
rtk cargo run --locked -p xtask --no-default-features -- check-model-coverage
rtk npm exec --yes --package markdownlint-cli2@0.18.1 -- markdownlint-cli2 --config .markdownlint.jsonc docs/reports/CUDA_MODEL_009_QWEN3_PRODUCT_UX_AUDIT.md
rtk git diff --check
```
