# CUDA-MODEL-007 Qwen3 0.6B Benchmark Qualification Review

Date: 2026-05-15
Campaign item: CUDA-MODEL-007
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen3-0.6b-instruct-q8_0

## Summary

CUDA-MODEL-007 records a governed benchmark qualification review for the exact
Qwen3 0.6B Q8_0 artifact covered by CUDA-MODEL-001 through CUDA-MODEL-006.

This review consumes the committed one-token, short-decode, and warm-session
strict CUDA proof receipts. It is not a fresh benchmark run and does not
promote speed. Each reviewed profile remains `not_accepted` because the current
evidence is one committed proof receipt per profile, CUDA total time is slower
than the same-artifact CPU reference total in those receipts, pure H2D timing is
not separated from model-load overhead, and no Qwen3 profile-specific speedup
threshold has been accepted.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-benchmark-qualification.json
```

Recorded facts:

```text
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
route = dense_regular_llm_cuda
model_id = qwen3-0.6b-instruct-q8_0
model_sha256 = 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
fallback_used = false
qualification_decision = not_accepted
accepted_profiles = []
blocked_profiles = one_token, short_decode_8, warm_session_3_turns
speedup_claim = false
benchmark_qualified_speedup = false
server_ready_claimed = false
full_cuda_residency_claimed = false
bitnet_packed_i2s_qk256_proof = false
```

## Profile Review

| Profile | CPU total ms | CUDA total ms | Runs per backend | Decision |
| --- | ---: | ---: | ---: | --- |
| one_token | 3865.0375 | 4752.5861 | 1 | not accepted |
| short_decode_8 | 5638.2572 | 6222.1281 | 1 | not accepted |
| warm_session_3_turns | 6697.5988 | 6832.0021 | 1 | not accepted |

All profiles are fallback-free, quality-gated, and use
`dense_regular_llm_cuda`. They are proof evidence, not speed evidence.

## Command

The review is generated from committed receipts and does not require CUDA
hardware:

```powershell
rtk cargo run --locked -p bitnet-bench-receipts --bin qwen3_cuda_benchmark_qualification_receipt --no-default-features -- --receipt-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-benchmark-qualification.json
```

## Claim Boundary

This receipt may claim:

- exact Qwen3 0.6B Q8_0 artifact identity was checked;
- RTX 5070 Ti CUDA was selected in the reviewed proof receipts;
- dense regular LLM CUDA route was used;
- one-token, short-decode, and warm-session Qwen3 proof receipts were reviewed;
- fallback was not used in the reviewed proof receipts;
- benchmark qualification was reviewed and speedup was rejected.

It must not claim:

- Qwen3 speedup;
- Qwen3 benchmark-qualified speed;
- Qwen3 server readiness;
- Qwen3 full CUDA residency;
- Qwen3 broad product readiness;
- dense Qwen proof is BitNet packed I2_S/QK256 proof;
- general dense GGUF CUDA readiness.

## Validation

```powershell
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-benchmark-qualification.json
rtk cargo test --locked -p bitnet-bench-receipts --no-default-features dense_gguf_qwen_benchmark_qualification
rtk cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
rtk cargo run --locked -p xtask --no-default-features -- campaign generate --check
rtk cargo run --locked -p xtask --no-default-features -- campaign doctor
rtk git diff --check
```
