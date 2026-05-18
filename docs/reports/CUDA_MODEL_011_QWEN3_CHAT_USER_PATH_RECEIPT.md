# CUDA-MODEL-011 Qwen3 Chat User-Path Receipt

Date: 2026-05-18
Campaign item: CUDA-MODEL-011
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen3-0.6b-instruct-q8_0

## Summary

CUDA-MODEL-011 records Qwen3 0.6B Q8_0 through the normal user-facing
`bitnet chat` path on the strict RTX 5070 Ti CUDA route.

The run used the pinned public Qwen3 GGUF artifact outside the repository:

```text
path = C:\bntmp\cuda-model-010\Qwen3-0.6B-Q8_0.gguf
sha256 = 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
```

The receipt proves a bounded two-turn chat/warm-session path with
`fallback_used = false`, `selected_backend = nvidia-rtx-5070-ti-cuda`, and
`selected_route = dense_regular_llm_cuda`. It does not promote Qwen3 to
`product_cli_ready`; that remains blocked on the separate CUDA-MODEL-012
promotion review.

## Evidence

```text
prompts = ci/hardware/windows-9950x3d-rtx5070ti/2026-05-18/qwen3-0_6b-chat-user-path-prompts.txt
receipt = ci/hardware/windows-9950x3d-rtx5070ti/2026-05-18/qwen3-0_6b-chat-user-path-cuda.json
source_receipt = ci/hardware/windows-9950x3d-rtx5070ti/2026-05-18/qwen3-0_6b-chat-user-path-cuda.source-warm-session.json
model_coverage_row = dense_qwen3_06b_q8_candidate
current_tier = accelerator_answer_ready
selected_backend = nvidia-rtx-5070-ti-cuda
selected_route = dense_regular_llm_cuda
runtime_api = cuda
fallback_used = false
quality_gate.passed = true
turns_count = 2
generated_tokens_total = 16
model_loaded_once = true
tokenizer_loaded_once = true
cuda_context_initialized_once = true
weights_uploaded_once = true
runtime_buffers_reused = true
per_turn_weight_upload = false
speedup_claim = false
full_cuda_residency_claimed = false
server_ready_claimed = false
bitnet_packed_i2s_qk256_proof = false
```

`bitnet receipts explain --format json` resolves the receipt to:

```text
model_coverage_row = dense_qwen3_06b_q8_candidate
current_tier = accelerator_answer_ready
selected_backend = nvidia-rtx-5070-ti-cuda
selected_route = dense_regular_llm_cuda
fallback_used = false
server_ready = false
speedup_claim = false
full_residency_claim = false
bitnet_packed_i2s_qk256_proof = false
dense_regular_llm_cuda_proof = true
```

## Claim Boundary

This receipt may claim:

- Qwen3 0.6B Q8_0 ran through the normal `bitnet chat` user path;
- the selected backend was `nvidia-rtx-5070-ti-cuda`;
- the selected route was `dense_regular_llm_cuda`;
- fallback was rejected;
- the session loaded model, tokenizer, CUDA context, and weights once;
- the bounded decoded chat answers were non-empty and passed the Qwen chat gate;
- the receipt is explainable against the Qwen3 candidate coverage row.

It must not claim:

- Qwen3 product CLI readiness;
- Qwen3 server readiness;
- Qwen3 speedup;
- Qwen3 full CUDA residency;
- Qwen3 broad dense GGUF readiness;
- Qwen3 proof inherited from Qwen2.5;
- dense regular LLM CUDA proof is BitNet packed I2_S/QK256 proof.

## Validation

```powershell
rtk cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features committed_qwen3_chat_user_path_receipts_validate
rtk cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli model_status_dashboard_lists_qwen3_as_accelerator_ready_candidate
rtk cargo run --locked -p xtask --no-default-features -- check-model-coverage
rtk C:\bntarget\bn-cuda-model-010-vs2022\release\bitnet.exe --device nvidia-rtx-5070-ti-cuda chat --model C:\bntmp\cuda-model-010\Qwen3-0.6B-Q8_0.gguf --input-file ci\hardware\windows-9950x3d-rtx5070ti\2026-05-18\qwen3-0_6b-chat-user-path-prompts.txt --max-tokens 8 --top-k 10 --receipt-path ci\hardware\windows-9950x3d-rtx5070ti\2026-05-18\qwen3-0_6b-chat-user-path-cuda.json
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-18\qwen3-0_6b-chat-user-path-cuda.json
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-18\qwen3-0_6b-chat-user-path-cuda.source-warm-session.json
rtk C:\bntarget\bn-cuda-model-010-vs2022\release\bitnet.exe receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-18\qwen3-0_6b-chat-user-path-cuda.json --format json
```
