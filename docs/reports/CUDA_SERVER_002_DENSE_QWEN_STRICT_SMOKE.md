# CUDA-SERVER-002 Dense Qwen Strict Server Smoke

CUDA-SERVER-002 records the first bounded strict RTX 5070 Ti server-smoke receipt
for the dense Qwen2.5 0.5B Q8_0 profile. It does not promote production server
readiness, global dense GGUF server readiness, speedup, full CUDA residency, or
BitNet packed I2_S/QK256 proof.

## Receipt

Committed receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/server-strict-dense-qwen25-q8-smoke.json
```

The receipt records:

```text
receipt_kind = server_shared_engine_chat_completion
requested_backend = nvidia-rtx-5070-ti-cuda
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
selected_route = dense_regular_llm_cuda
model_coverage_row = dense_qwen25_05b_q8_cuda
model_coverage_tier = product_cli_ready
fallback_used = false
generated_text_non_empty = true
quality_gate.passed = true
server_smoke_response_claimed = true
server_ready_claimed = false
speedup_claim = false
full_cuda_residency_claimed = false
dense_regular_llm_cuda_inference_claimed = true
bitnet_packed_i2s_qk256_proof = false
```

## Command

The smoke used the built server binary from an isolated target directory:

```powershell
server.exe `
  --host 127.0.0.1 `
  --port 18080 `
  --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf `
  --device nvidia-rtx-5070-ti-cuda `
  --log-level info `
  --log-format compact
```

The OpenAI-compatible request was:

```json
{
  "model": "qwen2.5-0.5b-instruct-q8_0",
  "messages": [
    {
      "role": "user",
      "content": "Say exactly: 4"
    }
  ],
  "max_tokens": 4,
  "temperature": 0.0,
  "top_p": 1.0,
  "stream": false
}
```

## Implementation Note

The server CUDA feature already enabled CUDA inference and kernels, but the
server's GGUF model loading path also needs `bitnet-models/cuda`. Without that
feature propagation, the server detects the RTX 5070 Ti and starts, but default
model loading rejects `Device::Cuda(0)` with:

```text
CUDA support not enabled; rebuild with --features gpu
```

CUDA-SERVER-002 therefore includes the narrow server feature propagation needed
for the existing shared-engine path to load the verified dense Qwen GGUF on the
strict configured CUDA device.

## Claim Boundary

This evidence proves only a bounded dense Qwen strict server-smoke response for
the exact Qwen2.5 0.5B Q8_0 RTX 5070 Ti profile. It does not prove production
server readiness, broad chat quality, global dense GGUF serving, BitNet QK256
serving, speedup, or full CUDA residency.
