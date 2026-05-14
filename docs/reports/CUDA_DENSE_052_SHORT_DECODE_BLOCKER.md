# CUDA-DENSE-052 Short-Decode Blocker

Date: 2026-05-14

## Summary

CUDA-DENSE-052 is blocked from product promotion.

The diagnostic run produced a real RTX 5070 Ti CUDA short-decode receipt for the
exact Qwen2.5 0.5B Q8_0 artifact. The receipt proves fallback-free execution and
CPU/CUDA generated-token equality, but the decoded output is not
user-acceptable, so it must not be used to claim short-decode product readiness.

## Diagnostic Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-short-decode-diagnostic.json
```

Recorded facts:

```text
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
route = dense_regular_llm_cuda
fallback_used = false
generated_tokens_count = 8
generated_token_ids_match = true
speedup_claim = false
bitnet_packed_i2s_qk256_proof = false
server_ready_claimed = false
```

Decoded text from the diagnostic run:

```text
opencvopencvopencv...
```

The full decoded value is preserved in the JSON receipt. It is valid decoded
text for diagnostic comparison, but it is not acceptable as a product
short-decode answer.

## Commands Run

```powershell
rtk E:\Code\Rust\BitNet-rust-195-msrv\target\release\bitnet.exe dense-gguf-qwen-short-decode-strict-cuda --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf --one-token-proof ci\hardware\windows-9950x3d-rtx5070ti\2026-05-13\dense-qwen25-q8-one-token-cuda.json --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-14\dense-qwen25-q8-short-decode-diagnostic.json
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-14\dense-qwen25-q8-short-decode-diagnostic.json
```

Additional deterministic prompt probes also preserved fallback-free CPU/CUDA
token equality, but produced similarly non-product decoded output.

## Claim Boundary

This diagnostic receipt may claim:

- exact Qwen2.5 0.5B Q8_0 artifact identity was checked;
- RTX 5070 Ti CUDA was selected;
- dense regular LLM CUDA route was used;
- CPU and CUDA selected token IDs matched across the recorded short decode;
- fallback was not used;
- kernel and transfer evidence was recorded.

It must not claim:

- Qwen2.5 short-decode product readiness;
- broad dense Qwen chat quality;
- server readiness;
- speedup;
- BitNet packed I2_S/QK256 proof;
- general dense GGUF CUDA readiness.

## Next Step

Root-cause the decoded-output quality before promoting CUDA-DENSE-052. The likely
fault domain is the dense Qwen prompt/tokenizer/runtime path, not CUDA fallback:
the diagnostic receipt already records fallback-free CPU/CUDA token equality.
