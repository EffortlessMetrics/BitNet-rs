# Intel 258V Platform Probe Notes - 2026-05-06

This bundle records OS-visible hardware identity on the local Lunar Lake laptop.
It does not claim OpenCL, Level Zero, OpenVINO, Arc 140V execution, NPU
execution, BitNet inference, parity, or benchmark performance.

`platform-probe-cli.json` was emitted by the `bitnet lunar-lake-probe` command
after the CLI receipt path landed. It is a runtime-visibility receipt only and
keeps `kernel_execution=false`, `graph_execution=false`, and
`bitnet_inference=false`.

`cpu-bitnet-validation.json` was refreshed after the canonical BitNet GGUF and
explicit tokenizer were made available locally under the ignored `models/`
tree. It records `status=preflight_ready`, the GGUF SHA-256, explicit tokenizer
source, CPU AVX2 visibility, and `fallback_used=false` without running BitNet
inference or CPU kernels.

`strict-bitnet-cpu-proof.json` was emitted by a strict one-token CPU run using
the real `ggml-model-i2_s.gguf` artifact and explicit tokenizer. It records
`loader.mode=real_gguf`, `minimal_loader_fallback_used=false`,
`mock_tensors_used=false`, `tokenizer_source=explicit`,
`requested_backend=cpu`, `selected_backend=cpu-rust`, `runtime_api=cpu`,
`kernel_id=i2_s-avx2-reference`, and `fallback_used=false`.
The top-level platform bundle remains `proof_stage=detected`; only the CPU lane
has a receipt-backed one-token decode smoke artifact.

## Captured Facts

- Machine: Lenovo `83MC`
- OS: Microsoft Windows 11 Home `10.0.26200`
- CPU: Intel Core Ultra 7 258V, 8 cores / 8 logical processors
- CPU runtime features: AVX2, FMA, SSE4.2 visible; AVX-512F not visible
- Memory: 32 GiB class system memory, reported as 8 x 4 GiB at 8533 MT/s
- Power scheme: Balanced
- Arc 140V: OS-visible as `Intel(R) Arc(TM) 140V GPU (16GB)`, PCI `8086:64A0`
- NPU: OS-visible as `Intel(R) AI Boost`, PCI `8086:643E`

## Runtime Gaps

- `clinfo` is not installed in PATH, so this bundle does not prove OpenCL runtime visibility.
- `sycl-ls` and `ze_info` are not installed in PATH, so this bundle does not prove Level Zero runtime visibility.
- Python cannot import `openvino`, so this bundle does not prove OpenVINO GPU or NPU visibility.
- The canonical BitNet GGUF and explicit tokenizer are present locally, so CPU
  BitNet validation is recorded as `preflight_ready`.
- A strict CPU proof generated one token from the real BitNet GGUF through the
  CPU path and wrote `strict-bitnet-cpu-proof.json`.
- The strict CPU proof generated text `'E` for the arithmetic smoke prompt, so
  it is not a correctness or parity receipt.
- The strict CPU proof is a one-token smoke profile: `first_token_ms=190337`,
  `prefill_ms=173199.784`, and `decode_steady_state_tok_s=null`. It is not a
  steady-state throughput or benchmark-performance receipt.
- The strict CPU proof receipt has known metadata limitations: `counts.n_tensors`
  and `counts.n_kv` are zero in this profile receipt shape despite real GGUF
  loading evidence elsewhere in the receipt, and the tokenizer block reports
  `type=sentencepiece` while the model contract identifies the tokenizer as
  LLaMA 3. Treat this artifact as strict selected-file execution evidence, not
  tokenizer semantic parity or receipt-schema completeness.
- CPU BitNet preflight readiness is not strict GGUF loading through the
  inference path, tokenizer resolution through the inference path, QK256/TL2
  kernel execution, or BitNet generation.
- The CLI probe reports Arc 140V runtime identity as unavailable because the
  OpenCL, Level Zero, and OpenVINO runtime tools were not available to the
  command, even though the OS/PnP artifact records the Arc 140V device identity.

## Claim Boundary

Detection is not execution. These artifacts only establish same-machine hardware
visibility and the absence of the runtime tooling needed for OpenCL, Level Zero,
and OpenVINO claims on this Windows host at capture time.

The CPU lane now also has a receipt-backed strict one-token BitNet CPU decode
smoke. That receipt shows selected CPU backend execution with strict real-GGUF
loading, explicit tokenizer file selection, and fallback disabled; it does not
prove tokenizer semantic parity, output correctness, scalar/AVX2 parity,
steady-state decode throughput, benchmark performance, Arc 140V execution, or
NPU execution.
