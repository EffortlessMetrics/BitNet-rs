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
