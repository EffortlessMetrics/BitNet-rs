# Intel 258V Platform Probe Notes - 2026-05-06

This bundle records OS-visible hardware identity on the local Lunar Lake laptop.
It does not claim OpenCL, Level Zero, OpenVINO, Arc 140V execution, NPU
execution, BitNet inference, parity, or benchmark performance.

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
- The canonical BitNet GGUF fixture was not present at the checked local paths, so CPU BitNet validation is recorded as `blocked_preflight`.

## Claim Boundary

Detection is not execution. These artifacts only establish same-machine hardware
visibility and the absence of the runtime tooling needed for OpenCL, Level Zero,
and OpenVINO claims on this Windows host at capture time.
