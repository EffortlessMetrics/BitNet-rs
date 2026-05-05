# Hardware Artifact Naming

## Purpose

Hardware receipts and probe outputs should use stable paths so claims can point to concrete artifacts.

No artifact means no hardware claim.

## Path Pattern

```text
ci/hardware/<machine-id>/<date>/<artifact-kind>.json
```

Use ISO dates:

```text
YYYY-MM-DD
```

## Examples

```text
ci/hardware/intel-i5-8250u/2026-05-05/cpu-probe.json
ci/hardware/intel-i5-8250u/2026-05-05/strict-cpu-proof.json

ci/hardware/amd-5700x/2026-05-05/cpu-probe.json
ci/hardware/amd-5700x/2026-05-05/strict-cpu-proof.json
ci/hardware/amd-5700x/2026-05-05/sustained-cpu-benchmark.json

ci/hardware/amd-9950x3d/2026-05-05/cpu-probe.json
ci/hardware/amd-9950x3d/2026-05-05/strict-cpu-proof.json
ci/hardware/amd-9950x3d/2026-05-05/cache-sensitive-benchmark.json

ci/hardware/intel-arc-a770/2026-05-05/opencl-probe.json
ci/hardware/intel-arc-a770/2026-05-05/opencl-smoke.json
ci/hardware/intel-arc-a770/2026-05-05/matmul-i2s-parity.json

ci/hardware/intel-258v/2026-05-05/platform-probe.json
ci/hardware/intel-258v/2026-05-05/arc-140v-opencl-smoke.json
ci/hardware/intel-258v/2026-05-05/npu-openvino-smoke.json

ci/hardware/apple-m4-mac-mini/2026-05-05/metal-probe.json
ci/hardware/apple-m4-mac-mini/2026-05-05/metal-smoke.json
ci/hardware/apple-m4-mac-mini/2026-05-05/mpsgraph-smoke.json

ci/hardware/nvidia-rtx-5070-ti/2026-05-05/cuda-probe.json
ci/hardware/nvidia-rtx-5070-ti/2026-05-05/cuda-smoke.json
ci/hardware/nvidia-rtx-5070-ti/2026-05-05/cuda-parity.json
```

## Artifact Kinds

| Kind | Meaning |
|---|---|
| `probe.json` | Detection/runtime visibility only |
| `smoke.json` | Tiny kernel or graph execution |
| `parity.json` | CPU comparison with tolerance |
| `benchmark.json` | Timing with power/thermal context |
| `proof.json` | Strict receipt-backed inference or kernel proof |

## Templates

Templates for future manual and automated receipts live in:

```text
ci/hardware/_templates/
```

Current templates:

```text
probe.json
smoke.json
parity.json
benchmark.json
strict-bitnet-proof.json
```

Templates are schema starters, not evidence. Replace every `TBD` field and write the completed receipt under `ci/hardware/<machine-id>/<date>/`.

Specific names are allowed when clearer:

```text
cpu-probe.json
opencl-probe.json
openvino-gpu-smoke.json
npu-openvino-smoke.json
metal-probe.json
metal-smoke.json
mpsgraph-smoke.json
cuda-probe.json
cuda-smoke.json
cuda-parity.json
matmul-i2s-parity.json
strict-cpu-proof.json
sustained-cpu-benchmark.json
```

## Rules

- Do not add binary or large artifacts to normal docs PRs.
- Probe artifacts cannot support execution claims.
- Smoke artifacts cannot support parity claims.
- Parity artifacts cannot support full inference claims.
- Benchmark artifacts must follow `docs/hardware/BENCHMARK_PROTOCOL.md`.
- Every artifact must preserve requested backend, selected backend, runtime API, resolved device identity, fallback status, proof stage, and artifact path.
- Every artifact that claims BitNet progress must also preserve model, tokenizer, quantization, kernel family, execution phase, reference path, and BitNet fallback fields from `docs/bitnet/BITNET_RECEIPT_FIELDS.md`.
