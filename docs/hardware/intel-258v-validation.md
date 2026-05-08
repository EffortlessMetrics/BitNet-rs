# Intel Core Ultra 7 258V Validation Profile

## Purpose

This file defines the validation bundle for the Core Ultra 7 258V Lunar Lake laptop. The machine is a tri-device validation platform:

| Device | Proof lane |
|---|---|
| CPU | `intel-258v-cpu-avx2` / `cpu-avx2` BitNet CPU lead, strict validation, and fallback |
| Integrated GPU | `intel-arc-140v-opencl` and `intel-arc-140v-openvino-gpu` |
| NPU | `intel-npu-openvino` / `intel_258v_npu_openvino` |

The 258V laptop should not be treated as a single generic Intel accelerator.

The 258V CPU lane is the BitNet CPU lead. It owns strict real-GGUF BitNet CPU validation, scalar-vs-AVX2 answer parity, phase receipts, and same-machine CPU reference artifacts used by the Arc 140V and Intel NPU lanes. The i5-8250U is now the SLM CPU lead and a legacy/low-power BitNet comparison lane; it does not block new BitNet CPU work.

Platform roadmap:

```text
docs/specs/intel-lunar-lake-258v-platform-roadmap.md
```

## Expected Platform Facts

Expected Core Ultra 7 258V profile:

| Component | Expected value |
|---|---|
| Platform | Lunar Lake |
| CPU | 8 cores / 8 threads |
| CPU topology | 4 P-cores + 4 low-power E-cores |
| CPU backend | CPU AVX2 |
| Memory | Up to 32GB LPDDR5X-8533 shared |
| Integrated GPU | Intel Arc 140V |
| GPU peak | 64 INT8 TOPS |
| GPU PCI device ID | 0x64A0 |
| NPU | Intel AI Boost NPU |
| NPU peak | 47 INT8 TOPS |
| Overall platform peak | 115 INT8 TOPS |

The CPU supports AVX2, but this profile should not assume AVX-512.

## Buildout Contract

The detailed buildout plan for backend identity, Arc 140V probing, platform receipts, and 258V CPU validation is maintained in:

```text
docs/specs/intel-lunar-lake-258v-buildout-plan.md
```

Use this validation profile for manual machine-fact collection. Use the buildout plan for implementation scope and acceptance criteria.

## Required Machine Facts

Record these before moving any 258V hardware lane beyond `scaffold`:

| Fact | Why it matters |
|---|---|
| Native Windows, native Linux, or WSL | Do not assume WSL can see the NPU. |
| OpenVINO version | NPU and GPU plugin support is version-sensitive. |
| Intel NPU driver version | Required for NPU receipts. |
| OpenVINO `available_devices` | Should show CPU/GPU/NPU when fully visible. |
| Arc 140V OpenCL visibility | Determines iGPU kernel lane viability. |
| Level Zero visibility | Future lower-level/SYCL path. |
| OpenVINO `GPU.0` full name | Confirms Arc 140V reference target. |
| NPU `compile_model(..., "NPU")` success | Compile path proof. |
| Static-shape tiny graph result | Runtime smoke proof. |
| Shared memory pressure | 32GB LPDDR5X is shared by CPU/GPU/NPU. |
| Power mode / thermal profile | Laptop results depend heavily on power policy. |

## Claim Boundary

- CPU AVX2 correctness does not count as Arc 140V or NPU execution.
- Arc 140V OpenCL execution does not count as NPU execution.
- OpenVINO NPU execution does not count as native OpenCL GPU execution.
- OpenVINO `GPU.0` smoke does not prove BitNet OpenCL kernel acceleration.
- OpenVINO `NPU` smoke does not prove full BitNet inference.
- CPU or GPU fallback cannot count as NPU execution.
- 258V CPU proof is the first priority on this platform; NPU and Arc proofs must compare against the 258V CPU reference when they make BitNet-adjacent parity claims.
- 258V CPU changes may own BitNet CPU sequencing when explicitly scoped; accelerator PRs must not reshape CPU dispatch or QK256 CPU kernels.
- Arc 140V visibility must preserve `requested_backend`, `selected_backend`, runtime API, exact device identity evidence, and `fallback_used=false`; generic Intel GPU visibility is not enough.

## Platform Probe Bundle Artifacts

`LNL258V-002` documents the same-machine probe bundle that later runs should
write under `ci/hardware/intel-258v/<date>/`. These paths are examples and
placeholders for future evidence; adding them to the docs does not commit a
real machine artifact and does not prove runtime execution.

```text
ci/hardware/intel-258v/YYYY-MM-DD/platform-probe.json
ci/hardware/intel-258v/YYYY-MM-DD/arc-140v-runtime-probe.json
ci/hardware/intel-258v/YYYY-MM-DD/arc-140v-openvino-gpu-smoke.json
ci/hardware/intel-258v/YYYY-MM-DD/npu-openvino-runtime-probe.json
ci/hardware/intel-258v/YYYY-MM-DD/platform-comparison-index.json
```

The bundle must keep each lane independently addressable:

| Artifact | Proof stage | Scope | Claim boundary |
|---|---|---|---|
| `platform-probe.json` | `runtime_detected` | OS, CPU, memory, power, OpenVINO device list, shared platform context | Machine visibility only |
| `arc-140v-runtime-probe.json` | `runtime_detected` | Arc 140V OpenCL, Level Zero, OpenVINO `GPU.0`, exact device identity | No OpenCL kernel execution claim |
| `arc-140v-openvino-gpu-smoke.json` | `kernel_smoke_tested` | Tiny static OpenVINO `GPU.0` graph execution with Arc 140V identity and CPU expected-output comparison | No native OpenCL, BitNet, QK256, or acceleration claim |
| `npu-openvino-runtime-probe.json` | `runtime_detected` | OS NPU evidence, OpenVINO `NPU`, driver/compiler/memory properties | No graph execution claim |
| `platform-comparison-index.json` | index only | Links CPU, Arc 140V, and NPU artifacts from the same machine/date | No independent proof claim |

The comparison index should preserve artifact paths and lane identities so later
CPU, GPU, and NPU receipts can be compared without inferring cross-lane proof:

```json
{
  "machine_id": "intel-258v",
  "date": "YYYY-MM-DD",
  "proof_stage": "runtime_detected",
  "artifacts": {
    "platform": "ci/hardware/intel-258v/YYYY-MM-DD/platform-probe.json",
    "arc140v": "ci/hardware/intel-258v/YYYY-MM-DD/arc-140v-runtime-probe.json",
    "arc140v_openvino_gpu": "ci/hardware/intel-258v/YYYY-MM-DD/arc-140v-openvino-gpu-smoke.json",
    "npu": "ci/hardware/intel-258v/YYYY-MM-DD/npu-openvino-runtime-probe.json"
  },
  "lanes": {
    "cpu": "intel-258v-cpu-avx2",
    "gpu": "intel-arc-140v-opencl",
    "openvino_gpu": "intel-arc-140v-openvino-gpu",
    "npu": "intel-npu-openvino"
  },
  "fallback_used": false
}
```

The bundle does not prove BitNet inference, Arc 140V execution, OpenVINO NPU
graph execution, parity, or benchmark performance.

### CLI Platform Probe

Use the CLI probe command to emit the visibility-only platform receipt from the
current machine without launching kernels or compiling OpenVINO graphs:

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- lunar-lake-probe \
  --json-out ci/hardware/intel-258v/YYYY-MM-DD/platform-probe.json
```

The command records `proof_stage=runtime_detected`, `runtime_api=platform_probe`,
`fallback_used=false`, and a `must_not_claim` list. It does not replace the
lane-specific Arc 140V, NPU, CPU BitNet, parity, or benchmark artifacts.

### Arc 140V OpenVINO GPU Smoke

Use the Arc 140V OpenVINO GPU smoke command to emit a tiny fixed-shape graph
receipt from `GPU.0` without loading BitNet models or running native OpenCL:

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- intel-arc-140v-openvino-gpu-smoke \
  --json-out ci/hardware/intel-258v/YYYY-MM-DD/arc-140v-openvino-gpu-smoke.json
```

The command records `proof_stage=kernel_smoke_tested` only when OpenVINO reports
an Arc 140V `GPU.0`, compiles the tiny static graph to that device, and matches
the CPU expected output. It keeps `fallback_used=false`,
`cpu_fallback_allowed=false`, `bitnet_inference=false`, and `qk256_decode=false`.
It does not prove native OpenCL kernels, BitNet inference, or Arc acceleration.

### CPU BitNet Validation Preflight

Use the CPU validation command to emit the Lunar Lake CPU lead artifact without
touching unrelated accelerator surfaces:

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- validate cpu-bitnet \
  --machine intel-258v \
  --model /models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --tokenizer /models/BitNet-b1.58-2B-4T/tokenizer.json \
  --backend cpu \
  --strict \
  --max-tokens 1 \
  --platform-artifact ci/hardware/intel-258v/YYYY-MM-DD/platform-probe.json \
  --json-out ci/hardware/intel-258v/YYYY-MM-DD/cpu-bitnet-validation.json
```

This command is validation-only. If the canonical GGUF or tokenizer is absent,
it writes `proof_stage=blocked_preflight` with a structured blocker. It does not
load BitNet tensors, run QK256/TL2 kernels, decode tokens, or make benchmark
claims.

### CPU Phase Benchmark Receipt

Use the CPU phase benchmark receipt emitter to turn strict CPU proof receipts
into phase-aware 258V CPU artifacts:

```bash
cargo run --locked -p bitnet-bench-receipts \
  --bin cpu_phase_benchmark_receipt \
  --no-default-features \
  -- \
  --strict-proof-receipt ci/hardware/intel-258v/YYYY-MM-DD/strict-bitnet-cpu-proof.json \
  --machine-id intel-258v \
  --hardware-lane intel-258v-cpu-avx2 \
  --selected-backend cpu-rust \
  --model-quant-format QK256/I2_S \
  --platform-artifact ci/hardware/intel-258v/YYYY-MM-DD/platform-probe.json \
  --receipt-out ci/hardware/intel-258v/YYYY-MM-DD/cpu-phase-benchmark.json
```

The first 258V phase receipt is:

```text
ci/hardware/intel-258v/2026-05-07/cpu-phase-benchmark.json
```

It records the available first-token strict CPU timing, selected backend/kernel,
fallback status, CPU feature set, 4 P-core / 4 low-power E-core topology,
shared LPDDR memory context, and Balanced power mode. Profiles that are not
backed by a supplied strict CPU proof remain explicit `not_run` gaps. This is a
phase receipt, not a sustained throughput claim and not an Arc 140V or Intel
NPU performance comparison. The CPU258V-003 profile summary records `smoke_1`
and `first_token` from the one-token proof and keeps `decode_128` and
`prefill_512` as explicit `not_run` gaps until matching strict proofs exist.

Follow-up CPU258V-005 evidence attempts are recorded at:

```text
ci/hardware/intel-258v/2026-05-08/cpu-phase-evidence-attempts.json
```

That artifact records timed-out strict CPU attempts for calibrated
`prefill_512` collection and preserves `decode_128` as `not_run`. It is
blocker evidence only; it does not prove prefill, decode, throughput, Arc 140V,
or Intel NPU performance.

CPU258V-006 adds a warm CPU phase runner so the model and tokenizer can be
loaded once before collecting long `prefill_512` and `decode_128` profile
receipts:

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- \
  --device cpu \
  cpu-phase-warm-session \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --tokenizer models/BitNet-b1.58-2B-4T/tokenizer.json \
  --strict-loader \
  --strict-tokenizer \
  --threads 8 \
  --prefill-prompt-file ci/hardware/intel-258v/YYYY-MM-DD/prefill-512-prompt.txt \
  --decode-tokens 128 \
  --cpu-kernel avx2 \
  --platform-artifact ci/hardware/intel-258v/YYYY-MM-DD/platform-probe.json \
  --json-out ci/hardware/intel-258v/YYYY-MM-DD/cpu-phase-warm-session.json
```

The command emits per-profile strict CPU receipts under
`cpu-phase-warm-session-profiles/`. Those receipts are inputs to
`cpu_phase_benchmark_receipt`; they are phase timing evidence only and do not
claim answer quality, sustained throughput, Arc 140V execution, Intel NPU
execution, or acceleration.

### CPU Answer Template Refresh

CPU258V-007 records the first 258V AVX2 answer-corpus refresh after the CPU
answer lane adopted the BitNet.cpp answer-ready prompt envelope:

```text
ci/hardware/intel-258v/2026-05-08/cpu-answer-corpus-avx2-bitnetcpp-template.json
```

The artifact records five timeout rows with `missing_child_receipt` kernels.
It is blocker evidence only: it shows that the newer answer-ready prompt path
did not complete within the bounded local child-run window. It does not prove
answer quality, scalar/AVX2 parity under the new prompt, sustained throughput,
Arc 140V execution, or Intel NPU execution.

CPU258V-008 adds a bounded answer-corpus case filter so the next 258V refresh
can isolate a single prompt before spending a full-corpus local decode window:

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- \
  --device cpu \
  answer-corpus \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --tokenizer models/BitNet-b1.58-2B-4T/tokenizer.json \
  --cpu-kernel avx2 \
  --case-id arithmetic-single-digit \
  --dump-logit-steps 1 \
  --logits-topk 5 \
  --per-prompt-timeout-seconds 420 \
  --json-out ci/hardware/intel-258v/YYYY-MM-DD/cpu-answer-corpus-avx2-bitnetcpp-template-case.json
```

The aggregate receipt preserves the full corpus `case_count` and records
`selected_case_count` plus `selected_case_ids`. This is diagnostic scope only:
it narrows timeout evidence collection and does not prove selected-case
completion, answer quality, scalar/AVX2 parity, sustained throughput, Arc 140V
execution, or Intel NPU execution.

## Windows PowerShell Bundle

```powershell
$ErrorActionPreference = "Continue"

Write-Host "=== Windows ==="
Get-ComputerInfo | Select-Object OsName, OsVersion, WindowsVersion, CsSystemType

Write-Host "=== CPU ==="
Get-CimInstance Win32_Processor | Format-List Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed

Write-Host "=== Memory ==="
Get-CimInstance Win32_PhysicalMemory | Format-Table Capacity, Speed, Manufacturer, PartNumber

Write-Host "=== Intel GPU / NPU PnP ==="
Get-PnpDevice | Where-Object {
  $_.FriendlyName -match "Arc|140V|NPU|Neural|AI Boost|VPU|Intel.*Graphics"
} | Format-List *

Write-Host "=== OpenCL ==="
where clinfo
clinfo | Select-String -Pattern "Platform Name|Device Name|Device Vendor|Driver Version|OpenCL C"

Write-Host "=== Level Zero / oneAPI ==="
where sycl-ls
sycl-ls
where ze_info
ze_info

Write-Host "=== OpenVINO ==="
python - <<'PY'
import json
import openvino as ov

core = ov.Core()
out = {
    "openvino_version": ov.__version__,
    "available_devices": list(core.available_devices),
    "devices": {}
}
for dev in core.available_devices:
    props = {}
    for prop in [
        "FULL_DEVICE_NAME",
        "SUPPORTED_PROPERTIES",
        "OPTIMAL_NUMBER_OF_INFER_REQUESTS",
        "NPU_DRIVER_VERSION",
        "NPU_COMPILER_VERSION",
        "NPU_DEVICE_TOTAL_MEM_SIZE",
        "NPU_DEVICE_ALLOC_MEM_SIZE",
        "NPU_MAX_TILES",
    ]:
        try:
            props[prop] = str(core.get_property(dev, prop))
        except Exception as e:
            props[prop] = "ERR: " + repr(e)
    out["devices"][dev] = props
print(json.dumps(out, indent=2))
PY
```

## Linux Bundle

```bash
set -eux

echo "=== OS ==="
uname -a
cat /etc/os-release || true

echo "=== CPU ==="
lscpu || true

echo "=== Memory ==="
free -h || true

echo "=== GPU / NPU PCI ==="
lspci -nn | grep -Ei 'vga|display|intel|arc|140v|64a0|npu|vpu|neural|accel' || true

echo "=== DRM render nodes ==="
ls -l /dev/dri/renderD* || true
stat -c "%G %n" /dev/dri/renderD* || true
groups "$USER"

echo "=== accel devices ==="
ls -l /dev/accel || true

echo "=== NPU driver logs ==="
dmesg | grep -Ei 'intel_vpu|ivpu|vpu|npu|accel' | tail -200 || true

echo "=== OpenCL ==="
which clinfo || true
clinfo | grep -Ei 'Platform Name|Device Name|Device Vendor|Device Version|Driver Version|OpenCL C|Max compute units|Global memory size' || true

echo "=== Level Zero / oneAPI ==="
which sycl-ls || true
sycl-ls || true
which ze_info || true
ze_info || true

echo "=== OpenVINO ==="
python3 - <<'PY'
import json
import openvino as ov

core = ov.Core()
out = {
    "openvino_version": ov.__version__,
    "available_devices": list(core.available_devices),
    "devices": {}
}
for dev in core.available_devices:
    props = {}
    for prop in [
        "FULL_DEVICE_NAME",
        "SUPPORTED_PROPERTIES",
        "OPTIMAL_NUMBER_OF_INFER_REQUESTS",
        "NPU_DRIVER_VERSION",
        "NPU_COMPILER_VERSION",
        "NPU_DEVICE_TOTAL_MEM_SIZE",
        "NPU_DEVICE_ALLOC_MEM_SIZE",
        "NPU_MAX_TILES",
    ]:
        try:
            props[prop] = str(core.get_property(dev, prop))
        except Exception as e:
            props[prop] = "ERR: " + repr(e)
    out["devices"][dev] = props
print(json.dumps(out, indent=2))
PY
```

## First Platform Receipt

The first 258V platform receipt should establish visibility only:

```json
{
  "platform": "core-ultra-7-258v",
  "cpu_backend": "intel-258v-cpu-avx2",
  "gpu_backend": "intel-arc-140v-opencl",
  "npu_backend": "intel-npu-openvino",
  "openvino_available_devices": ["CPU", "GPU", "NPU"],
  "openvino_npu_full_name": "...",
  "npu_driver_version": "...",
  "npu_compiler_version": "...",
  "npu_total_mem_size": 0,
  "npu_alloc_mem_size": 0,
  "npu_max_tiles": 1,
  "opencl_arc_140v_visible": true,
  "level_zero_visible": true,
  "npu_visible": true,
  "fallback_used": false,
  "status": "runtime_detected"
}
```

This is not an inference claim. Smoke, parity, and benchmark receipts come later.

The code-facing visibility probe for this first receipt lives in
`bitnet-device-probe` as `probe_lnl258v_platform()`. It emits a JSON-ready
`Lnl258vPlatformProbe` with nested CPU, Arc 140V, NPU, OpenVINO, memory, and
power sections. Unsupported runtime tools must be represented as `false`,
empty lists, or `null` fields rather than panics or fallback claims.

## Ownership

Proof lanes:

- CPU AVX2 remains under CPU runtime proof.
- Arc 140V OpenCL and OpenVINO GPU are owned by the Intel Arc GPU workstream.
- Intel AI Boost NPU and OpenVINO NPU are owned by the Intel NPU workstream.

The platform profile ties the lanes together for comparison, but it does not merge their claims.
