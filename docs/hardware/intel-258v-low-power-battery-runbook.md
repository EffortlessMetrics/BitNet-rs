# Intel 258V Low-Power Battery Runbook

This runbook is the operator checklist for `LNL258V-POWER-006`. It does not add
low-power evidence by itself. It exists so the battery-mode run has clear stop
rules before any route decision or power claim is updated.

## Scope

`POWER-006` is only valid when the Core Ultra 7 258V laptop is running on
battery power. AC or charging samples are blocker evidence, not promotion
evidence.

The run must preserve these boundaries:

- no route promotion unless the promotion lane is explicitly updated;
- no speedup or power-advantage claim from AC-only telemetry;
- no native OpenCL, native NPU, or acceleration claim from OpenVINO receipts;
- no BitNet QK256/I2_S behavior claim from dense Qwen low-power evidence;
- no hidden fallback.

## Preflight

Start from a clean checkout at current `main`, then build the CLI:

```powershell
cargo build --locked -p bitnet-cli --no-default-features --features cpu,full-cli --bin bitnet
```

Confirm Windows reports battery mode before collecting evidence:

```powershell
Get-CimInstance Win32_Battery |
  Select-Object BatteryStatus, EstimatedChargeRemaining, Status |
  Format-List
```

Stop if `BatteryStatus=2` or if the telemetry receipt below reports
`ac_power_inferred=true`.

Before unplugging for the physical run, emit the machine-readable plan receipt
from the current committed blocker evidence:

```powershell
target/debug/bitnet.exe lunar-lake low-power-plan `
  --artifact-root ci/hardware/intel-258v/2026-05-08 `
  --power-profile-evidence lunar-lake-power-profile-evidence.json `
  --blocked-ask-receipt lunar-lake-operator-ask-auto-low-power-blocked.json `
  --battery-telemetry-context lunar-lake-low-power-battery-telemetry-blocked.json `
  --json-out lunar-lake-low-power-battery-plan.json `
  --created-utc <plan-utc> `
  --strict
```

This plan is not battery evidence. Continue only if it records
`operator_plan_ready=true`; if `can_collect_battery_evidence_now=false`, keep
following the strict battery telemetry preflight below before collecting route
samples.

## Battery Start Receipt

Capture the required before-sample with strict battery enforcement:

```powershell
target/debug/bitnet.exe lunar-lake telemetry-context `
  --artifact-root ci/hardware/intel-258v/2026-05-08 `
  --require-battery `
  --json-out lunar-lake-low-power-battery-before.json `
  --created-utc <battery-run-start-utc> `
  --strict
```

Continue only if the receipt records:

- `capture_requirements.battery_mode_required=true`;
- `capture_requirements.battery_mode_sample_recorded=true`;
- `capture_requirements.requirement_satisfied=true`;
- `power.ac_power_inferred=false`.

If strict mode fails after writing a blocked receipt, do not rename it to the
`before` artifact and do not update promotion evidence.

## Route/Profile Samples

Run the low-power route/profile matrix on battery for these route identities:

| Route | Backend | Status Before POWER-006 |
|---|---|---|
| `dense_slm_default_cpu` | `cpu-rust` | control/default route |
| `dense_slm_openvino_gpu_candidate` | `openvino-gpu` | candidate for `low_power` |
| `dense_slm_openvino_npu_candidate` | `openvino-npu` | candidate for `low_power` |

Each route sample must keep `fallback_used=false`, preserve the route identity,
and record answer-gate, timing, memory, power, and thermal context. If a route
falls back or cannot run on battery, keep the failure as blocker evidence.

Do not use the already committed AC-only low-power corpus/profile receipts as
battery evidence. They can remain comparison context only.

## Battery End Receipt

Capture the after-sample immediately after the route/profile run:

```powershell
target/debug/bitnet.exe lunar-lake telemetry-context `
  --artifact-root ci/hardware/intel-258v/2026-05-08 `
  --require-battery `
  --json-out lunar-lake-low-power-battery-after.json `
  --created-utc <battery-run-end-utc> `
  --strict
```

Stop if the after-sample is not a valid battery-mode sample.

## Energy Proxy

Build the energy proxy only from battery-mode before/after telemetry:

```powershell
target/debug/bitnet.exe lunar-lake energy-proxy `
  --artifact-root ci/hardware/intel-258v/2026-05-08 `
  --before-telemetry-context lunar-lake-low-power-battery-before.json `
  --after-telemetry-context lunar-lake-low-power-battery-after.json `
  --route dense_slm_openvino_npu_candidate `
  --profile low_power `
  --sample-count <battery-run-sample-count> `
  --json-out lunar-lake-low-power-energy-proxy.json `
  --created-utc <battery-run-end-utc> `
  --strict
```

The proxy is not a power-advantage claim unless the refreshed power-profile
evidence later qualifies it against the CPU/GPU/NPU route matrix.

## Refresh Artifacts

After valid battery telemetry and route/profile samples exist, rebuild the
power, regression, comparison, and audit surfaces:

```powershell
target/debug/bitnet.exe lunar-lake power-profile `
  --artifact-root ci/hardware/intel-258v/2026-05-08 `
  --route-profile-comparison lunar-lake-route-profile-comparison.json `
  --cold-warm-benchmark lunar-lake-cold-warm-profile-benchmark.json `
  --telemetry-context lunar-lake-power-thermal-context.json `
  --battery-telemetry-context lunar-lake-low-power-battery-after.json `
  --energy-proxy lunar-lake-low-power-energy-proxy.json `
  --json-out lunar-lake-power-profile-evidence.json `
  --created-utc <battery-run-end-utc> `
  --strict

target/debug/bitnet.exe lunar-lake regress `
  --artifact-root ci/hardware/intel-258v/2026-05-08 `
  --answer-corpus-v2 ci/quality/lunar-lake-answer-corpus-v2.yaml `
  --route-profile-comparison lunar-lake-route-profile-comparison.json `
  --cold-warm-benchmark lunar-lake-cold-warm-profile-benchmark.json `
  --durability-bundle lunar-lake-durability-bundle.json `
  --bitnet-semantic-intake lunar-lake-bitnet-semantic-intake.json `
  --power-profile-evidence lunar-lake-power-profile-evidence.json `
  --warm-resident-ask-receipt lunar-lake-operator-ask-auto-npu-warm-resident-math-brief.json `
  --blocked-ask-receipt lunar-lake-operator-ask-auto-low-power-blocked.json `
  --json-out ci/hardware/intel-258v/2026-05-08/lunar-lake-regression-bundle-v2.json `
  --created-utc <battery-run-end-utc> `
  --strict

target/debug/bitnet.exe lunar-lake compare `
  --artifact-root ci/hardware/intel-258v/2026-05-08 `
  --operator-receipt lunar-lake-operator-readiness.json `
  --regression-bundle lunar-lake-regression-bundle-v2.json `
  --json-out ci/hardware/intel-258v/2026-05-08/lunar-lake-operator-comparison.json `
  --created-utc <battery-run-end-utc> `
  --strict
```

Refresh `lunar-lake-excellence-audit.json` only after the generated evidence
actually changes the completion audit.

## Promotion Rule

`low_power` remains blocked unless all of these are true:

- answer gates pass for the route/profile evidence being considered;
- `fallback_used=false`;
- timing is stable for the sampled profile;
- before/after battery telemetry is valid battery-mode evidence;
- the power-profile evidence records benchmark-qualified power advantage;
- strict regression and operator comparison preserve the same decision.

If any condition is missing, keep `low_power` unpromoted and record the blocker.
