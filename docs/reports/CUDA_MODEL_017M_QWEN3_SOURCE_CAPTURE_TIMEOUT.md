# CUDA-MODEL-017M Qwen3 Source Capture Timeout

Status: diagnostic source-capture attempt recorded; no CUDA-MODEL-017 source
receipt was emitted

## Scope

`CUDA-MODEL-017` requires repeated same-artifact Qwen3 CPU/CUDA comparator
source receipts for these profiles:

- `one_token`
- `short_decode_8`
- `short_decode_32`
- `warm_session_3_turns`
- `decode_128_from_warm_context`

Each profile requires at least three source receipts before the
`qwen3_cuda_repeated_comparator` aggregate can be generated. This report records
that current source still cannot emit the first strict Qwen3 CUDA one-token
source receipt on the Windows 9950X3D + RTX 5070 Ti lane.

## Manifest Preflight

The repeated comparator manifest command passed under WSL using the current
swarm source:

```powershell
rtk wsl -d Ubuntu -- bash -lc 'cd /mnt/c/b/bitnet-rs-swarm && CARGO_TARGET_DIR=/mnt/d/codex-targets/bitnet-campaign-generate-342-wsl cargo run --locked -p bitnet-bench-receipts --no-default-features --bin qwen3_cuda_repeated_comparator_receipt -- --print-manifest'
```

The manifest requires source receipts under:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-01/
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-02/
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-03/
```

No matching `qwen3-perf-016` source receipts are committed yet. A deliberate
aggregate probe with run-01 paths failed closed with missing-input errors for
all five profiles, as expected.

## Source Capture Attempt

The local Qwen3 artifact exists at:

```text
C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf
```

The attempted strict CUDA one-token source capture used the current post-#335
CUDA debug binary:

```powershell
D:\codex-targets\bitnet-qwen-rmsnorm-fused-op-cuda\debug\bitnet.exe dense-gguf-qwen-one-token-strict-cuda `
  --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf `
  --device nvidia-rtx-5070-ti-cuda `
  --all-layer-plan ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-cuda-all-layer-plan.json `
  --model-boundary-fixtures ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-model-boundary-fixtures.json `
  --kv-cache-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-kv-cache-policy.json `
  --sampling-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-sampling-policy.json `
  --json-out D:\codex-runs\cuda-model-017-source-probe\run-01\qwen3-0_6b-one-token-cuda.json
```

The command timed out after 15 minutes. The output directory contained no source
receipt JSON. Process inspection showed a live `bitnet.exe` process, which was
terminated with:

```powershell
taskkill /PID 895004 /T /F
```

## Blocker Classification

The previous CUDA-MODEL-017L trace-heavy diagnostic stopped before
`block.attention_norm_fused_rms_finish` on the CUDA target. This normal
non-trace source-capture attempt also failed to emit the required one-token
source receipt.

The next implementation boundary should therefore target Qwen3 CUDA layer-0
RMSNorm execution in the normal source-capture path, not the repeated comparator
aggregate generator. The aggregate generator is behaving correctly by refusing
to produce `qwen3_cuda_repeated_comparator` without source receipts.

## Claim Boundary

This report proves only that current source still lacks the first required
Qwen3 CUDA-MODEL-017 source receipt. It does not prove hardware execution or
promote any product claim.

These remain false:

- Qwen3 speedup
- Qwen3 benchmark-qualified speed
- Qwen3 full CUDA residency
- Qwen3 broad dense GGUF readiness
- Qwen3 server readiness from this run
- Qwen2.5 proof inheritance
- BitNet packed I2_S/QK256 proof
- `qwen3_cuda_repeated_comparator` aggregate availability
