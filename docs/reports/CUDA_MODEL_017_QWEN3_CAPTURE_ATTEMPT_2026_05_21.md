# CUDA-MODEL-017 Qwen3 Capture Attempt - 2026-05-21

## Scope

CUDA-MODEL-017 requires repeated same-artifact Qwen3 CPU/CUDA comparator source receipts for:

- `one_token`
- `short_decode_8`
- `short_decode_32`
- `warm_session_3_turns`
- `decode_128_from_warm_context`

This attempt was limited to unblocking and starting source receipt capture for the exact Qwen3 0.6B Q8_0 artifact on the Windows 9950X3D + RTX 5070 Ti lane.

## Environment Observed

- GPU: NVIDIA GeForce RTX 5070 Ti
- Driver: 591.86
- Model: `Qwen3-0.6B-Q8_0.gguf`
- Model SHA-256: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`
- Local model path: `C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf`

## Unblocked First

Two current-source tooling blockers were fixed before attempting CUDA-MODEL-017 receipts:

- PR #238 fixed a Windows CUDA `dedupe_paths` helper collision in `bitnet-cli`.
- PR #241 made the Qwen3 one-token strict CUDA command resolve Qwen3 prerequisite receipt defaults instead of Qwen2.5 defaults.

The second fix was validated with:

```text
cargo test --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli qwen3_capture_defaults
cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
```

Both commands required Visual Studio 2022, CUDA 12.9, and explicit `NVCC_CCBIN`.

## Capture Attempt Result

The Qwen3 one-token strict CUDA command was attempted with the current debug binary after the prerequisite-default fix:

```text
target\debug\bitnet.exe dense-gguf-qwen-one-token-strict-cuda --model <Qwen3 GGUF> --json-out target\qwen3-one-token-defaults-smoke.json
```

Result:

- The command no longer failed on Qwen2.5 prerequisite defaults.
- The command exceeded a 15-minute timeout.
- No JSON receipt was written.
- The remaining `bitnet.exe` process was stopped.

Release capture was then attempted by building `bitnet-cli` with CUDA in release mode.

Results:

- Default target release build exceeded a 20-minute timeout and did not produce `target\release\bitnet.exe`.
- Isolated target release build with `CARGO_TARGET_DIR=target\cuda-model-017-release` exceeded a 30-minute timeout and did not produce `target\cuda-model-017-release\release\bitnet.exe`.
- Fresh cargo processes from the isolated timed-out release build were stopped.

## Claim Boundary

This report does not provide CUDA-MODEL-017 source receipts and does not generate the `qwen3_cuda_repeated_comparator` aggregate.

The following remain false:

- `speedup_claim`
- `benchmark_qualified_speedup`
- `full_cuda_residency_claimed`
- broad dense GGUF readiness
- Qwen2.5 proof inheritance for Qwen3
- BitNet packed I2_S/QK256 proof

## Next Proof Needed

CUDA-MODEL-017 remains blocked on a current-source capture path that can finish at least the Qwen3 `one_token` CPU/CUDA comparator receipt on this machine.

The next attempt should first make one of these true:

- a release CUDA `bitnet-cli` build completes from a clean or isolated target directory, or
- the Qwen3 one-token capture command is instrumented enough to identify where the 15-minute debug run is spending time, without emitting a receipt until the strict proof finishes.

Only after the one-token source receipt finishes should the lane proceed to the remaining four profiles and three-run aggregate generation.
