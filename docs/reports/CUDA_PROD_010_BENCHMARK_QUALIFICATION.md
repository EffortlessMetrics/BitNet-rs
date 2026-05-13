# CUDA-PROD-010 Benchmark Qualification

## Scope

This report records the governed benchmark qualification review for the official
Microsoft BitNet 2B I2_S/QK256 CUDA lane on the 9950X3D + RTX 5070 Ti bench.

It does not run a fresh benchmark. It consumes the existing strict BitNet CUDA
benchmark receipts and makes the product profile decisions explicit.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/cuda-prod-010-benchmark-qualification.json
```

The receipt is a `strict_cuda_benchmark_qualification_review` and keeps:

```text
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
fallback_used = false
speedup_claim = false
benchmark_qualified_speedup = false
dense_cuda_evidence_used = false
bitnet_packed_i2s_qk256_only = true
```

## User-Facing Inspection

The receipt is inspectable through the normal benchmark and receipt explanation
surfaces:

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  bench \
  --device cuda \
  --cuda-benchmark-receipt ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/cuda-prod-010-benchmark-qualification.json

cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  receipts explain \
  ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/cuda-prod-010-benchmark-qualification.json
```

These commands report the governed benchmark qualification receipt. They do not
run a fresh benchmark and do not create a new speedup claim.

The benchmark receipt report also supports `--format json` for machine-readable
summary output and `--format csv` for profile review rows.

## Profile Decisions

| Profile | Evidence status | Decision | Reason |
| --- | --- | --- | --- |
| `one_token` | missing | not accepted | Product benchmark profile receipt is not committed. |
| `short_decode_8` | single-run baseline | not accepted | Baseline exists, but the profile is not repeated and transfer timing is incomplete. |
| `short_decode_32` | missing | not accepted | Product benchmark profile receipt is not committed. |
| `warm_session_3_turns` | missing | not accepted | Product benchmark profile receipt is not committed. |
| `warm_session_10_turns` | missing | not accepted | Product benchmark profile receipt is not committed. |

No profile is accepted for speedup. There is no global CUDA speed claim.

## Inputs

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-004-benchmark-qualification.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-benchmark.json
```

The first input preserves the earlier strict ask and warm-session review. The
second input provides the existing `short_decode_8` CPU AVX-512 versus RTX 5070
Ti CUDA baseline.

## Claim Boundary

This review is BitNet packed I2_S/QK256 only. Dense regular-LLM CUDA evidence is
excluded and cannot satisfy this receipt.

Generic `cuda`, WGPU, Vulkan, CPU fallback, and hardware visibility do not
satisfy strict RTX 5070 Ti CUDA benchmark qualification.

## Next Evidence

The next benchmark work should add repeated product-profile receipts for:

```text
one_token
short_decode_8
short_decode_32
warm_session_3_turns
warm_session_10_turns
```

Each profile needs same-artifact CPU AVX-512 and RTX 5070 Ti CUDA measurements,
complete transfer timing, power/thermal context where available, and an explicit
accepted or rejected speedup decision.

## Validation

```bash
cargo run --locked -p bitnet-bench-receipts --bin strict_cuda_benchmark_qualification_receipt --no-default-features -- --receipt-out ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/cuda-prod-010-benchmark-qualification.json
cargo test --locked -p bitnet-bench-receipts --no-default-features
cargo test --locked -p bitnet-cli --no-default-features --features full-cli --test cli_arg_tests bench_
cargo test --locked -p bitnet-cli --no-default-features --features full-cli report_only_cuda_benchmark_receipt_skips_startup_backend_selection
cargo check --locked -p bitnet-cli --no-default-features --features cpu,full-cli
cargo fmt -p bitnet-cli -p bitnet-bench-receipts -- --check
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- bench --device cuda --cuda-benchmark-receipt ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/cuda-prod-010-benchmark-qualification.json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- bench --device cuda --cuda-benchmark-receipt ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/cuda-prod-010-benchmark-qualification.json --format json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- bench --device cuda --cuda-benchmark-receipt ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/cuda-prod-010-benchmark-qualification.json --format csv
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/cuda-prod-010-benchmark-qualification.json
python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/cuda-prod-010-benchmark-qualification.json
git diff --check
```

The tracker TOML files parse with Python `tomllib`, including the
`CUDA-PROD-010` active item and in-progress event. The full campaign checker was
also attempted:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
```

That local run failed before manifest validation while building
`sentencepiece-sys`: CMake could not find a `Visual Studio 17 2022` instance.
This is a local native toolchain blocker, not a parsed tracker failure.

The exact CUDA feature command was also probed:

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda --cuda-benchmark-receipt ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/cuda-prod-010-benchmark-qualification.json
```

That local run failed before CLI execution while building `candle-kernels`:
`nvcc fatal: Cannot find compiler 'cl.exe' in PATH`. This is a local MSVC/CUDA
toolchain setup blocker, separate from the governed receipt report path.
