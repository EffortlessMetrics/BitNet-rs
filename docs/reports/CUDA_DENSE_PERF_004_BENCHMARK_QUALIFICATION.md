# CUDA-DENSE-PERF-004 Benchmark Qualification Review

## Scope

`CUDA-DENSE-PERF-004` reviews the current dense Qwen RTX 5070 Ti benchmark
evidence without upgrading any speed claim.

The reviewed evidence is useful:

- `CUDA-DENSE-PERF-001` records a governed dense Qwen CUDA benchmark baseline;
- `CUDA-DENSE-PERF-002` records repeated same-artifact CPU/CUDA comparator runs
  for one-token, short-decode, and warm-session profiles;
- `CUDA-DENSE-PERF-003` adds measured device-to-host logits download timing to
  the strict CUDA runtime receipts.

The evidence is still baseline/comparator evidence. It is not an accepted
speedup claim.

## Receipt

Primary receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-gguf-qwen-benchmark-qualification.json
```

Inputs reviewed:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-cuda-benchmark-baseline.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-gguf-qwen-repeated-comparator.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-003-transfer-timing/dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-003-transfer-timing/dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-003-transfer-timing/dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json
```

## Decision

Result:

```text
speedup_claim=false
benchmark_qualified_speedup=false
qualification_decision.status=not_accepted
```

No dense Qwen profile is upgraded by this review.

## Reviewed Profiles

| Profile | CPU mean total ms | CUDA mean total ms | CPU/CUDA ratio | D2H logits ms | Decision | Blocking gap |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `one_token` | 2872.8428 | 3978.5710 | 0.7221 | 0.8534 | Not accepted | CUDA mean is slower than CPU mean; H2D timing remains unmeasured; no profile-specific threshold is accepted. |
| `short_decode_8` | 3528.0687 | 4199.9896 | 0.8400 | 6.3089 | Not accepted | CUDA mean is slower than CPU mean; H2D timing remains unmeasured; no profile-specific threshold is accepted. |
| `warm_session_3_turns` | 4596.1352 | 5034.9288 | 0.9129 | 18.7415 | Not accepted | CUDA mean is slower than CPU mean; H2D timing remains unmeasured; no profile-specific threshold is accepted. |

## Claim Boundary

May claim:

- a dense Qwen benchmark qualification review exists for the current baseline,
  repeated comparator, and D2H transfer timing receipts;
- reviewed dense Qwen profiles are same-artifact, deterministic,
  fallback-free, and generated-token matched;
- D2H logits download timing is measured for reviewed strict CUDA runtime
  receipts.

Must not claim:

- accepted dense Qwen CUDA speedup;
- `benchmark_qualified_speedup=true`;
- full CUDA residency;
- server readiness;
- BitNet packed I2_S/QK256 proof from dense CUDA evidence;
- host-to-device transfer timing is measured.

## Next Work

The next performance work should either:

- measure host-to-device transfer timing for dense Qwen runtime paths; or
- reduce the dense Qwen CUDA total time before attempting another
  profile-specific speed qualification review.

## Validation

Expected validation:

```text
cargo run --locked -p bitnet-bench-receipts --bin dense_qwen_cuda_benchmark_qualification_receipt --no-default-features
cargo test --locked -p bitnet-bench-receipts --no-default-features dense_gguf_qwen_benchmark_qualification -- --nocapture
cargo fmt -p bitnet-bench-receipts -- --check
cargo run --release --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
cargo run --release --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```
