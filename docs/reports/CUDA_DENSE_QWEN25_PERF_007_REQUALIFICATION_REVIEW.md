# CUDA-DENSE-QWEN25-PERF-007 Requalification Review

Date: 2026-05-19
Campaign item: CUDA-DENSE-QWEN25-PERF-007
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen2.5-0.5b-instruct-q8_0
Coverage row: `dense_qwen25_05b_q8_cuda`
Linked plan: `plans/native-rust-inference/dense-qwen25.md`
Linked spec: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`

## Summary

This review consumes the committed Qwen2.5 current-source benchmark evidence,
the persistent-handle receipt alias work, and the logits-transfer accounting
work. It does not run a new benchmark and it does not promote any new claim.

Decision:

- accepted speed profiles: none;
- `benchmark_qualified=false`;
- `speedup_claim=false`;
- `full_residency_claim=false`;
- `server_ready=true` remains exact-profile only for the existing
  non-streaming shared-engine `/v1/chat/completions` receipt;
- `dense_regular_llm_cuda_proof=true` remains scoped to Qwen2.5;
- `bitnet_packed_i2s_qk256_proof=false`.

The blocker is real, not procedural: the reviewed CUDA totals are still slower
than the same-artifact CPU totals, pure H2D event timing is still unavailable,
and the current logits-transfer accounting records full-logits D2H until a
device top-k or greedy sampler exists.

## Evidence

Primary committed evidence:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/dense-qwen25-q8-one-token-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-short-decode-current-source.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-warm-session-current-source.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-benchmark-qualification-current-source.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-17/server-strict-dense-qwen25-q8-smoke.json
docs/reports/CUDA_DENSE_054_BENCHMARK_QUALIFICATION_CURRENT_SOURCE.md
docs/reports/CUDA_DENSE_QWEN25_RESIDENCY_BOTTLENECKS.md
docs/reports/CUDA_SERVER_005_DENSE_QWEN_EXACT_PROFILE_SERVER_READY.md
```

Supporting optimization closeouts:

```text
CUDA-DENSE-QWEN25-OPS-002: persistent-handle receipt aliases validated for
warm sessions.

CUDA-DENSE-QWEN25-OPS-003: logits-transfer accounting is exposed and validated,
but `device_to_host_bytes_reduced=false` remains correct because the CPU sampler
still requires full logits until a device top-k sampler exists.
```

## Exact-Profile Speed Review

The current-source benchmark review already rejected every reviewed profile:

| Profile | CPU mean total ms | CUDA mean total ms | CPU/CUDA ratio | H2D envelope ms | D2H logits ms | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `one_token` | 2872.8428 | 3978.5710 | 0.7221 | 8071.3571 | 0.9181 | `not_accepted` |
| `short_decode_8` | 3528.0687 | 4199.9896 | 0.8400 | 4319.3893 | 6.9240 | `not_accepted` |
| `warm_session_3_turns` | 4596.1352 | 5034.9288 | 0.9129 | 3936.0470 | 24.5661 | `not_accepted` |

No later committed receipt changes those comparator totals. OPS-002 improves
the receipt surface for persistent-handle evidence, and OPS-003 records the
logits-transfer accounting boundary, but neither supplies a new same-artifact
CPU/CUDA comparator with lower CUDA totals or reduced D2H bytes.

The H2D value remains an envelope:

```text
host_to_device_ms_source = wall_clock_model_load_with_cuda_weight_upload
host_to_device_ms_scope = model_load_wall_clock_envelope
host_to_device_ms_includes_non_transfer_overhead = true
pure_host_to_device_ms_source = not_measured_by_dense_qwen_runtime
```

That means it cannot be treated as pure CUDA event copy timing or as an
accepted decomposition of TTFT.

## Residency Review

Persistent-handle aliases are useful support evidence, but they do not prove
full CUDA residency. A full-residency promotion still needs a receipt that maps
the relevant phases and request lifecycle:

- model handle;
- CUDA context;
- uploaded weights;
- workspace reuse;
- KV cache;
- norm, RoPE, attention, MLP, and LM head phases;
- logits or device-side selection;
- request/session lifecycle.

Current committed Qwen2.5 receipts keep `fallback_used=false`, but they do not
prove every required phase is resident. `full_residency_claim=false` remains the
only supportable claim.

## Decision

This review rejects requalification for speed and full residency.

May claim:

- Qwen2.5 remains product CLI-ready for the bounded RTX 5070 Ti
  `dense_regular_llm_cuda` ask/chat lane.
- Qwen2.5 remains exact-profile server-ready only for the refreshed
  non-streaming shared-engine `/v1/chat/completions` receipt.
- Qwen2.5 has governed current-source benchmark review evidence with explicit
  `not_accepted` decisions for the reviewed profiles.
- Qwen2.5 receipts expose persistent-handle aliases and logits-transfer
  accounting boundaries.

Must not claim:

- accepted Qwen2.5 CUDA speedup;
- `benchmark_qualified=true`;
- full CUDA residency;
- pure H2D event copy timing;
- broad dense GGUF server readiness;
- BitNet packed I2_S/QK256 proof from dense CUDA evidence.

## Next Proof

The next proof should be a refreshed exact-profile comparator after at least one
real runtime optimization. The receipt should include:

- reduced D2H bytes from a device top-k or greedy sampler, or an explicit
  explanation if full logits remain required;
- pure H2D event timing separated from model-load wall-clock overhead;
- repeated same-artifact CPU and RTX 5070 Ti CUDA timings;
- launch count, kernel timing, D2H/H2D bytes and timing, and VRAM high-water;
- a phase residency map for KV, attention, MLP, LM head, logits, and request
  lifecycle.

Until that lands, the model coverage matrix should keep
`benchmark_qualified=false`, `speedup_claim=false`, and
`full_residency_claim=false`.

## Validation

This report is derived from committed receipts and reports only. It did not run
new inference, benchmark, server, CUDA, tokenizer, loader, or kernel code.

```powershell
rtk cargo run --locked -p xtask --no-default-features -- check-model-coverage
rtk git diff --check
```
