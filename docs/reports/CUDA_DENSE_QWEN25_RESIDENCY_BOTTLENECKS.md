# CUDA-DENSE-QWEN25-OPS-001 Qwen2.5 Residency Bottlenecks

Date: 2026-05-19
Campaign item: CUDA-DENSE-QWEN25-OPS-001
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen2.5-0.5b-instruct-q8_0
Coverage row: `dense_qwen25_05b_q8_cuda`
Linked plan: `plans/native-rust-inference/dense-qwen25.md`
Linked spec: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`

## Summary

Dense Qwen2.5 0.5B Q8_0 is already product CLI-ready, exact-profile
server-ready for non-streaming `/v1/chat/completions`, and backed by strict
RTX 5070 Ti `dense_regular_llm_cuda` receipts. This report does not promote
any new claim. It ranks the remaining performance and residency blockers from
the committed one-token, short-decode, warm-session, benchmark review,
H2D/D2H, and server readiness receipts.

The current evidence points to cold setup and upload cost first, then launch and
runtime orchestration, then logits transfer. D2H logits timing is measured and
worth reducing, but it is not the dominant blocker while model-load/upload
envelopes remain measured in seconds and the reviewed CUDA totals are still
slower than same-artifact CPU totals.

## Evidence

Primary receipts and reports:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/dense-qwen25-q8-one-token-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-short-decode-current-source.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-warm-session-current-source.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-benchmark-qualification-current-source.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-17/server-strict-dense-qwen25-q8-smoke.json
docs/reports/CUDA_DENSE_QWEN25_Q8_PRODUCT_AUDIT.md
docs/reports/CUDA_DENSE_PERF_004_BENCHMARK_QUALIFICATION.md
docs/reports/CUDA_DENSE_PERF_005_H2D_TRANSFER_ENVELOPE.md
docs/reports/CUDA_DENSE_PERF_006_H2D_ENVELOPE_QUALIFICATION.md
docs/reports/CUDA_SERVER_005_DENSE_QWEN_EXACT_PROFILE_SERVER_READY.md
```

Current-source benchmark review:

| Profile | CPU mean total ms | CUDA mean total ms | CPU/CUDA ratio | H2D envelope ms | D2H logits ms | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `one_token` | 2872.8428 | 3978.5710 | 0.7221 | 8071.3571 | 0.9181 | `not_accepted` |
| `short_decode_8` | 3528.0687 | 4199.9896 | 0.8400 | 4319.3893 | 6.9240 | `not_accepted` |
| `warm_session_3_turns` | 4596.1352 | 5034.9288 | 0.9129 | 3936.0470 | 24.5661 | `not_accepted` |

The H2D value is intentionally an envelope:

```text
host_to_device_ms_source = wall_clock_model_load_with_cuda_weight_upload
host_to_device_ms_scope = model_load_wall_clock_envelope
host_to_device_ms_includes_non_transfer_overhead = true
pure_host_to_device_ms_source = not_measured_by_dense_qwen_runtime
```

It is not pure CUDA event copy timing and should not be treated as an additive
decomposition of the comparator total.

Current-source runtime receipt shape:

| Receipt | Route | Kernel launches | H2D bytes | D2H bytes | Fallback |
| --- | --- | ---: | ---: | ---: | --- |
| `dense-qwen25-q8-one-token-cuda.json` | `dense_regular_llm_cuda` | 338 | 675710816 | 607744 | `false` |
| `dense-qwen25-q8-short-decode-current-source.json` | `dense_regular_llm_cuda` | 2704 | 675710816 | 4861952 | `false` |
| `dense-qwen25-q8-warm-session-current-source.json` | `dense_regular_llm_cuda` | 8112 | 675710816 | 14585856 | `false` |

Server readiness evidence:

| Field | Value |
| --- | --- |
| Endpoint | `/v1/chat/completions` |
| Request profile | `non_streaming_chat_completion` |
| Route | `dense_regular_llm_cuda` |
| Fallback | `false` |
| Quality gate | `passed` |
| Runtime `server_ready_claimed` | `false` |
| Model coverage server readiness | exact-profile `server_ready=true` |

## Bottleneck Ranking

### 1. Cold model load and H2D upload envelope

The top blocker is cold setup around model load and CUDA weight upload. The
current-source benchmark review records multi-second H2D envelopes for every
profile, while all reviewed CUDA mean totals remain slower than same-artifact
CPU means. Because the H2D value includes non-transfer overhead, this is a
lifecycle and accounting blocker, not just a raw copy-speed blocker.

Next proof: `CUDA-DENSE-QWEN25-OPS-002` should prove persistent handles with
`model_loaded_once=true`, `cuda_context_once=true`, `weights_uploaded_once=true`,
`per_request_model_load=false`, `workspace_reused=true`, and
`fallback_used=false`.

### 2. Per-request setup and workspace reuse

The warm-session proof demonstrates a bounded loaded-once session shape for the
strict runtime proof path, but the server and ordinary request lifecycle still
need explicit persistent-handle evidence before the product can claim repeated
requests avoid reloads or reuploads. Exact-profile server readiness proves the
non-streaming endpoint can answer with receipts; it does not prove low overhead,
concurrency, broad readiness, or persistent server-side residency.

Next proof: record ask/chat/server receipts that show the same loaded handles
are reused across multiple user requests without fallback or claim promotion.

### 3. Kernel launch count and runtime orchestration

Kernel launches scale with the profile: 338 launches for one token, 2704 for
short decode, and 8112 for the three-turn warm session. The launch count and
runtime orchestration remain plausible contributors after setup is removed,
especially because D2H timing is small relative to the multi-second CUDA totals.

Next proof: add per-profile launch counts, kernel timing, and per-token wall
time to optimization receipts so the next review can separate launch overhead
from math throughput.

### 4. D2H logits transfer

D2H logits transfer is measured and grows with generated work: roughly 0.9 ms,
6.9 ms, and 24.6 ms in the current-source review. It is not the first blocker
while H2D/setup envelopes are seconds, but it is still a bounded optimization
target once persistent handles land.

Next proof: `CUDA-DENSE-QWEN25-OPS-003` should reduce full-logits D2H copies
when greedy or top-k proof is sufficient, preserving selected-token equality
and top-k evidence.

### 5. KV movement and full residency

The strict runtime receipts keep `fallback_used=false` and
`full_cuda_residency_claimed=false`. Some dense tensor residency evidence exists,
but current receipts do not prove every relevant phase is fully resident across
KV, norm, RoPE, attention, MLP, LM head, logits, and request lifecycle state.

Next proof: a residency receipt should identify which phases are resident, which
phases still move through host memory, and whether KV persists across turns and
requests.

### 6. Server overhead

The Qwen2.5 server receipt is valuable but narrow. It supports exact-profile
readiness for non-streaming `/v1/chat/completions`; it does not qualify broad
dense GGUF server readiness, streaming, concurrency, speed, or full residency.

Next proof: attach per-request timing and lifecycle fields to server receipts so
server overhead can be separated from runtime setup and decode cost.

## Claim Boundary

May claim:

- Qwen2.5 has strict RTX 5070 Ti `dense_regular_llm_cuda` one-token,
  short-decode, warm-session, benchmark review, and exact-profile server
  readiness receipts.
- The current-source benchmark review keeps all reviewed speedup decisions
  `not_accepted`.
- H2D model-load envelope and D2H logits timings are recorded.
- The next optimization target is persistent handles before logits/top-k
  transfer reduction.

Must not claim:

- accepted dense Qwen CUDA speedup;
- `benchmark_qualified_speedup=true`;
- full CUDA residency;
- broad dense GGUF server readiness;
- official BitNet packed I2_S/QK256 proof from dense CUDA evidence;
- pure CUDA event H2D copy timing.

## Next Work

1. `CUDA-DENSE-QWEN25-OPS-002`: persistent handles and workspace reuse.
2. `CUDA-DENSE-QWEN25-OPS-003`: logits/top-k transfer reduction.
3. `CUDA-DENSE-QWEN25-PERF-007`: exact-profile requalification review after
   the two optimization PRs land.

## Validation

This report is derived from committed receipts and reports only. It did not run
new inference, benchmark, server, CUDA, tokenizer, loader, or kernel code.

```powershell
rtk python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/dense-qwen25-q8-one-token-cuda.json
rtk python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-short-decode-current-source.json
rtk python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-warm-session-current-source.json
rtk python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-benchmark-qualification-current-source.json
rtk python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-17/server-strict-dense-qwen25-q8-smoke.json
rtk git diff --check
```
