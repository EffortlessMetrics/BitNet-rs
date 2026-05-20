# BITNET-SPEC-NPU-COLD-WARM-CACHE

Status: draft
Proposal: `docs/proposals/BITNET-PROP-0007-npu-productization.md`
Plan: `plans/npu/implementation-plan.md`

## Purpose

Define the NPU cold/cache/warm timing contract. NPU routes must separate
first-ever compile/load costs from cached and resident use so hot-path numbers do
not become misleading cold one-off claims.

## Required timing fields

```json
{
  "npu_timing": {
    "driver_version": "...",
    "openvino_version": "...",
    "pipeline_construct_ms": 0,
    "first_ever_compile_and_infer_ms": 0,
    "cached_pipeline_construct_ms": 0,
    "cache_hit": true,
    "cache_mode": "OPTIMIZE_SPEED",
    "cache_dir": ".npucache",
    "blob_exported": false,
    "blob_path": null,
    "first_token_ms": 0,
    "decode_total_ms": 0,
    "steady_tok_per_s": 0,
    "warm_second_ask_total_ms": 0,
    "resident_10x_total_ms": 0
  }
}
```

## Profiles

- `npu_cold_one_off`
- `npu_cached_one_off`
- `npu_warm_second_ask`
- `npu_resident_10x_ask_short`
- `npu_resident_warm_chat`
- `npu_low_power_short_answer`

## Promotion rules

- Do not promote NPU for cold one-off asks while compile/load dominates.
- Promote only warm/resident profiles where cache/resident timing and quality pass.
- Record `CACHE_DIR`, `CACHE_MODE`, `BLOB_PATH`, and `EXPORT_BLOB` whenever used.
- Separate first-ever compile and cached compile/load paths.
- Record driver and OpenVINO versions with timing receipts.
- If power, thermal, or NPU utilization cannot be measured, record an explicit unavailable reason.

## OpenVINO context

OpenVINO NPU model caching is relevant because first-ever inference includes
compilation, device load/init, and first inference, while cached paths can import
or reuse compiled artifacts. The NPU contract therefore treats cache configuration
as proof metadata, not an optional benchmark note.
