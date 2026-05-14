# Apple M3 MacBook Air Dense Qwen Operator Profile

Date: 2026-05-14
Work item: `M3MBA-004B`

## Result

The dense Qwen2.5 0.5B Q8_0 bounded operator profile passed on the Apple M3 MacBook Air CPU/NEON lane.

Aggregate receipt: `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-operator.json`

Per-profile receipts:

- `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-operator-profiles/warm_16.json`
- `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-operator-profiles/warm_32.json`
- `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-operator-profiles/warm_64.json`

Receipt-check summary:

```json
[
  {
    "path": "ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-operator.json",
    "artifact_kind": "apple_m3_air_slm_operator_profiles",
    "requested_backend": "apple-m3-air-cpu-neon",
    "selected_backend": "apple-m3-air-cpu-neon",
    "runtime_api": "cpu",
    "fallback_used": false,
    "prompt_count": 3,
    "generated_tokens": 166,
    "passed": true,
    "regression": null
  }
]
```

## Host Context

- Machine identity: `apple-m3-macbook-air`
- Chip: Apple M3
- CPU cores: 8
- GPU cores: 10
- Memory: 16 GiB unified memory
- macOS: 26.3.1, build `25D2128`
- Power before and after run: AC power, internal battery 100%, charged
- Thermal status after run: no thermal, performance, or CPU power warning recorded by `pmset -g therm`
- Battery sensor sample after run: raw battery temperature 30.49 C, virtual temperature 31.69 C
- Storage before run window: 60,640,268 KiB available on `/System/Volumes/Data`
- Storage after run window: 58,380,308 KiB available on `/System/Volumes/Data`

The storage delta includes local release build artifacts and receipt outputs. No model binary is committed.

## Model And Tokenizer

- Model id: `qwen2.5-0.5b-instruct-q8_0`
- Model file: `qwen2.5-0.5b-instruct-q8_0.gguf`
- Format: GGUF
- Architecture: `qwen2`
- Quantization: `Q8_0`
- SHA-256: `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`
- Size: 675,710,816 bytes
- Tokenizer source: GGUF metadata
- Tokenizer authority: present, strict
- BOS/EOS: 151643 / 151645

## Operator Settings

- Command: `mac validate --profile-set operator`
- Backend: `apple-m3-air-cpu-neon`
- Runtime API: `cpu`
- Fallback used: false
- Profile set: `operator`
- Profiles: `warm_16`, `warm_32`, `warm_64`
- Prompt count per profile: 3
- Generation: deterministic greedy top-1
- Temperature: 0.0
- Allocation audit: enabled
- Profile execution model: one warm-session run per token budget
- Comparison status: diagnostic-only timing for this M3 Air, model, backend, and profile set

## Timing Snapshot

| Profile | Max Tokens | Generated Tokens | Total Session ms | Warm Prompt ms | Decode ms | Decode tok/s | Warm Prompt tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `warm_16` | 16 | 34 | 13,772.780 | 8,659.309 | 2,590.497 | 13.125 | 3.926 |
| `warm_32` | 32 | 50 | 14,973.999 | 10,454.173 | 3,833.620 | 13.043 | 4.783 |
| `warm_64` | 64 | 82 | 17,104.071 | 12,319.152 | 6,033.050 | 13.592 | 6.656 |

First-token mean by profile:

- `warm_16`: 2,132.333 ms
- `warm_32`: 2,339.667 ms
- `warm_64`: 2,196.000 ms

These timings are only for this model, profile set, backend, and host context. They are not a broad M3 performance claim and do not replace the Apple M4 Mac mini envelope.

## Allocation Audit

The aggregate allocation audit uses process-global allocator counter deltas and records optimization as deferred. Top aggregate hotspots by allocated bytes:

| Component | Alloc Count | Alloc Bytes |
|---|---:|---:|
| `prompt_setup` | 1,369,684 | 7,421,520,002 |
| `decode_total` | 2,698,538 | 3,032,603,117 |
| `prompt_prefill` | 5,055,600 | 2,919,654,816 |
| `model.forward` | 2,689,200 | 2,728,683,616 |
| `prompt_tokenize` | 13,681,054 | 1,139,431,090 |

The allocation audit is diagnostic. It does not claim an optimization or performance improvement.

## Claim Boundary

This receipt is dense Qwen SLM operator-profile evidence for the Apple M3 MacBook Air CPU/NEON lane only. It does not claim BitNet behavior, full Metal inference, MPSGraph inference, Neural Engine execution, QK256 support, speedup, or replacement of the Apple M4 Mac mini timing envelope.
