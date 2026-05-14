# Apple M3 MacBook Air Dense Qwen Smoke

Date: 2026-05-14
Work item: `M3MBA-004A`

## Result

The dense Qwen2.5 0.5B Q8_0 smoke path passed on the Apple M3 MacBook Air CPU/NEON lane.

Evidence receipt: `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-smoke.json`

Receipt-check summary:

```json
[
  {
    "path": "ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-smoke.json",
    "artifact_kind": "slm_apple_m3_air_warm_session",
    "requested_backend": "apple-m3-air-cpu-neon",
    "selected_backend": "apple-m3-air-cpu-neon",
    "runtime_api": "cpu",
    "fallback_used": false,
    "prompt_count": 14,
    "generated_tokens": 152,
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
- Power: AC power, internal battery 100%, charged
- Thermal status after run: no thermal, performance, or CPU power warning recorded by `pmset -g therm`
- Battery sensor sample after run: raw battery temperature 30.54 C, virtual temperature 32.19 C
- Storage before run window: 61,706,220 KiB available on `/System/Volumes/Data`
- Storage after run window: 58,690,844 KiB available on `/System/Volumes/Data`

The storage delta includes local release build artifacts and the cached model. No model binary is committed.

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

## Smoke Settings

- Command: `mac validate`
- Backend: `apple-m3-air-cpu-neon`
- Runtime API: `cpu`
- Fallback used: false
- Corpus: `ci/quality/apple-m4-slm-quality-corpus.yaml`
- Corpus artifact kind: `apple_m4_slm_quality_corpus`
- Corpus cases: 7
- Repeat runs: 2
- Prompt template: `qwen2.5`
- Generation: deterministic greedy top-1
- Temperature: 0.0
- Max new tokens: 16 from corpus defaults

## Timing Snapshot

- Total session: 48,111.906 ms
- Model load: 4,287.698 ms
- Tokenizer load: 74.319 ms
- Generated tokens: 152
- Prompt count: 14
- Cold session throughput: 3.159 generated tok/s
- Warm prompt throughput: 3.514 generated tok/s
- Decode throughput: 13.342 generated tok/s

These timings are only for this model, corpus, backend, and host context. They are not a broad M3 performance claim.

## Claim Boundary

This receipt is dense Qwen SLM smoke evidence for the Apple M3 MacBook Air CPU/NEON lane only. It does not claim BitNet behavior, full Metal inference, MPSGraph inference, Neural Engine execution, QK256 support, speedup, or replacement of the Apple M4 Mac mini timing envelope.
