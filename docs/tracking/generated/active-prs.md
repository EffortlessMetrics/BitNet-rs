<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-079 | #291 | `codex/slm-cpu-079-post-aligned-matvec-artifact` | Capture or ingest the first real i5-8250U Qwen3 Q8_0 warm-session artifact after the SLM-CPU-078 aligned packed Q8_0 exact-tensor matvec implementation. The artifact must prove prompt IDs, generated IDs, decoded text, model SHA, tokenizer source/strictness, selected CPU backend/kernel identity, dense hook identity, and fallback_used=false remain unchanged against the Qwen3 Q8_0 appliance oracle, and must serialize enough packed-sidecar instrumentation counters to classify whether aligned packed matvec compute improved, regressed, or remains blocked. It must not claim speedup or sustained throughput without a real before/after artifact, must not enable packed Q8_0 by default, must not broaden beyond the exact tensor, and must not touch Q4/Q5, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths. |
