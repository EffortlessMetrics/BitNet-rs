<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-043 | #5794 | `codex/slm-cpu-043-q8-packed-linear-prototype` | Use the SLM-CPU-042 eager Q8_0 dequant-to-F32-before-Candle boundary to attempt the first behavior-preserving Q8_0 dense linear locality implementation, such as a packed Q8_0 sidecar or dequant-fused dense-linear prototype on a narrow path. The slice must compare against the existing eager F32 path and preserve generated IDs, decoded text, strict GGUF tokenizer authority, selected CPU backend/kernel, model SHA, and fallback=false before claiming any bounded improvement. If no safe runtime change is made, it must produce a concrete no-change blocker artifact naming the API or layout gap for the next slice. It must not claim sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
