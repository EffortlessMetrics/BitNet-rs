<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-046 | #5860 | `codex/slm-cpu-046-q8-sidecar-dispatch-selector` | Add the first behavior-preserving dense-linear dispatch selector contract after SLM-CPU-045 by making the runtime choose the existing eager F32 path explicitly while exposing a packed Q8_0 sidecar candidate only as unavailable until generated-ID, decoded-text, strict tokenizer authority, selected CPU backend/kernel, model SHA, and fallback=false equivalence gates pass. The slice may add selector structs, reasons, fixture tests, and dashboard/tracker documentation, but must not execute packed Q8_0 sidecar compute, claim speedup, alter generated IDs/text, change tokenizer or backend semantics, start Q4/Q5 runtime support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths. |
