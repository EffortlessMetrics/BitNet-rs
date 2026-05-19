<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-051 | #5897 | `codex/slm-cpu-051-selector-readiness-impl` | Add the behavior-preserving selector-readiness gate for the packed Q8_0 sidecar lane after SLM-CPU-050. The slice must consume the generated-ID/text equivalence gate and production-compute-hook availability report to produce a machine-checkable selector update readiness artifact that can say whether all evidence is present before a later runtime selector change. It must keep the production selector on eager_f32_candle, keep sidecar_runtime_compute_allowed=false, and make any remaining blockers explicit. It may add readiness structs, fixture tests, and dashboard/tracker documentation. It must not execute packed Q8_0 sidecar compute in production, claim speedup, alter generated IDs/text, change tokenizer/backend semantics, start Q4/Q5 runtime support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths. |
