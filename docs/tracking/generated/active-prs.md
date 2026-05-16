<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-020 | #5108 | `model/slm-cpu-020-smollm2-cpu-sanity` | Retry strict SmolLM2 360M Q8_0 CPU sanity after exact metadata-scoped normalization validation. The slice must keep the generic llama normalization guard fail-closed, preserve exact SmolLM2 artifact and metadata scope, record tokenizer/prompt/generation reachability or the next blocker with fallback_used=false, and keep CPU answer readiness false unless the bounded quality gate passes. |
