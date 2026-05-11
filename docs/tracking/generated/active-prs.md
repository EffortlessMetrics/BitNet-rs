<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| server-real-inference | SERVER-005 | #4490 | `codex/server-real-inference/SERVER-005-shared-engine-runtime` | Wire one non-streaming server chat-completions request path to the same validated local inference engine surface used by CLI ask/chat for already verified product-ready models, preserving strict model verification, tokenizer/prompt authority, planner route, fallback rejection, per-request receipt semantics, and explicit unavailable responses when no verified engine configuration is present. |
