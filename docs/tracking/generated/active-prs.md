<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-OP-001 | #4620 | `codex/lunar-lake/LNL258V-OP-001-operator-readiness` | Add a Lunar Lake operator readiness command that reads the committed 258V artifact bundle, emits an explicit route-policy/readiness receipt with dense Qwen CPU as the default ask path, BitNet CPU as the strict reference path, OpenVINO GPU/NPU as dense SLM candidates, fallback_used=false evidence, answer-gate evidence, phase evidence, route reasons, and strict claim boundaries without running new inference or claiming speedup, Arc/NPU acceleration, full BitNet inference on accelerators, or QK256 accelerator decode. |
