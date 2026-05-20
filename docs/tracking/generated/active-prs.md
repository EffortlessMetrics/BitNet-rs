<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-064 | #6138 | `codex/slm-cpu-064-first-swarm-intake-review` | Review the first audited Kaby SLM artifact package returned from bitnet-rs-swarm under the SLM-CPU-063 intake gate. The package must include candidate_summary.json, before_receipt.json, after_receipt.json, equivalence_report.json, timing_report.json or timing_not_claimed.json, and source_commit.txt. The review must record whether the package preserves model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook-selection identity, and fallback_used=false before any BitNet-rs runtime promotion item is opened. The slice is release/evidence intake only and must not implement runtime compute or claim speedup, sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
