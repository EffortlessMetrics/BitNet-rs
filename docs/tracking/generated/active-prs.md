<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-inference-excellence | M4-EXCELLENCE-002 | #5066 | `codex/apple-m4-inference-excellence/M4-EXCELLENCE-002-bitnet-warm-refresh` | Run and commit a second BitNet variable warm-session refresh for the accepted artifact/tokenizer identity so the BitNet warm dashboard group can move from insufficient_history to comparable matching history while chat and serve remain disabled. |
| slm-cpu | SLM-CPU-019 | #5312 | `codex/slm-cpu-019-kaby-performance-dashboard` | Formalize the Kaby Lake Qwen3 Q8_0 CPU performance dashboard from existing 1/2/4/8-thread envelope and operator-profile receipts. The dashboard must record cold load, warm-session load-once, prefill, first-token latency, steady decode, per-prompt timing boundaries, allocation or buffer-reuse counters where available, memory footprint, storage/free-space context, and thermal/power fields as measured or explicitly unavailable. It may choose an operator default thread count only from recorded evidence and must preserve generated-ID, tokenizer, backend, model SHA, and fallback=false claim boundaries without adding runtime optimization or sustained-throughput claims. |
