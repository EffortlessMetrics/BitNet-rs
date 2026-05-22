<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-081 | #349 | `codex/slm-cpu-081-repeated-packed-q8-timing` | Collect or define the repeated before/after i5-8250U Qwen3 Q8_0 warm-session timing evidence needed after SLM-CPU-079's aligned packed-Q8 exact-tensor matvec counter improvement. The slice must compare the established eager-F32 or accepted sidecar oracle against the opt-in exact-tensor packed_q8_sidecar path across repeated runs, preserve model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, and fallback_used=false, and classify timing as improved, regressed, inconclusive, or not claimed. It must keep packed_q8_sidecar opt-in and exact-tensor scoped, keep the default runtime unchanged, and must not claim end-to-end speedup unless repeated receipts prove it within the bounded host/model/thread/corpus context. |
