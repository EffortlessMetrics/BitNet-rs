<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-008U | #4617 | `codex/slm-cpu-008u-output-weight-layout` | Fix the remaining Qwen3 output-head layout candidate by treating GGUF output.weight with hidden/vocab dimensions as token-major storage that must be reshaped to vocab/hidden without transposing values, while preserving true transposed lm_head.weight handling. This slice must not claim Qwen3 first-token parity until a real i5-8250U artifact refresh proves bitnet-rs matches the reference token 19 / '4' with strict loader/tokenizer provenance and fallback=false. |
