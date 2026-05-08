<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-slm-answer | SLM-M4-004 | #3956 | `codex/apple-m4-slm-answer/SLM-M4-004-warm-session` | Add warm-session behavior so the validated model and tokenizer are loaded once, multiple prompts can run in one process, per-prompt receipts are emitted, and model_load, tokenize, prefill, decode, sampling, and total timing are separated. |
| model-artifacts | MODEL-ARTIFACT-004 | #3939 | `codex/model-artifacts/MODEL-ARTIFACT-004-ikllama-intended-runner` | Record intended ik_llama.cpp runner evidence for official-derived BitNet GGUF candidates, including tdh111 IQ2_BN_R4 prompt-suite output and official Microsoft I2_S prompt-suite output, without promoting an answer_ready artifact or changing runtime behavior. |
| slm-cpu | SLM-CPU-004 | #3957 | `codex/slm-cpu-004-answer-corpus` | Add a tiny SLM answer corpus with receipt rows recording model SHA, architecture, tokenizer source, prompt IDs, generated IDs, decoded text, selected backend, fallback status, quality pass/fail, and failed rules. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
