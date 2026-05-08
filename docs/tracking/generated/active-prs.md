<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| model-artifacts | MODEL-ARTIFACT-004 | #3939 | `codex/model-artifacts/MODEL-ARTIFACT-004-ikllama-intended-runner` | Record intended ik_llama.cpp runner evidence for official-derived BitNet GGUF candidates, including tdh111 IQ2_BN_R4 prompt-suite output and official Microsoft I2_S prompt-suite output, without promoting an answer_ready artifact or changing runtime behavior. |
| slm-cpu | SLM-CPU-003 | #3940 | `codex/slm-cpu-003-tiny-strict-receipt` | Run one tiny strict dense CPU receipt on the verified candidate with real GGUF, strict tokenizer, dense architecture adapter, CPU backend, fallback=false, prompt IDs, generated IDs, and decoded text. Correct answer is not required for bring-up if the receipt is diagnosable. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
