<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Small dense model CPU proof Campaign Status

- Campaign: `slm-cpu`
- State: `active`
- Objective: Make the Intel i5-8250U a strict CPU proof host for small dense transformer GGUF models by proving metadata, tokenizer, architecture adapter, CPU execution, and tiny answer evidence without claiming throughput.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| SLM-CPU-000 | pr_open | #3902 | `codex/slm-cpu-000-8250u-lane` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Define a separate 8250U dense SLM CPU proof lane with explicit target policy, dense GGUF requirements, tokenizer authority rules, architecture adapter boundaries, receipt fields, and claim boundaries. The first candidate policy prefers Qwen2.5-0.5B-Instruct GGUF Q4_K_M or Q8_0 after verifying exact artifact metadata. |
| SLM-CPU-001 | ready | TBD | `codex/slm-cpu-001-model-manifest` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a model candidate manifest, artifact policy, and 8250U runbook. The first target is Qwen2.5-0.5B-Instruct GGUF with Q4_K_M preferred and Q8_0 optional, but only after exact artifact path, SHA256, GGUF architecture string, tokenizer metadata, tensor naming, and chat template policy are verified. |
| SLM-CPU-002 | proposed | TBD | `codex/slm-cpu-002-metadata-preflight` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add strict SLM CPU metadata preflight that verifies model path, pinned SHA256, GGUF architecture, tokenizer source, context cap, quant format, adapter selection, and fallback=false without running inference. |
| SLM-CPU-003 | proposed | TBD | `codex/slm-cpu-003-tiny-run` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run one tiny strict dense CPU receipt on the verified candidate with real GGUF, strict tokenizer, dense architecture adapter, CPU backend, fallback=false, prompt IDs, generated IDs, and decoded text. Correct answer is not required for bring-up if the receipt is diagnosable. |
| SLM-CPU-004 | proposed | TBD | `codex/slm-cpu-004-answer-corpus` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a tiny SLM answer corpus with receipt rows recording model SHA, architecture, tokenizer source, prompt IDs, generated IDs, decoded text, selected backend, fallback status, quality pass/fail, and failed rules. |
| SLM-CPU-005 | proposed | TBD | `codex/slm-cpu-005-reference-divergence` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a machine-checkable reference divergence artifact schema and validator comparing bitnet-rs against a known-good external run by model SHA, prompt/template/BOS policy, prompt IDs, generated IDs, decoded text, top-k when available, and first divergence. |

## Hard Constraints

- Do not edit BitNet QK256/I2_S kernels.
- Do not change BitNet proof receipt semantics.
- Do not claim sustained 8250U throughput.
- Do not claim server, GPU, OpenVINO, UHD 620, or NPU execution.
- Do not use model name recognition as proof of GGUF compatibility.
