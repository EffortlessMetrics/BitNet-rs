# MODEL-ARTIFACT-002 Reference-Good BitNet Search

**Date:** 2026-05-07
**Campaign:** `model-artifacts`
**Status:** blocked; no recorded BitNet GGUF/tokenizer artifact is `answer_ready`

## Summary

`MODEL-ARTIFACT-002` promoted the prior hardware-lane artifact search into the
shared model-artifact gate. No candidate is currently allowed to unblock CPU,
CUDA, Apple M4, NPU, SLM, server, or other coherent local-answer claims.

The current blocker remains model artifact authority. Some GGUFs are
structurally valid and some can produce plausible continuations for part of the
suite, but none of the recorded candidates passes the deterministic answer
prompt suite under the reference runner with tokenizer/pre-tokenizer and prompt
template authority recorded.

This report does not change runtime behavior and does not claim CPU or CUDA
answer readiness.

## Gate

The shared answer artifact gate is:

```text
docs/model-artifacts/ANSWER_ARTIFACT_GATE.md
ci/model-artifacts/artifact-manifest.toml
ci/model-artifacts/candidate-artifacts.toml
ci/model-artifacts/rejected-artifacts.toml
ci/quality/bitnet-answer-corpus.yaml
```

An accepted artifact must record:

```text
repo or source path
file name
sha256
byte size
format and architecture
quantization family
tokenizer and pre-tokenizer authority
prompt-template authority
reference runner command and version or commit
deterministic prompt-suite result
```

The required state is:

```text
answer_ready
```

No candidate reached that state.

## Local Environment Check

The local isolated worktree did not have a runnable reference-good setup:

| Check | Result |
|---|---|
| Local GGUF under `models/` | none |
| Hugging Face cache BitNet GGUF | none |
| Local `llama-cli.exe` | not found |
| `xtask download-model --list` | release xtask works; debug xtask stack-overflows on Windows |
| `xtask fetch-cpp --backend cpu` | timed out after 900 seconds and was stopped |

The timed-out bootstrap means this PR records shared artifact state from prior
reference-runner evidence and current environment facts rather than pretending a
new reference-good artifact was produced locally.

## Upstream Metadata Note

The upstream source repo `microsoft/bitnet-b1.58-2B-4T` publishes tokenizer
metadata with:

```text
tokenizer_class = PreTrainedTokenizerFast
bos_token = <|begin_of_text|>
eos_token = <|eot_id|>
chat_template = Role: content<|eot_id|> ... Assistant:
```

That metadata is useful for future prompt-template diagnosis, but it is not by
itself an answer-ready GGUF artifact. The current GGUF artifact remains rejected
because reference-runner evidence records missing `tokenizer.ggml.pre` authority
and prompt-suite failure.

## Candidate Decisions

| Candidate | Source | Reference result | Decision |
|---|---|---|---|
| `microsoft_bitnet_b158_2b_4t_gguf_i2s_current` | Hugging Face GGUF | loads but lacks pre-tokenizer authority and fails prompt suite | rejected |
| `jpacifico_aramis_2b_i2s` | Hugging Face GGUF | published and metadata-repaired variants fail output quality | rejected |
| `richarderkhov_bitnet_b158_large_q8_0` | Hugging Face GGUF | plausible France/Rust, math repeats prompt | rejected |
| `imi2_bitnet_b158_large_instruct_tq2_0` | Hugging Face GGUF | plausible France/Rust, math fails | rejected |
| `bosco_bessie_bitnet_instruct_100k_published` | Hugging Face GGUF | reference loader aborts | rejected |
| `greensky_bitnet_b158_3b_q2_q1` | Hugging Face GGUF | reference loader rejects tensor bounds | rejected |
| `larenspear_bitnet_b158_large_q4_k_m` | Hugging Face GGUF | prompt variants fail the math gate | rejected |
| `bifrost_sol_2b_i2s` | Hugging Face GGUF | missing pre-tokenizer warning and prompt suite failure | rejected |
| `bosco_bitnet_instruct_100k_regenerated_q8_0` | local conversion | loads after conversion fixes but fails prompt suite | rejected |

Machine-readable candidate state is in:

```text
ci/model-artifacts/candidate-artifacts.toml
ci/model-artifacts/rejected-artifacts.toml
```

## Decision

`MODEL-ARTIFACT-002` is blocked.

No CPU, CUDA, Apple M4, NPU, SLM, server, or other hardware lane may claim
coherent local answers from the recorded candidates. Diagnostic-only receipts
remain allowed if they keep:

```text
answer_readiness_claim = false
claim = diagnostic_only
speedup_claim = false
```

## Next Unblocker

The next PR should acquire or regenerate a new candidate and run it through the
same shared gate. A useful next attempt must produce one of:

1. an upstream-supported BitNet GGUF/tokenizer artifact that passes the shared
   prompt suite under a reference runner;
2. a source-to-GGUF regeneration recipe that produces coherent reference output,
   not just a loadable GGUF;
3. an explicit project decision to use a different supported answer model family,
   with claim wording changed so it does not masquerade as BitNet packed
   inference proof.

Until then, answer-readiness lanes should stay blocked or diagnostic-only.
