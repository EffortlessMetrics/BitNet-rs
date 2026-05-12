# Apple Silicon BitNet Candidate Matrix

The MacBook lane uses this matrix to decide which 1-bit / 1.58-bit artifacts are worth testing on Apple Silicon before sending an accepted artifact back to the M4 Mac mini for strict local-answer proof.

The machine-readable matrix is:

```text
ci/hardware/apple-silicon-macbook/bitnet-candidate-matrix.toml
```

## Rules

- Dense Qwen evidence is not BitNet evidence.
- A candidate is not Apple answer-ready until the MacBook lane records source, exact file, SHA256, size, tokenizer authority, reference-runner command, coherent prompt-suite output, and cleanup status.
- Unsupported model/kernel routes may produce diagnostic receipts only.
- Rejected candidates should be deleted unless a later item explicitly keeps them for a bounded diagnostic reason.
- Never commit model binaries.

## Candidate Order

| Priority | Candidate | First Apple route | Status |
|---:|---|---|---|
| 1 | `microsoft/bitnet-b1.58-2B-4T-gguf` `ggml-model-i2_s.gguf` | ARM `I2_S`, then `TL1` | Shared answer gate says the official I2_S artifact is answer-ready when paired with external Microsoft tokenizer authority and `tokenizer.ggml.pre=llama-bpe`; MacBook must rerun before Apple claims. |
| 2 | `1bitLLM/bitnet_b1_58-large` | ARM `I2_S` or `TL1` | Smaller 0.7B control candidate; must record GGUF file, tokenizer authority, and coherent reference output. |
| 3 | `1bitLLM/bitnet_b1_58-3B` | ARM `TL1` diagnostic only | `I2_S` is upstream-unsupported for this model; use only supported/listed TL diagnostic routes until runner evidence exists. |
| 4 | `tiiuae/Falcon-E-1B-Instruct-GGUF` | Verify `I2_S` runner path | Secondary BitNet-like family after Microsoft and 1bitLLM behavior is understood. |
| 5 | `tiiuae/Falcon-E-3B-Instruct-GGUF` | Verify `I2_S` runner path | Larger secondary family; use only if storage and smaller-candidate results justify it. |

## Required Record For Each Probe

Every candidate probe should record:

```text
source repo
revision
file
size_bytes
sha256
model family
format
quantization
kernel route
tokenizer authority
pre-tokenizer authority
prompt template
reference runner
reference command
prompt outputs
acceptance or rejection
cleanup status
```

## Reference Prompt Rubric

Use `ci/quality/bitnet-answer-corpus.yaml` as the shared prompt suite unless the
item records a narrower candidate-specific suite. A candidate is coherent only
when the reference runner:

```text
loads the named GGUF without tokenizer fallback
uses the recorded prompt template
returns non-empty generated text for every required prompt
does not emit repeated special tokens as the answer body
does not answer with tokenizer/control-token garbage
passes the shared answer gate or records the exact failing prompt IDs
```

Rejected runs should keep enough output in the report for review, but model
binaries stay local-only.

## Claim Boundary

This matrix is planning evidence. It does not prove Rust Apple BitNet local answers, full Apple Metal inference, QK256 on Apple Silicon, Neural Engine execution, MPSGraph model inference, or broad Apple Silicon performance.
