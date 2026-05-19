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
| 2 | `HF1BitLLM/Llama3-8B-1.58-100B-tokens` | ARM `I2_S` / ARM `TL1` candidate | Large Llama3-derived BitNet-family candidate; HF artifact is safetensors-visible with no approved BitNet-rs GGUF, so MacBook work starts with artifact inventory, tokenizer/prompt authority, conversion/runner authority, and route-layout proof only. |
| 3 | `1bitLLM/bitnet_b1_58-large` | ARM `I2_S` or `TL1` | Smaller 0.7B control candidate; currently blocked on artifact/conversion authority because the recorded official repo revision exposes safetensors/tokenizer files but no official GGUF. Follow [BITNET-PROP-0009](../proposals/BITNET-PROP-0009-bitnet-b158-large-control-model.md), the [source map](../bitnet/bitnet-b158-large/README.md), and the B158-large specs before any answer/backend claim. |
| 4 | `1bitLLM/bitnet_b1_58-3B` | ARM `TL1` diagnostic candidate; ARM `TL2` unsupported | Separate TL-model lane. Blocked at revision `af89e318d78a70802061246bf037199d2fb97020`: the official repository has safetensors shards and tokenizer files but no GGUF, and the current M3 Air free-space state cannot safely absorb the shards without cleanup or an approved conversion plan. Use only TL1 diagnostic routes until a verified runner path and coherent reference output exist; do not inherit Microsoft 2B `I2_S`/QK256 proof. |
| 5 | `tiiuae/Falcon-E-1B-Instruct-GGUF` | ARM `I2_S` artifact inventory, tokenizer/prompt audit, reference-good, then CPU/NEON | Compact secondary BitNet-like family; registered only until exact source/file/SHA/size/tokenizer/reference output and cleanup status are recorded. |
| 6 | `tiiuae/Falcon-E-3B-Instruct-GGUF` | ARM `I2_S` after 1B source-map and storage checks | Larger secondary family; must not inherit 1B proof and remains registered only until its own receipts pass. |
| 7 | `tiiuae/Falcon3-1B-Instruct-1.58bit-GGUF` `ggml-model-i2_s.gguf` | ARM `I2_S`, TL1 listed but unpromoted | First Falcon3 direct I2_S family target; candidate only until artifact, tokenizer/prompt, reference, I2_S layout, and CPU/NEON receipts exist. |
| 8 | `tiiuae/Falcon3-7B-Instruct-1.58bit-GGUF` `ggml-model-i2_s.gguf` | ARM `I2_S`, TL1 listed but unpromoted | Second Falcon3 direct I2_S target and larger Apple pressure candidate; needs independent 7B receipts. |
| 9 | `tiiuae/Falcon3-3B-Instruct-1.58bit` | Conversion-required `I2_S`; TL1 listed but unpromoted | Safetensors/conversion candidate only until exact conversion/runner authority exists. |
| 10 | `tiiuae/Falcon3-10B-Instruct-1.58bit` | Conversion-required `I2_S`; TL1 listed but unpromoted | Later safetensors/conversion candidate after the direct-GGUF path is boring. |

## bitnet_b1_58-large control-model lane

`1bitLLM/bitnet_b1_58-large` must start as an artifact authority and conversion
lane, not as an Apple performance lane. The first accepted evidence is exact
source inventory, tokenizer authority, conversion or official-GGUF authority,
and reference-runner output. MacBook diagnostics may inspect the candidate, but
M4 CPU/NEON, Metal, and speed claims remain blocked until the shared answer gate
and the B158-large Apple contract pass.

## Falcon-E Boundary

Falcon-E candidates are tracked by `docs/bitnet/falcon-e-family/README.md` and
`plans/falcon-e-family/implementation-plan.md`. Falcon-E proof is not Falcon3,
Microsoft BitNet 2B, 1bitLLM, dense Falcon, or generic Falcon proof. Direct
`I2_S` GGUF availability only allows registered-candidate planning until exact
artifact identity, tokenizer/prompt authority, reference-good output, I2_S
layout proof, and backend-specific receipts exist.

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

## Falcon3 Family Boundary

Falcon3 is tracked separately from Falcon-E, Microsoft BitNet 2B, 1bitLLM, Llama3-8B-1.58, and dense SLM evidence. A Falcon3 candidate must record its own exact source revision, file, SHA256, size, tokenizer authority, prompt template, reference-runner command, coherent prompt-suite output, I2_S/TL layout proof where applicable, backend/fallback receipt, and cleanup status before any answer/backend claim. Falcon3 1B proof does not prove Falcon3 7B, 3B, or 10B. Falcon3 I2_S proof does not prove TL1/TL2.

## Claim Boundary

This matrix is planning evidence. It does not prove Rust Apple BitNet local answers, full Apple Metal inference, QK256 on Apple Silicon, Neural Engine execution, MPSGraph model inference, or broad Apple Silicon performance.

### 3B TL candidate boundary

`1bitLLM/bitnet_b1_58-3B` is governed by
[`BITNET-PROP-0010`](../proposals/BITNET-PROP-0010-bitnet-b158-3b-tl-model.md)
and the 3B TL specs. For Apple, only ARM TL1 is a listed candidate route; ARM
`I2_S` and ARM TL2 are unsupported unless the compatibility ledger changes. No
Apple answer, benchmark, Metal, or server claim may be made until artifact
inventory, TL1 conversion/runner authority, tokenizer/prompt authority,
reference-good output, TL1 scalar/NEON fixtures, and strict Apple receipts pass
with `fallback=false`.

## TL1 route boundary

TL1 is an ARM-first table-lookup route tracked by `docs/bitnet/tl1/README.md`,
`plans/tl1/implementation-plan.md`, and
`docs/tracking/campaigns/tl1/CAMPAIGN.md`. Apple candidate planning must treat
TL1 as distinct from `I2_S`/QK256 and distinct from TL2; no TL1 answer/backend
claim is valid before TL1 layout authority, scalar oracle, artifact authority,
and reference-good output are proven.
