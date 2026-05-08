# MODEL-ARTIFACT-005 Authority Dimensions

**Date:** 2026-05-08
**Campaign:** `model-artifacts`
**Status:** manifest refinement only; no artifact is promoted to `answer_ready`; no runtime changes

## Purpose

This report documents the authority-dimension split introduced in MODEL-ARTIFACT-005.
Prior work items (001–004) used a coarse binary:

```
answer_ready = true | false
```

That binary hid important distinctions:

- **Target alignment** — is this artifact the one our CUDA product path is built around?
- **Runner authority** — which runner is authoritative for this artifact?
- **Tokenizer authority** — is the tokenizer model present and identified?
- **Pre-tokenizer authority** — is the pre-tokenizer rule present or only defaulted?
- **Prompt-suite result** — did the artifact pass the tiny deterministic suite under its intended runner?
- **Unblock scope** — what, if anything, can this artifact unblock?

MODEL-ARTIFACT-005 adds these dimensions as machine-readable fields on every artifact
record in `ci/model-artifacts/*.toml` without changing any gate thresholds, runtime
behavior, or answer-readiness claims.

## Authority Dimension Definitions

| Dimension | Values | Meaning |
|---|---|---|
| `answer_readiness_scope` | `official_target` | This is the artifact the official I2_S CUDA lane is built around. |
| | `alternate_quant_control` | Useful as a control lane; not the official I2_S CUDA target. |
| | `diagnostic_only` | Diagnostic evidence only; cannot unblock any answer lane. |
| `target_alignment` | `official_i2s_cuda_target` | The official Microsoft I2_S GGUF for the RTX 5070 Ti CUDA path. |
| | `official_derived_alt_quant` | Derived from the same base model but in a different quantization. |
| | `unrelated` | Different architecture, size, or training lineage. |
| `runner_authority` | `stock_llama_cpp` | Stock llama.cpp; cannot load all BitNet-specific quant types. |
| | `ik_llama_cpp` | ik_llama.cpp; intended runner for IQ2_BN family artifacts. |
| | `microsoft_bitnet` | Official Microsoft BitNet reference runner. |
| | `unknown` | Runner not identified or not tested. |
| `tokenizer_authority_dim` | `present` | GGUF contains a recognized tokenizer model. |
| | `missing` | No tokenizer model field or unrecognized value. |
| `pretokenizer_authority_dim` | `present` | `tokenizer.ggml.pre` is present in the GGUF. |
| | `missing` | `tokenizer.ggml.pre` is absent; runner used default behavior. |
| | `defaulted` | A default pre-tokenizer was applied and the artifact still failed. |
| | `externally_supplied` | Pre-tokenizer authority comes from an external tokenizer.json. |
| `prompt_suite_result` | `passed` | Tiny deterministic suite passed under intended runner. |
| | `failed` | One or more suite rows failed. |
| | `blocked` | Runner could not load the artifact to run the suite. |
| | `not_run` | Suite not yet executed for this artifact. |
| `can_unblock_official_i2s_cuda` | `true` / `false` | Whether this artifact can unblock the RTX 5070 Ti I2_S CUDA answer path. |
| `can_unblock_alt_quant_control` | `true` / `false` | Whether this artifact can serve as an alternate-quant control lane. |

## Current Artifact Decisions

### Official Microsoft I2_S (`ggml-model-i2_s.gguf`)

```toml
answer_readiness_scope = "official_target"
target_alignment       = "official_i2s_cuda_target"
runner_authority       = "ik_llama_cpp"
tokenizer_authority_dim    = "missing"
pretokenizer_authority_dim = "missing"
prompt_suite_result    = "failed"
can_unblock_official_i2s_cuda = false
can_unblock_alt_quant_control = false
```

**Evidence:** Loads under ik_llama.cpp (commit 9a26522) with warning
`load: missing pre-tokenizer type, using: 'default'`, then emits repeated colon
output (`::::`) for every deterministic prompt-suite row.

**Decision:** Remains `rejected_prompt_suite_failed`. The artifact is structurally
valid, but pre-tokenizer authority is absent and output is non-coherent. This is
the exact artifact the RTX 5070 Ti QK256/I2_S CUDA answer path requires. Until it
passes the prompt suite under a reference runner, the CUDA answer lane remains
blocked.

**Next action:** MODEL-ARTIFACT-006 (tokenizer/pre-tokenizer authority audit)
must determine whether an external `tokenizer.json` can supply the missing
pre-tokenizer authority, or whether artifact regeneration is required.

---

### tdh111 IQ2_BN_R4 (`bitnet1582b4t-iq2_bn_r4.gguf`)

```toml
answer_readiness_scope = "alternate_quant_control"
target_alignment       = "official_derived_alt_quant"
runner_authority       = "ik_llama_cpp"
tokenizer_authority_dim    = "present"
pretokenizer_authority_dim = "missing"
prompt_suite_result    = "passed"
can_unblock_official_i2s_cuda = false
can_unblock_alt_quant_control = true
```

**Evidence:** Under ik_llama.cpp (commit 9a26522), loads and passes the five-row
deterministic tiny prompt suite:

| Prompt | Expected | Output | Pass |
|---|---|---|---|
| `math_2_plus_2` | `4` or starts with `4` | `4` | yes |
| `capital_france` | mentions Paris | `The capital of France is Paris.` | yes |
| `yes_no_water` | starts yes/no | `Yes.` | yes |
| `colors_four` | readable color list | `1. Red 2. Blue 3. Green 4. Yellow` | yes |
| `bitnet_one_sentence` | readable one sentence | `BitNet is a computer network protocol...` | yes (readability only) |

The runner still reports `load: missing pre-tokenizer type, using: 'default'`.

**Decision:** Remains `rejected_missing_tokenizer_authority` for shared answer
readiness, but is now explicitly recorded as an `alternate_quant_control` artifact
that `can_unblock_alt_quant_control = true`.

**What this means:** tdh111 IQ2_BN_R4 is useful evidence that the general ik_llama.cpp
runner pipeline can produce coherent BitNet-derived answers. It does **not** prove
that the official Microsoft I2_S artifact works, and it does **not** unblock the RTX
5070 Ti official I2_S CUDA answer path. The quantization format (IQ2_BN_R4) is
runner-specific and not the I2_S format targeted by the Rust CUDA kernel path.

**What this does NOT mean:**
- It does not prove the Rust CPU or CUDA tokenizer is correct.
- It does not prove the Rust QK256/I2_S kernel path is correct.
- It does not constitute a speed or quality claim for any hardware lane.

---

### All Other Candidates

All remaining candidates in `ci/model-artifacts/candidate-artifacts.toml` and
`ci/model-artifacts/rejected-artifacts.toml` are classified as `diagnostic_only`
with `target_alignment = "unrelated"` and both `can_unblock_*` flags set to
`false`. Their existing rejection state and rejection reasons are unchanged.

## Why This Split Matters

Before MODEL-ARTIFACT-005, both artifacts were described with `answer_ready = false`.
That coarse binary obscured two different situations:

1. **Official Microsoft I2_S** — the artifact we *need* to work, under the runner
   where it *should* work, producing non-coherent output. The blocking dimension is
   pre-tokenizer authority + prompt-suite failure.

2. **tdh111 IQ2_BN_R4** — a *different quant* of the same model, under its *intended*
   runner, producing coherent output. Its blocking dimension is only pre-tokenizer
   authority. If that is resolved, it could provide a useful alternate-quant control
   lane — but not the official I2_S CUDA lane.

The authority dimensions make these distinctions machine-readable so that:
- CI and gate tooling can enforce that `can_unblock_official_i2s_cuda = false` for
  all current artifacts.
- Future promotion of the official I2_S artifact can be done by flipping
  `can_unblock_official_i2s_cuda` to `true` and `answer_ready_artifact_available`
  to `true` — but only after all gate requirements are actually met.
- The alternate-quant control lane has a defined promotion path separate from the
  official I2_S CUDA lane.

## Claim Boundary

This report does **not** change any of the following:
- Gate thresholds for `answer_ready`.
- State of `answer_ready_artifact_available` (remains `false`).
- Runtime tokenizer behavior.
- Model loader behavior.
- CPU or CUDA kernel behavior.
- Any backend answer-readiness claim.
- Any speedup claim.

## Next Unblocker: MODEL-ARTIFACT-006

The concrete next step is a tokenizer/pre-tokenizer authority audit for both key artifacts.

For each artifact, MODEL-ARTIFACT-006 must record:
- Whether `tokenizer.ggml.pre` is absent in the GGUF or present but not parsed.
- Whether the Hugging Face repo ships a `tokenizer.json` beside the GGUF.
- Whether `tokenizer.json` defines the missing pre-tokenizer behavior.
- Whether ik_llama.cpp uses a default pre-tokenizer that differs from the Rust
  tokenizer pipeline.
- Whether a documented compatibility decision can supply `tokenizer.json` as
  external pre-tokenizer authority without changing the GGUF SHA256.

Deliverables:
- `ci/model-artifacts/tokenizer-authority.toml` — machine-readable per-artifact
  tokenizer/pre-tokenizer authority records.
- `docs/reports/MODEL_ARTIFACT_006_TOKENIZER_AUTHORITY.md` — findings and
  classification of each artifact.

**MODEL-ARTIFACT-006 does not change runtime behavior.** It only records whether
a valid authority source exists. If it does, MODEL-ARTIFACT-008 (Path A) can then
promote the official I2_S artifact with documented external tokenizer authority. If
it does not, MODEL-ARTIFACT-008B (Path B) initiates artifact regeneration.

## Decision Matrix

| Finding after MODEL-ARTIFACT-006 | Next action |
|---|---|
| Official I2_S passes with external tokenizer authority (Path A) | MODEL-ARTIFACT-008: promote official I2_S with external tokenizer authority; run 9950X3D / RTX 5070 Ti parity. |
| Official I2_S cannot pass under any valid runner (Path B) | MODEL-ARTIFACT-008B: regenerate official-target I2_S GGUF with required metadata; run parity. |
| tdh111 remains the only coherent candidate (Path C) | MODEL-ARTIFACT-008C: add alternate-quant BitNet answer control lane; do **not** claim official I2_S CUDA readiness. |
| Rust token IDs differ from reference runner | Fix tokenizer/template authority in a separate tracker item. |
| Rust token IDs match but logits differ | Fix model math / output head / quantization in a separate tracker item. |
| CPU passes, CUDA diverges | Fix CUDA/QK256 path in the RTX 5070 Ti lane. |
