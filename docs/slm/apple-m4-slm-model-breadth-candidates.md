# Apple M4 Dense SLM Model Breadth Candidates

`M4-MODEL-001` selects exact dense instruct GGUF candidates for the next M4
model-breadth evaluation steps. This document is selection metadata only: no
model artifact is downloaded, accepted, registered, or user-supported by this
item.

The current M4 supported set remains:

```text
qwen2.5-0.5b-instruct-q8_0    default
qwen2.5-0.5b-instruct-q4_k_m  supported non-default
```

## Selection Rules

Candidates must satisfy all of the following before `M4-MODEL-002` may run
reference output sanity:

- exact upstream repository, revision, and GGUF filename are recorded;
- expected artifact size is below the M4 model-breadth storage budget;
- license notes are recorded before any artifact is fetched;
- tokenizer and prompt-template expectations are explicit;
- durable evidence is planned under `ci/quality`, `docs/slm`, or tracking docs;
- scratch outputs may use `target`, but `target` is not a committed evidence
  path;
- model binaries are never committed.

## Selected Candidate Set

### Priority 1: `qwen3-0.6b-q8_0`

- Source repository: `Qwen/Qwen3-0.6B-GGUF`.
- Source URL:
  `https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/blob/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf`.
- Revision: `23749fefcc72300e3a2ad315e1317431b06b590a`.
- File: `Qwen3-0.6B-Q8_0.gguf`.
- Expected size: `639446688` bytes.
- License notes: upstream model card reports `apache-2.0`.
- Expected GGUF architecture: `qwen3`.
- Expected quantization: `Q8_0`.
- Tokenizer expectations: Qwen tokenizer metadata must be present in GGUF; the
  base model reports `Qwen2Tokenizer` and `<|im_end|>` as EOS.
- Prompt-template expectation: Qwen3 chat template, with non-thinking answer
  mode selected explicitly for short deterministic smoke prompts.
- Storage budget: under `750 MiB` for the downloaded artifact, with no
  committed binary.
- Why selected: closest official successor to the current Qwen2.5 dense path,
  small enough for the M4 mini, and Apache-licensed.
- Rejection criteria:
  - missing GGUF tokenizer or chat-template authority;
  - reference runner cannot produce sane short answers with deterministic
    settings and non-thinking prompt mode;
  - prompt template produces only reasoning boilerplate or repeated-token junk;
  - Rust M4 support would require unplanned architecture work outside
    `M4-MODEL-003`.

### Priority 2: `smollm2-360m-instruct-q8_0`

- Source repository: `HuggingFaceTB/SmolLM2-360M-Instruct-GGUF`.
- Source URL:
  `https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/blob/593b5a2e04c8f3e4ee880263f93e0bd2901ad47f/smollm2-360m-instruct-q8_0.gguf`.
- Revision: `593b5a2e04c8f3e4ee880263f93e0bd2901ad47f`.
- File: `smollm2-360m-instruct-q8_0.gguf`.
- Expected size: `386404992` bytes.
- License notes: upstream model card reports `apache-2.0`.
- Expected GGUF architecture: Llama-family dense model metadata.
- Expected quantization: `Q8_0`.
- Tokenizer expectations: original model config reports `GPT2Tokenizer`,
  ChatML-style `<|im_start|>` / `<|im_end|>` tokens, and `LlamaForCausalLM`
  architecture; the GGUF metadata must confirm the actual tokenizer authority.
- Prompt-template expectation: SmolLM2 ChatML template with an explicit system
  message so the reference prompts and Rust prompts are identical.
- Storage budget: under `500 MiB` for the downloaded artifact, with no
  committed binary.
- Why selected: small cross-family instruct model from the Hugging Face small
  model line; useful to prove the M4 runner can support more than Qwen-class
  dense models.
- M4-MODEL-003 outcome: rejected by Rust M4 quality for this round. The artifact
  is reference-good, but the current strict Rust loader rejects it before
  generation and diagnostic compatibility probes still produce incoherent text.
  See
  [apple-m4-slm-model-breadth-rust-m4-quality.md](apple-m4-slm-model-breadth-rust-m4-quality.md).
- Rejection criteria:
  - GGUF metadata lacks tokenizer authority or chat template;
  - reference runner output is empty, non-UTF-8, repeated-token junk, or
    semantically implausible on the bounded prompt suite;
  - prompt-template defaults differ between reference and Rust paths in a way
    that cannot be represented explicitly;
  - Rust M4 support requires a broad Llama-family adapter outside
    `M4-MODEL-003`.

## Not Selected This Round

### Gemma 3 270M IT GGUF

Gemma-class models remain interesting for M4 breadth, but this round does not
select a Gemma artifact. The currently observed small GGUF path is community
published, while the campaign requires exact trusted or official artifacts
before evaluation.

### Phi-Class Small Instruct Models

Phi-class models remain on the watchlist. This round does not select a Phi
artifact because the M4 lane still needs an exact sub-1 GiB instruct GGUF with
license, tokenizer, prompt-template, and architecture expectations that are as
boring as the Qwen3 and SmolLM2 candidates above.

### Qwen2.5 0.5B Instruct Q8_0 / Q4_K_M

These are already the supported M4 dense SLM models. They remain the regression
baseline, not new breadth candidates.

## Next Item Contract

`M4-MODEL-002` should evaluate the selected candidates in priority order. It
must record exact SHA256, GGUF metadata, tokenizer authority, prompt-template
authority, reference command, prompt outputs, and accept/reject evidence before
any Rust M4 quality work begins.

## Follow-Up Candidate Cycle

`M4-MODEL-006` opens the next candidate cycle because the first breadth round
did not produce a model that could be registered: Qwen3 was rejected by the
current reference path, and SmolLM2 was reference-good but rejected by Rust M4
quality.

### Priority 1: `gemma-3-270m-it-q8_0`

- Source repository: `ggml-org/gemma-3-270m-it-GGUF`.
- Source URL:
  `https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF/blob/e7647be17ae1108f2f605ed061ca0608b171afff/gemma-3-270m-it-Q8_0.gguf`.
- Revision: `e7647be17ae1108f2f605ed061ca0608b171afff`.
- File: `gemma-3-270m-it-Q8_0.gguf`.
- Expected size: `291545600` bytes.
- LFS SHA256: `0ef57d2c838458a1952664260dcba38e5bdda37494f3af732f06e4add24068e3`.
- License notes: the upstream base model `google/gemma-3-270m-it` reports
  `license:gemma` and requires Gemma license acknowledgement on Hugging Face.
- Expected GGUF architecture: Gemma 3 text-family metadata; the GGUF must
  confirm the exact architecture before M4 promotion.
- Expected quantization: `Q8_0`.
- Tokenizer expectations: Gemma tokenizer metadata must be present in the GGUF;
  the reference step must record tokenizer model, pre-tokenizer authority, BOS,
  EOS, PAD policy, and special-token handling.
- Prompt-template expectation: Gemma instruction/chat template, recorded from
  GGUF metadata or an explicit reference-template decision before Rust M4 runs.
- Storage budget: under `500 MiB` for the downloaded artifact, with no
  committed binary.
- Why selected: official Gemma-class small instruct base model, storage-light
  GGUF artifact, and a different leading dense SLM family from Qwen. This is a
  useful M4 model-runner breadth probe only if reference and Rust M4 gates pass.
- Rejection criteria:
  - Gemma license access is unavailable to the operator;
  - GGUF metadata lacks tokenizer or chat-template authority;
  - reference runner output is empty, non-UTF-8, repeated-token junk, or
    semantically implausible on the bounded prompt suite;
  - prompt template cannot be represented explicitly in the Rust M4 path;
  - Rust M4 support requires broad Gemma architecture work outside
    `M4-MODEL-008`.

## Follow-Up Candidate Contract

`M4-MODEL-007` should evaluate `gemma-3-270m-it-q8_0` with the reference runner
only. It must record exact SHA256, GGUF metadata, tokenizer authority,
prompt-template authority, reference command, prompt outputs, license/access
status, and accept/reject evidence before any Rust M4 quality work begins.
