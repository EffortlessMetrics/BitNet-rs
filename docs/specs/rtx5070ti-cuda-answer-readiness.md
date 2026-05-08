# RTX 5070 Ti CUDA Answer Readiness

## Purpose

The RTX 5070 Ti CUDA lane has receipt-backed execution proof and, after
`CUDA-ANSWER-010`, strict deterministic answer-corpus proof. The next product
milestone is a normal reusable user path: one command or warm session, one or
more questions, coherent answers, strict fallback rejection, and answer
receipts.

The strict CUDA receipts prove that the official BitNet GGUF can route BitNet
linear work through QK256 CUDA kernels with upload-once weights and zero BitNet
linear CPU fallback. The `CUDA-ANSWER-010` receipts additionally prove that the
strict CUDA `ask` path and committed answer corpus can produce coherent
deterministic answers through that backend. Productization still needs to make
the reusable command/session surface boring, documented, and benchmarked without
weakening claim boundaries. The answer-readiness lane must exercise the full
user-visible generation stack:

```text
question
-> prompt template
-> tokenizer
-> prompt prefill
-> incremental decode
-> stop criteria
-> answer text
-> receipt
```

## Target Command

The intended user command is:

```bash
cargo run --locked -p bitnet-cli --no-default-features \
  --features cpu,cuda,full-cli \
  -- ask \
  --device nvidia-rtx-5070-ti-cuda \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --question "What is BitNet, in one sentence?" \
  --max-new-tokens 96 \
  --strict-cuda \
  --receipt-out target/bitnet/receipts/ask-rtx5070ti.json
```

The exact answer text can vary once sampling is enabled. The first readiness
gate must use deterministic greedy decode so failures are reproducible.

## Current Evidence

The completed NVIDIA proof lane plus `CUDA-ANSWER-010` support these claims:

- The selected backend is `nvidia-rtx-5070-ti-cuda`.
- The runtime API is `cuda`.
- The official BitNet GGUF and explicit tokenizer are recorded.
- BitNet linear work routes through `qk256_gemv_cuda`.
- QK256 CUDA kernel invocation counts are greater than zero.
- Weights are uploaded once.
- Per-token weight upload is false.
- BitNet linear CPU fallback is zero.
- The strict CUDA `ask` path answers the constrained `math_2_plus_2` prompt.
- The RTX 5070 Ti CUDA answer corpus passes all five committed deterministic
  answer-readiness cases.
- `speedup_claim` remains false unless a later same-model benchmark upgrades it.

That is strict corpus-scoped CUDA answer proof. It is not broad chat quality,
production server readiness, or a speedup claim.

Answer readiness also depends on the shared model-artifact gate in
`docs/model-artifacts/ANSWER_ARTIFACT_GATE.md`. `MODEL-ARTIFACT-007` marks the
official Microsoft I2_S artifact `answer_ready` for backend gates only when
paired with the documented external tokenizer/pre-tokenizer authority and
BitNet.cpp answer prompt envelope.

## Non-Goals

- Do not make dense regular-LLM CUDA work satisfy this contract.
- Do not use WGPU, Vulkan, D3D12, or generic `cuda` as RTX 5070 Ti CUDA proof.
- Do not make speedup claims in answer receipts by default.
- Do not weaken strict CUDA fallback behavior for user convenience.
- Do not mutate the existing proof commands just to hide product gaps.

## CPU-First Rule

Before CPU or CUDA answer quality is debugged as a backend issue, the model
artifact must be answer-capable under the shared gate. If a reference runner
also produces non-coherent output for the same artifact and prompt suite, CPU
and CUDA runs against that artifact are diagnostic-only and should classify the
failure as `model_artifact_blocked`.

Before CUDA answer quality is debugged, the CPU reference path must answer the
same prompt corpus with the official GGUF, strict loader, and explicit tokenizer.
If CPU output is also garbage, the defect is not a CUDA-kernel defect.

Initial CPU answer corpus:

| ID | Prompt | Gate |
|---|---|---|
| `math_2_plus_2` | `What is 2+2? Answer with only the number.` | Exact constrained answer. |
| `colors_four` | `Name four common colors.` | Readable list. |
| `bitnet_one_sentence` | `Explain BitNet in one sentence.` | Readable one-sentence explanation. |
| `python_add` | `Write a short Python function that adds two numbers.` | Contains a plausible function. |
| `capital_france` | `What is the capital of France?` | Mentions Paris. |
| `five_word_summary` | `Summarize this sentence in five words: BitNet uses very low-bit weights to reduce memory use.` | Non-empty, concise summary. |
| `shapes_three` | `List three common shapes.` | Readable list. |
| `yes_no_water` | `Answer yes or no: is water wet?` | Starts with yes or no. |

CPU baseline acceptance:

- Answer is non-empty after special-token trimming.
- Answer is valid printable UTF-8.
- No repeated raw special tokens are present.
- No obvious tokenizer garbage is present.
- Constrained prompts meet their exact gates.
- Open prompts are readable enough to inspect manually.

## Prompt Template Authority

The answer path for the official BitNet GGUF must use the BitNet.cpp answer
template recorded by the CPU/reference evidence. The model-artifact gate remains
the authority for promoting any template to answer-ready status: if GGUF
metadata, reference-runner evidence, or tokenizer metadata changes the accepted
template family, the answer artifact record must capture that before the CUDA
answer lane can claim coherent output.

The prompt envelope must control:

- BOS handling.
- `User:` / `Assistant:` envelope text.
- `<|eot_id|>` and `<|end_of_text|>` stop behavior.
- Special-token skipping during answer decode.

Answer receipts must include:

```json
{
  "prompt_template": {
    "family": "bitnetcpp-answer",
    "bos_inserted": true,
    "assistant_prefix_inserted": true,
    "rendered_sha256": "...",
    "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
    "special_tokens_skipped_on_decode": true
  }
}
```

## Prompt Prefill Requirement

Answer readiness requires real prompt prefill. The ask path must tokenize the
full prompt, prefill all prompt tokens, populate the KV cache, then decode new
tokens incrementally.

Receipt requirement:

```json
{
  "prompt_prefill": {
    "exercised": true,
    "tokens": 0,
    "kv_cache_behavior": "prefill_then_incremental_decode",
    "kv_cache_device": "cpu|cuda"
  }
}
```

CPU KV cache is acceptable for the first answer-readiness milestone if it is
recorded honestly. Moving KV cache to CUDA is a later speed/productization
improvement, not a prerequisite for initial coherent output.

## Strict CUDA Behavior

For `--device nvidia-rtx-5070-ti-cuda --strict-cuda`, these conditions must be
hard failures:

- CUDA unavailable.
- Selected backend differs from `nvidia-rtx-5070-ti-cuda`.
- CPU fallback is attempted.
- BitNet linear layer fallback count is non-zero.
- Unsupported op is encountered.
- Minimal, mock, or fallback loader is used.
- Tokenizer is missing or ambiguous.
- QK256 CUDA kernel stats are missing or have zero invocations.
- Weights are uploaded per token.
- Answer quality gate fails after the answer receipt is written.

Generic `cuda` remains distinct from the RTX 5070 Ti proof lane and must not be
reported as `nvidia-rtx-5070-ti-cuda`.

## Answer Receipt

The answer receipt template lives at:

```text
ci/hardware/_templates/bitnet-cuda-answer-receipt.json
```

Required high-level invariants:

- `artifact_kind == "bitnet_cuda_answer"`.
- `requested_backend == selected_backend == "nvidia-rtx-5070-ti-cuda"`.
- `runtime_api == "cuda"`.
- `fallback_used == false`.
- `speedup_claim == false` unless a same-model benchmark qualifies it.
- `prompt_prefill.exercised == true`.
- `bitnet.weights_uploaded_once == true`.
- `bitnet.per_token_weight_upload == false`.
- `kernel_stats` contains `qk256_gemv_cuda` with `invocations > 0`.
- `execution_coverage.bitnet_linear_layers_cpu_fallback == 0`.
- `quality.garbage_filter_passed == true`.

## Garbage Filter

The garbage filter is a readiness tripwire, not a model-quality evaluator. It
must reject:

- Empty answers after special-token trimming.
- Invalid UTF-8 or replacement characters.
- Mostly punctuation or control-character output.
- Raw Llama special-token markers in visible answer text.
- Repetition above the configured token or substring threshold.
- Non-finite logits or invalid token IDs when those are recorded.

The filter must be deterministic and receipt-visible. Failure should include the
specific failed rule so the next debugging step is obvious. Strict CUDA ask
runs must still write the answer receipt before exiting non-zero on a failed
quality gate, so the artifact records the backend proof and the quality failure
in the same place.

## CPU/CUDA Answer Parity

After CPU answer quality is established, the CUDA ask path must compare against
the CPU reference corpus:

- CPU greedy token IDs.
- CUDA greedy token IDs.
- First divergence index.
- Top-k logits at divergence when available.
- Decoded CPU answer.
- Decoded CUDA answer.
- Prompt template and tokenizer authority for both lanes.

Initial CUDA readiness requires exact token match for constrained prompts.
Open prompts may diverge after an agreed prefix only if both outputs remain
readable and the divergence artifact records the first differing token and
logit context.

## Speed Policy

Answer receipts default to:

```json
{
  "speedup_claim": false
}
```

Speed claims can only be upgraded by a later same-model, same-tokenizer,
same-prompt-profile, fallback-free CPU/CUDA benchmark receipt that records load
time, weight upload time, prompt prefill, first-token latency, steady-state
decode, kernel time, transfer time, VRAM high-water mark, power, temperature,
driver, CUDA runtime, toolkit, and NVRTC versions.

## Work Items

### CUDA-ANSWER-001 - CPU Answer Corpus Baseline

Add the fixed prompt corpus and prove the strict CPU path produces sane,
deterministic greedy answers before debugging CUDA answer quality.

### CUDA-ANSWER-002 - Prompt Template Authority

Add or harden the strict BitNet.cpp answer template, including tokenized
envelope tests and receipt fields for BOS, assistant prefix, stop tokens, and
special-token decode policy.

### CUDA-ANSWER-003 - CPU Ask Command

Add `bitnet ask --device cpu` with real prompt prefill, normal decode loop,
answer text, and answer receipt generation.

### CUDA-ANSWER-004 - Strict RTX 5070 Ti CUDA Ask Command

Route `bitnet ask --device nvidia-rtx-5070-ti-cuda --strict-cuda` through the
proven CUDA backend with strict fallback rejection and answer receipts.

### CUDA-ANSWER-005 - CPU/CUDA Answer Corpus Parity

Record CPU and CUDA greedy token sequences, decoded answers, divergence
artifacts, and quality-gate results for the fixed answer corpus.

### CUDA-ANSWER-010 - Strict RTX 5070 Ti CUDA Corpus Proof

Record the post-#4024 strict CUDA answer evidence for the official Microsoft
I2_S artifact: constrained `ask` returns `4`, the five-case answer corpus passes,
the selected backend is `nvidia-rtx-5070-ti-cuda`, runtime API is `cuda`,
fallback is false, the selected kernel is `qk256_gemv_cuda`, prompt prefill is
exercised, upload-once weights are preserved, and `speedup_claim=false`.

### CUDA-ANSWER-006 - Interactive CUDA Chat Session

Add a session path that loads the model once, uploads weights once, reuses the
CUDA BitNet context, and emits per-turn or session-summary receipts.

### CUDA-ANSWER-007 - Answer Throughput Qualification

Benchmark the user-facing ask/chat path only after answer quality passes. Keep
`speedup_claim=false` until the benchmark policy explicitly qualifies it.

## Diagnosis Commands

Current `main` should be sanity-checked before implementation PRs that touch
generation behavior. If command names differ, that mismatch is part of the
product-surface gap.

CPU sanity:

```bash
cargo run --locked -p bitnet-cli --no-default-features \
  --features cpu,full-cli \
  -- generate \
  --device cpu \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --prompt "What is 2+2? Answer with only the number." \
  --max-new-tokens 16 \
  --greedy
```

CUDA sanity:

```bash
cargo run --locked -p bitnet-cli --no-default-features \
  --features cpu,cuda,full-cli \
  -- generate \
  --device nvidia-rtx-5070-ti-cuda \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --prompt "What is 2+2? Answer with only the number." \
  --max-new-tokens 16 \
  --greedy \
  --json-out target/bitnet/receipts/cuda-answer-smoke.json
```

## Debug Order

If CPU is good and CUDA diverges early, debug in this order:

1. QK256 unpack and scale handling.
2. Projection weight name mapping.
3. RMSNorm semantics.
4. RoPE base and position handling.
5. KV-cache append and read behavior.
6. Logits extraction and output head.
7. Greedy sampler tie-breaking.

If CPU is also bad, debug in this order:

1. Prompt envelope.
2. Tokenizer authority and decode policy.
3. BOS, EOS, and stop-token handling.
4. Loader or weight mapping.
5. RoPE and KV-cache behavior.
6. Sampling and logits post-processing.

## Related Docs

- `docs/specs/nvidia-rtx-5070-ti-roadmap.md`
- `docs/bitnet/BITNET_RUNTIME_PHASES.md`
- `docs/bitnet/BITNET_RECEIPT_FIELDS.md`
- `docs/explanation/architecture/adr-014-prompt-template-auto-detection.md`
- `ci/hardware/_templates/strict-bitnet-cuda-proof.json`
