# BITNET-SPEC-OPENVINO-QUALITY-CORPUS: OpenVINO Dense SLM Quality Corpus Contract

Status: draft
Owner: intel/openvino
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-DENSE-SLM](BITNET-SPEC-OPENVINO-DENSE-SLM.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion; defines quality receipt gates
Policy impact: no policy exception

## Purpose

Define how Lunar Lake OpenVINO dense SLM routes use the bounded answer corpus v2
as quality evidence. This spec keeps route/profile quality evidence separate
from timing, power, server readiness, native OpenCL, NPU kernel, and BitNet
QK256/I2_S claims.

This spec does not run inference, promote OpenVINO GPU or NPU, claim broad chat
quality, claim speedup, or prove BitNet behavior.

## Corpus Scope

The canonical corpus input is:

```text
ci/quality/lunar-lake-answer-corpus-v2.yaml
```

Every OpenVINO corpus receipt must bind each case to:

- `case_id`;
- `profile`;
- `category`;
- prompt text and rendered prompt evidence;
- expected answer gate;
- generation config;
- model/export identity;
- route identity;
- fallback status.

The corpus is a bounded regression lens. Passing it can qualify a route for a
named profile review, but it is not a broad chat-quality claim.

## Required Profiles

Quality receipts must preserve these profile labels exactly unless a future spec
adds or retires one:

| Profile | Intent |
| --- | --- |
| `regression_tiny` | cheap smoke for strict regression |
| `ask_short` | short prompt, short answer |
| `ask_normal` | ordinary local ask |
| `prefill_heavy` | long prompt, bounded answer |
| `decode_heavy` | short prompt, longer answer |
| `structured` | constrained or format-sensitive output |
| `low_power` | quality input for later low-power route review |
| `warm_resident` | quality input for resident route review |

If a receipt lacks a required profile, it must record that profile as
`not_run`, not silently omit it from route comparison.

## Required Categories

The bounded corpus should cover:

- arithmetic;
- exact copy;
- yes/no;
- short factual;
- short reasoning;
- instruction following;
- stop-token behavior;
- multi-turn or role-template behavior;
- structured output;
- long-prompt summarization.

Routes may be promoted only for profiles whose cases pass or whose failures are
explicitly marked diagnostic-only by a later accepted spec.

## Receipt Shape

Minimum corpus receipt shape:

```json
{
  "artifact_kind": "openvino_dense_slm_corpus_v2",
  "route_id": "openvino_dense_slm_gpu_arc140v",
  "proof_family": "openvino_dense_slm_gpu_arc140v",
  "requested_backend": "openvino-gpu",
  "selected_backend": "openvino-gpu",
  "runtime_api": "openvino_genai",
  "runtime_device": "GPU.0",
  "fallback_used": false,
  "model_id": "qwen2_5_0_5b_instruct_openvino_int4_sym",
  "prompt_template": "qwen2.5",
  "cases": [
    {
      "case_id": "math_2_plus_2",
      "profile": "regression_tiny",
      "category": "arithmetic",
      "rendered_prompt": "<string-or-redacted-with-hash>",
      "prompt_sha256": "<sha256>",
      "prompt_token_count": 32,
      "generated_text": "4",
      "generated_text_preview": "4",
      "generated_token_ids": {
        "source": "direct|retokenized|unavailable",
        "ids": []
      },
      "stop_reason": "eos|stop_token|max_new_tokens|unknown",
      "answer_gate": {
        "kind": "normalized_exact|contains_all|forbidden_none|json_schema|custom",
        "passed": true,
        "failure_class": null
      }
    }
  ],
  "summary": {
    "passed": 12,
    "failed": 0,
    "not_run": 0
  }
}
```

Receipts may redact long prompts, but must retain a stable prompt hash and token
count. Any redaction must be identical across routes for comparison.

## Prompt and Token Evidence

Each case must record:

- raw prompt or redacted prompt plus hash;
- rendered prompt or rendered prompt hash;
- chat template source;
- add-generation-prompt policy;
- stop/EOS policy;
- prompt token count;
- generated token count when available.

OpenVINO GenAI may not expose direct generated token IDs. Receipts must mark the
source explicitly:

| Source | Meaning | Promotion use |
| --- | --- | --- |
| `direct` | token IDs emitted by the runtime/tokenizer path | strongest evidence |
| `retokenized` | decoded output retokenized with the same tokenizer | output accounting only |
| `unavailable` | no token IDs available | quality can pass; token parity cannot be claimed |

Retokenized IDs must not be described as direct generated IDs.

## Generation Config

Every corpus run must record generation settings:

```json
{
  "max_new_tokens": 32,
  "temperature": 0.0,
  "top_p": null,
  "top_k": null,
  "sampling": "greedy",
  "seed": null,
  "eos_token_id": "<id-or-unavailable>",
  "stop_sequences": [],
  "beam_search": false,
  "parallel_sampling": false
}
```

Changing generation config resets quality comparability for route-promotion
review. NPU receipts must follow the extra constraints in
`BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE`.

## Exact-Answer Generation Policy

Exact-answer cases are promotion gates, not broad chat-quality tests. A route
must pass them under the accepted fixture policy before the matching profile can
be promoted.

Normalized exact-answer failures are classified into two policy paths:

| Failure type | Required handling |
| --- | --- |
| fixture-budget overgeneration | The expected answer appears but extra text violates the exact gate; adjust fixture budget, stop policy, or generation policy only through an accepted spec/update, then rerun the affected routes. |
| true exact-answer instruction miss | The expected answer does not appear under tested budgets; do not treat smaller budgets as a fix, and keep the route blocked until prompt, template, generation policy, or model behavior is corrected and rerun. |

Generation-budget sensitivity evidence may explain why a failure happened, but
it does not by itself pass the case. A smaller-budget pass means the fixture or
stop/max-token policy may be too permissive for that exact-answer case; it is
not route-promotion evidence until the canonical fixture is changed and rerun.

Max-token truncation is not the same thing as EOS or stop-token correctness.
Receipts must keep `stop_reason`, `max_new_tokens`, stop sequences, and
generated-token source visible so later reviewers can distinguish:

- an answer that naturally stops after the exact token;
- an answer clipped by a stricter token budget;
- an answer that ignores the instruction and produces the wrong token;
- an answer whose decoded text passes but whose direct token IDs are
  unavailable.

For the current Lunar Lake corpus-v2 evidence:

- `yes_no_clear_sky` is fixture-budget overgeneration-sensitive because
  `max_new_tokens=1` passes on CPU, GPU.0, and NPU while the fixture budget
  fails.
- `stop_token_one_word_done` is a true exact-answer instruction miss for the
  tested budgets because no tested budget passes on CPU, GPU.0, or NPU.

No OpenVINO GPU/NPU route may be promoted for a profile containing either case
until the canonical fixture or generation policy is updated and the route passes
the rerun, or a later accepted spec marks the case diagnostic-only for that
profile.

## Failure Taxonomy

Failures must be classified with one primary class:

| Failure class | Meaning |
| --- | --- |
| `wrong_answer` | content contradicts the expected answer |
| `overgenerated_exact_answer` | expected answer appears but extra text violates exact gate |
| `instruction_miss` | output ignores requested format or role |
| `stop_eos_miss` | stop/EOS behavior is wrong |
| `template_mismatch` | prompt/rendering differs from route contract |
| `tokenizer_mismatch` | token accounting or tokenizer identity differs |
| `structured_format_miss` | JSON or structured output fails |
| `case_sensitive_keyword_miss` | keyword present only with wrong case |
| `required_term_miss` | one or more required terms absent |
| `runtime_error` | route failed before usable generation |
| `not_run` | profile/case intentionally not executed |
| `diagnostic_only` | case is excluded from promotion by accepted spec |

Secondary notes may add generation-budget sensitivity, route blockers, or known
OpenVINO visibility limits.

## Promotion Inputs

Corpus evidence can support promotion review only when:

1. The exact route/profile cases pass.
2. `fallback_used=false`.
3. Model/export/tokenizer/template identity matches the route contract.
4. Prompt evidence and generation config are recorded.
5. Failure taxonomy has no unclassified failures.
6. The receipt states what it does not prove.

Quality evidence alone cannot promote a route. Timing, power, stability, and
route-policy evidence are separate gates.

## Rejection Examples

| Evidence | Required decision |
| --- | --- |
| GPU passes `regression_tiny` only | Do not promote `ask_normal` |
| NPU answer passes but generated IDs are retokenized | No direct token-ID parity claim |
| Candidate route has `fallback_used=true` on one case | Block promotion |
| Exact-answer failure is known budget-sensitive | Keep candidate blocked until fixture or generation policy is resolved |
| Dense SLM corpus passes | Do not treat as BitNet QK256/I2_S proof |

## Acceptance

This spec is complete when it defines:

1. Required profile and category coverage for corpus-v2 receipts.
2. Prompt, template, stop/EOS, and generation config evidence.
3. Direct versus retokenized generated-token accounting.
4. A route/profile failure taxonomy.
5. Promotion inputs and rejection examples that preserve strict claim
   boundaries.
