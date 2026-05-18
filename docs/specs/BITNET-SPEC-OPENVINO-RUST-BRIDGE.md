# BITNET-SPEC-OPENVINO-RUST-BRIDGE: OpenVINO Rust Bridge Contract

Status: draft
Owner: intel/openvino
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-DENSE-SLM](BITNET-SPEC-OPENVINO-DENSE-SLM.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion; defines Rust bridge stages
Policy impact: no policy exception

## Purpose

Define how Lunar Lake OpenVINO proof harnesses may move from Python-driven
receipts toward Rust CLI and runtime surfaces without weakening route identity,
fallback discipline, or claim boundaries.

This spec does not implement a bridge, delete Python harnesses, promote routes,
claim speedup, claim broad dense SLM quality, prove server readiness, or prove
BitNet QK256/I2_S behavior.

## Bridge Stages

The OpenVINO bridge must advance in ordered stages:

| Stage | Name | Required proof | Product claim |
| --- | --- | --- | --- |
| 0 | Python proof harness | committed Python receipts and validators | proof harness only |
| 1 | Rust CLI wrapper | Rust command invokes Python with strict args and receipt schema | wrapper only |
| 2 | Rust receipt validator | Rust validator accepts and rejects Python receipt schema | validation only |
| 3 | Rust subprocess bridge | Rust owns route command, process control, and receipt normalization | controlled bridge |
| 4 | Rust OpenVINO binding | Rust calls OpenVINO Runtime or GenAI through binding/subprocess abstraction | exact-route candidate |
| 5 | Rust-native product surface | `ask`, `chat`, `bench`, or `server` surfaces emit equivalent receipts | exact-profile surface |

No stage may skip receipt equivalence with the previous stage. Python harnesses
must remain available until the Rust stage emits equivalent receipts and passes
the same validators.

## Stage 0: Python Proof Harness

Stage 0 is the current proof baseline for OpenVINO GenAI CPU/GPU/NPU evidence.
It must record:

- exact command line;
- Python version;
- OpenVINO and OpenVINO GenAI versions;
- model/export identity;
- tokenizer/template identity;
- selected device;
- fallback status;
- quality and timing receipt fields required by the OpenVINO specs.

Stage 0 receipts are valid proof artifacts, but they are not Rust product UX.

## Stage 1: Rust CLI Wrapper

Stage 1 may add commands such as:

```text
bitnet lunar-lake openvino-run --device gpu --profile ask_short -- ...
```

The wrapper must:

- pass strict, explicit arguments to the Python harness;
- capture stdout/stderr and exit code;
- require a receipt path;
- fail closed if the receipt is missing or invalid JSON;
- not rewrite route identity fields;
- record wrapper metadata separately from runtime metadata.

The wrapper must not claim Rust-native OpenVINO execution.

## Stage 2: Rust Receipt Validator

Stage 2 adds a Rust-side validator for OpenVINO receipts. It must check:

- requested and selected backend consistency;
- `fallback_used=false` for strict routes;
- route ID and proof family consistency;
- dense SLM receipts do not claim BitNet QK256/I2_S behavior;
- OpenVINO GPU receipts do not claim native OpenCL proof;
- NPU receipts include cold/cache/warm fields when NPU promotion is attempted;
- direct versus retokenized generated-token IDs are marked correctly;
- model/export identity matches the manifest.

Validation failures must be machine-readable.

## Stage 3: Rust Subprocess Bridge

Stage 3 lets Rust own the route command while still using a subprocess harness.
It must record:

```json
{
  "bridge_stage": 3,
  "bridge_kind": "rust_subprocess_python",
  "subprocess_command": "<redacted-or-recorded>",
  "subprocess_exit_code": 0,
  "receipt_schema_version": "<version>",
  "receipt_validator": "rust",
  "fallback_used": false
}
```

Stage 3 can improve operator UX, but it remains a bridge. It must not be
described as a Rust-native OpenVINO binding.

## Stage 4: Rust OpenVINO Binding

Stage 4 may use an OpenVINO Runtime binding, OpenVINO GenAI binding, or a stable
library subprocess abstraction that exposes equivalent route identity. It must
prove parity against Stage 0 or Stage 3 for the exact model/profile:

- same route ID;
- same selected device;
- same prompt/template policy;
- same generation config;
- equivalent quality result;
- equivalent timing fields or explicit phase gaps;
- no hidden fallback.

If Stage 4 cannot expose generated token IDs or phase metrics that Stage 0
could not expose either, the gap may remain, but it must be recorded.

## Stage 5: Rust-Native Product Surface

Stage 5 surfaces may include:

```text
bitnet lunar-lake ask --device openvino-gpu ...
bitnet lunar-lake chat --device openvino-npu ...
bitnet lunar-lake bench --route openvino_dense_slm_gpu_arc140v ...
bitnet serve --route openvino_server_exact_profile ...
```

Each surface must emit the same route, model, fallback, quality, timing, and
claim-boundary fields as the proof harness. Product UX does not waive proof
requirements.

## Receipt Equivalence

A later bridge stage can replace an earlier stage only when an equivalence
receipt records:

```json
{
  "artifact_kind": "openvino_bridge_equivalence",
  "old_stage": 0,
  "new_stage": 3,
  "route_id": "openvino_dense_slm_gpu_arc140v",
  "profile": "ask_short",
  "model_id": "qwen2_5_0_5b_instruct_openvino_int4_sym",
  "quality_equivalent": true,
  "route_identity_equivalent": true,
  "fallback_equivalent": true,
  "timing_fields_equivalent_or_explained": true,
  "known_gaps": []
}
```

Equivalence is exact-route and exact-profile only.

## Failure Handling

All bridge stages must fail closed when:

- the selected device differs from the requested strict device;
- fallback is observed;
- receipt schema is missing or invalid;
- model/export identity differs from the manifest;
- Python or OpenVINO dependency versions are missing for proof receipts;
- route proof family conflicts with the receipt fields;
- a dense SLM route claims BitNet QK256/I2_S proof.

## Non-Goals

This spec does not:

- choose a Rust OpenVINO binding;
- require deletion of Python proof harnesses;
- promote OpenVINO GPU or NPU;
- define server readiness;
- define broad chat UX;
- claim native OpenCL or native NPU kernel execution;
- prove BitNet full inference on OpenVINO.

## Acceptance

This spec is complete when it defines:

1. Ordered bridge stages from Python proof harness to Rust-native surfaces.
2. Required proof for wrapper, validator, subprocess, binding, and product
   surface stages.
3. Receipt equivalence requirements before replacing prior harnesses.
4. Fail-closed behavior for fallback, identity, schema, and claim leakage.
5. Explicit non-goals preserving docs-only claim boundaries.
