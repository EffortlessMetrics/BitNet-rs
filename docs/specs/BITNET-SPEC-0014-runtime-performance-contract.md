# BITNET-SPEC-0014: Runtime Performance Contract

Status: proposed
Linked proposal:
[BITNET-PROP-0003](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
Linked specs:
[BITNET-SPEC-0013](BITNET-SPEC-0013-model-onboarding-proof-ladder.md)
Applies to: benchmark receipts, `bitnet bench`, `bitnet receipts explain`,
time-to-first-token claims, throughput claims, speedup claims, residency
claims, server readiness performance summaries

## Purpose

BitNet-rs must keep performance claims auditable. A successful local inference
run does not prove speedup, low time to first token, full residency, or server
readiness unless the receipt breaks the run into the phases and comparators
needed for that exact claim.

This spec defines the minimum runtime performance fields and promotion rules
for TTFT, throughput, speedup, and residency claims.

## Source-Of-Truth Authorities

This spec relies on:

- [Native Rust inference product proposal](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
- [Model onboarding proof ladder](BITNET-SPEC-0013-model-onboarding-proof-ladder.md)
- [9950X3D + RTX 5070 Ti CUDA product contract](BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md)
- [Server readiness proof boundary](BITNET-SPEC-0010-server-readiness-proof-boundary.md)
- [CUDA Capability Matrix](../status/CUDA_CAPABILITY_MATRIX.md)
- `ci/model-artifacts/model-coverage-matrix.toml`
- `ci/hardware/**`
- benchmark review reports and receipts committed by the relevant lane

Receipts remain evidence for what happened. This spec defines which receipt
fields must exist before status docs, model coverage rows, or CLI summaries may
promote a performance claim.

## Required Fields

Runtime performance receipts that support TTFT, throughput, speed, residency,
or server readiness claims must include the following fields when the phase
exists for the command:

```text
model_load_ms
tokenizer_load_ms
prompt_render_ms
tokenize_ms
cuda_context_init_ms
weight_upload_ms
prefill_ms
first_token_ms
decode_total_ms
steady_tok_per_s
kernel_time_ms
launch_count
H2D_bytes
H2D_ms
D2H_bytes
D2H_ms
VRAM_high_water
power_temperature_context
fallback_used
```

If a field is not applicable to a backend or profile, the receipt must say that
explicitly. Missing fields must not be silently interpreted as zero.

## Identity Fields

Every performance receipt must also identify the exact scope of the result:

```text
model_artifact
model_coverage_row
tokenizer_authority
prompt_template
requested_backend
selected_backend
selected_route
runtime_api
profile_name
prompt_id_or_prompt_shape
input_tokens
output_tokens
streaming
server_endpoint
receipt_id
receipt_path
```

The profile name must be precise enough to distinguish one-token probes,
short-decode profiles, warm sessions, server requests, and long decode runs.

## Profiles

The initial governed profiles are:

```text
one_token
short_decode_8
short_decode_32
prefill_128_decode_16
prefill_512_decode_32
warm_session_3_turns
warm_session_10_turns
decode_128_from_warm_context
server_nonstream_chat_completions
```

A new profile may be added by a later spec or benchmark review, but a result
from one profile must not be summarized as another profile.

## TTFT Claims

A time-to-first-token claim requires:

- `prompt_render_ms`;
- `tokenize_ms`;
- `cuda_context_init_ms` or explicit not-applicable reason;
- `weight_upload_ms` or explicit warm-session reuse reason;
- `prefill_ms`;
- `first_token_ms`;
- `fallback_used=false` for accelerator TTFT claims;
- model, tokenizer, prompt, backend, route, and profile identity.

TTFT may be reported for a single backend without claiming speedup. A TTFT
speedup claim also requires the exact same-artifact CPU comparator.

## Throughput Claims

A throughput claim requires:

- decode profile name;
- input token count;
- output token count;
- `decode_total_ms`;
- `steady_tok_per_s`;
- `kernel_time_ms` when an accelerator route is claimed;
- `launch_count` when accelerator launch overhead is relevant;
- fallback status;
- same model/backend/profile identity.

Throughput must not be inferred from one-token proof or server-smoke readiness.

## Speedup Claims

Speedup requires an exact comparator:

- same model artifact;
- same tokenizer and prompt-template policy;
- same prompt or governed prompt shape;
- same output token target or termination rule;
- CPU reference timing;
- accelerator timing;
- fallback rejected;
- accepted or rejected decision and reason.

Speedup is exact-profile only. A speedup accepted for `short_decode_32` does
not imply speedup for one-token, warm-session, server, other model, other
artifact, other quantization, or other backend profiles.

## Residency Claims

Full residency claims require per-phase proof:

- model load location;
- weight upload status;
- whether weights are uploaded once or per request;
- KV-cache location;
- attention, norm, RoPE, MLP, LM-head, and sampling transfer behavior when
  those phases are in scope;
- H2D and D2H byte counts;
- H2D and D2H timing;
- `VRAM_high_water`;
- host memory fallback status;
- workspace reuse status for warm sessions or server paths.

Upload-once weights alone are not full residency. Reduced D2H bytes alone are
not full residency. Full residency remains false until the receipt proves every
phase required by the relevant route.

## Server Performance And Readiness

Server performance summaries must preserve server scope:

- endpoint;
- streaming mode;
- request profile;
- selected backend;
- selected route;
- readiness scope;
- receipt ID attached to the response metadata;
- fallback status;
- TTFT and decode fields when the server path makes performance claims.

Exact-profile server readiness does not imply broad serving, concurrency,
streaming, speedup, or full residency.

## Promotion Rules

- TTFT claim requires first-token phase breakdown.
- Throughput claim requires decode profile and token count.
- Speedup requires exact same-artifact CPU comparator.
- Full residency requires per-phase residency proof.
- Server readiness performance claims require endpoint and streaming scope.
- Dense CUDA performance cannot satisfy BitNet QK256 performance.
- BitNet QK256 performance cannot satisfy dense regular-LLM performance.
- Generic `cuda` performance cannot satisfy strict RTX 5070 Ti performance
  until the receipt records the selected backend.
- Missing timing, transfer, or residency fields keep the related claim false.

## Receipt Explanation Contract

`bitnet receipts explain` should summarize performance evidence without
inventing claims. For each receipt it should be able to say:

```text
profile: <name>
selected backend: <backend>
selected route: <route>
fallback used: <true|false>
TTFT: <qualified|reported only|not available>
throughput: <qualified|reported only|not available>
speedup: <accepted|rejected|not reviewed>
residency: <full|partial|not proven>
server readiness: <exact profile|broad|not ready>
forbidden claims: <list>
```

## Proof Commands

Current docs-only validation:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
```

Runtime PRs that promote a performance claim must include the exact benchmark
or server command, plus receipt explanation:

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda --model <model> --profile <profile>
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- receipts explain --latest --format json
```

## Non-Goals

- Do not implement new benchmark collection in this spec.
- Do not promote any speed, residency, or server readiness claim in this spec.
- Do not change model coverage rows, receipts, runtime code, policy TOMLs, CI
  workflows, or generated dashboards.
- Do not require CUDA-only fields for CPU-only claims when the receipt records
  a clear not-applicable reason.
- Do not make one successful benchmark profile a global speed claim.

## Related Policy Or Manifest Sources

- `ci/model-artifacts/model-coverage-matrix.toml`
- `ci/hardware/windows-9950x3d-rtx5070ti/**`
- `policy/docs-source-of-truth.toml`
- `policy/ci-lanes.toml`
- `policy/ci-risk-packs.toml`
