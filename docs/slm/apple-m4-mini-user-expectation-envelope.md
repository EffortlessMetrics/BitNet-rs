# Apple M4 Mini Dense SLM User Expectation Envelope

This page defines what a healthy M4 Mac mini should do for the supported dense
SLM path. It is an operator expectation document, not a broad Apple Silicon
benchmark. The evidence is scoped to the recorded M4 Mac mini receipts and the
supported dense Qwen model family.

## Supported User Path

Primary commands:

```bash
bitnet mac models
bitnet mac status
bitnet mac report-refresh
bitnet mac regression-dashboard
bitnet mac ask "What is 2+2?"
bitnet mac chat
bitnet mac smoke
bitnet mac doctor
bitnet mac regression <receipt.json> --baseline <baseline.json>
```

`bitnet mac models` is the operator-facing model-selection view. It lists the
default model, supported explicit-only dense models, cache state, the BitNet
one-shot ask plus fixed warm-session row, lifecycle policy states, and
disk-headroom guidance without downloading artifacts. Selectable states are
`default`, `supported-non-default`, and `supported-ask` within their recorded
scopes. `diagnostic-only`, `candidate`, `deprecated`, `rejected`, and `retired`
rows are not selectable operator routes. The text view also prints exact
`Next fetch` and `Next verify` commands for the recommended first supported
model when disk headroom is adequate. The JSON view includes per-row required
evidence, cache migration behavior, operator warnings, rollback guidance, and
claim-boundary updates. The BitNet row includes receipt bridge commands for
validating the strict `answer-corpus` proof and the fixed-prompt warm-session
proof. It is limited to explicit one-shot ask or fixed-prompt warm reuse with a
verified GGUF plus external tokenizer:

`bitnet mac status` is the model-free operator summary. It writes an
`apple_m4_inference_status` receipt with disk/cache posture, dense SLM readiness,
BitNet ask/warm readiness, local report inventory, supported command examples,
and explicit claim boundaries. It does not run live inference and keeps BitNet
chat and serve disabled.

`bitnet mac report-refresh` is the model-free report manifest. It writes an
`apple_m4_report_refresh_manifest` receipt that inventories committed dense SLM
and BitNet report families for advisory/nightly/release refreshes. It reads
committed receipts only; it does not download models, run live inference, or use
dense SLM evidence as BitNet evidence.

`bitnet mac regression-dashboard` is the model-free dashboard layer above that
manifest. It writes JSON and Markdown artifacts that group committed reports by
evidence family, model identity, tokenizer authority, backend, and fallback
state before offering regression commands. Groups with only one report are
marked `insufficient_history`.

The command-to-receipt operator map lives in
`docs/slm/apple-m4-operator-envelope-v2.md`. The durable refresh cadence,
matching-history thresholds, `resident_100` status, and disk/cache guidance
live in `docs/slm/apple-m4-operator-envelope-v3.md`.

```bash
bitnet mac ask \
  --model-id microsoft-bitnet-b1.58-2B-4T-i2s \
  --model-path models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json \
  "What is 2+2? Answer with only the number."
```

The lifecycle policy is not a runtime proof and does not add supported models,
change the default, enable BitNet chat or serve, or broaden Apple Silicon,
Metal, QK256, Neural Engine, MPSGraph, MacBook, quality, performance, or
speedup claims.

BitNet serve is explicit and gate-required. `bitnet mac serve --model-family
bitnet` fails before cache lookup or bind unless
`--bitnet-serve-gate-receipt` points at a ready
`bitnet_apple_m4_serve_gate` receipt. The gate records consumed BitNet chat
evidence, streaming-semantics evidence, timeout/failure evidence, and a
`mac serve-check --completion` receipt proving health/ready and per-request
receipt export on the gated local route. If a diagnostic-only, candidate,
deprecated, rejected, retired, or unknown model ID is passed to the dense Mac
commands, the wrapper fails before cache repair guidance and points back to
`bitnet mac models`.

BitNet chat is explicit and gate-required. `bitnet mac chat --model-family
bitnet` fails before prompt collection unless `--bitnet-chat-gate-receipt`
points at a ready `bitnet_apple_m4_chat_gate` receipt. The gate records
variable warm-session determinism, timeout/failure evidence,
streaming-semantics evidence, strict backend/fallback fields, and unchanged
serve-disabled claim boundaries. The chat route consumes that ready receipt and
writes `bitnet_apple_m4_chat_session`; BitNet serve remains a separate gated
route.

The first BitNet one-shot ask runtime receipt is:

```text
ci/hardware/apple-m4-mac-mini/2026-05-13/bitnet-mac-ask/bitnet-mac-ask-runtime-receipt.json
```

It records the fixed prompt `What is 2+2? Answer briefly.`, text
`2+2 equals 4.`, 8 generated token IDs, accepted model/tokenizer identity,
`apple-m4-cpu-neon` routing, `runtime_api=cpu`, `fallback_used=false`,
`first_token_ms=7536`, `decode_steady_state_tok_s=2.083`, and chat/server
disabled in the claim boundary. Treat this as a narrow one-shot route proof,
not broad BitNet chat quality or a performance envelope.

When `bitnet mac ask` starts with a verified model, it prints a compact stderr
summary covering model ID, quantization, cache root, backend, fallback status,
receipt path, and short SHA before generation begins.
For slow one-shot runs, pass `--progress` to emit stderr milestones for
tokenizer/model verification, model load, tokenizer load, prompt tokenization,
prefill, first token, decode completion, and receipt validation while generated
text remains on stdout. Pass `--quiet` when scripts need to suppress operator
stderr status and progress lines.
If the BitNet one-shot route fails during tokenizer verification, model
verification, or generation, it writes a `bitnet_apple_m4_mac_ask_failure`
receipt to `--json-out` before returning the error. That receipt records the
failure stage, repair guidance, explicit timeout-boundary status,
`fallback_used=false`, empty partial generation, and unchanged BitNet chat,
serve, Metal, QK256, Neural Engine, MPSGraph, broad-performance, and speedup
claim boundaries. The stderr error also prints compact `Repair guidance:` lines
covering the accepted external tokenizer SHA, explicit GGUF replacement or
cache fetch/verify commands, `bitnet mac models` cache inspection, and the
unchanged BitNet chat/serve-disabled boundary.
First-run missing-cache failures include both the exact `bitnet model fetch`
repair command and a `bitnet mac models --cache-dir ...` command with current
disk guidance, so low-space operators can choose the right supported model
before fetching.

The default model remains:

```text
model_id = qwen2.5-0.5b-instruct-q8_0
model = Qwen2.5 0.5B Instruct Q8_0
sha256 = ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
size_bytes = 675710816
cache_size_mib = 644.41
tokenizer_model = gpt2
tokenizer_pre = qwen2
prompt_template = qwen2.5
backend = apple-m4-cpu-neon
fallback_used = false
```

The supported non-default storage-conscious model is:

```text
model_id = qwen2.5-0.5b-instruct-q4_k_m
model = Qwen2.5 0.5B Instruct Q4_K_M
sha256 = 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
size_bytes = 491400032
cache_size_mib = 468.64
tokenizer_model = gpt2
tokenizer_pre = qwen2
prompt_template = qwen2.5
backend = apple-m4-cpu-neon
fallback_used = false
```

The supported non-default larger Qwen-class model is:

```text
model_id = qwen2.5-1.5b-instruct-q4_k_m
model = Qwen2.5 1.5B Instruct Q4_K_M
sha256 = 6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e
size_bytes = 1117320736
cache_size_mib = 1065.56
tokenizer_model = gpt2
tokenizer_pre = qwen2
prompt_template = qwen2.5
backend = apple-m4-cpu-neon
fallback_used = false
selection = explicit only; pass --model-id qwen2.5-1.5b-instruct-q4_k_m
```

The 1.5B model is supported for the M4 dense SLM lane after reference-output,
Rust M4 quality, deterministic duplicate-prompt, cache verification, and Mac
cache-check gates. It is not the default because it has a larger cache footprint
and materially slower warm-session timing than the 0.5B default.

Bounded M4 registration receipts for the 1.5B model:

```text
ci/quality/apple-m4-slm-model-breadth-qwen15-reference-sanity.toml
ci/quality/apple-m4-slm-model-breadth-qwen15-rust-m4-quality.toml
ci/quality/apple-m4-slm-model-breadth-qwen15-cache-registration.toml
```

The bounded Rust M4 quality gate for the 1.5B model recorded:

| Prompt | Generated tokens | TTFT ms | Decode tok/s | Normalized output |
|---|---:|---:|---:|---|
| `What is 2+2? Answer briefly.` | 9 | 22707 | 2.052 | `2+2 equals 4.` |
| `Name the capital of France.` | 8 | 14102 | 2.064 | `The capital of France is Paris.` |
| `Write one short sentence about Rust.` | 16 | 14808 | 2.028 | `Rust is a systems programming language known for its safety, speed, and` |

This is quality and registration evidence, not a release-mode performance
envelope. Publish a separate release-mode profile before making user-facing
latency expectations for the 1.5B model.

## Release-Mode Warm Envelope

Evidence receipt:

```text
ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/release-baseline.json
```

This receipt was recorded from the release-mode performance profile set. Cold
model load is separated from warm prompt timing. The profile set loads the model
and tokenizer once per profile and runs the bounded prompt group for that token
budget.

| Profile | Requested max tokens | Generated tokens | Cold model load ms | Tokenizer load ms | Warm prompt wall ms | Approx warm wall ms / prompt | TTFT mean ms | Decode tok/s | Total session ms | Peak memory MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `warm_16` | 16 | 34 | 3254.630 | 72.183 | 7666.475 | 2555.492 | 1885.000 | 14.962 | 12576.726 | 3772.469 |
| `warm_32` | 32 | 50 | 3118.775 | 49.173 | 8407.277 | 2802.426 | 1779.333 | 15.317 | 13117.047 | 4009.078 |
| `warm_64` | 64 | 82 | 3105.861 | 49.333 | 10459.755 | 3486.585 | 1763.000 | 15.269 | 15228.741 | 4026.438 |
| `warm_128` | 128 | 123 | 3174.666 | 54.097 | 13158.941 | 4386.314 | 1775.333 | 15.313 | 17896.347 | 4033.422 |

Healthy expectations for this M4 Mac mini:

- `requested_backend = apple-m4-cpu-neon`;
- `selected_backend = apple-m4-cpu-neon`;
- `runtime_api = cpu`;
- `fallback_used = false`;
- cold load is visible and separated from warm prompt timing;
- warm time-to-first-token is around the recorded release envelope for matching
  model, profile, and machine context;
- peak memory for the Q8_0 release profile is roughly 3.8-4.1 GB.
- 1.5B Q4_K_M performance is not covered by this release envelope yet; use its
  bounded registration receipts only to confirm coherent output and routing.

## Resident Soak Envelope

Evidence receipts:

```text
ci/hardware/apple-m4-mac-mini/2026-05-09/M4-SLM-EX-008/resident-25-64.json
ci/hardware/apple-m4-mac-mini/2026-05-09/M4-SLM-EX-008/resident-50-128.json
ci/hardware/apple-m4-mac-mini/2026-05-09/M4-SLM-EX-008/summary.json
```

These receipts exercise longer resident sessions. They are for stability and
memory/timing drift, not release-mode speed claims.

| Profile | Prompts | Max new tokens | Generated tokens | TTFT mean ms | Decode tok/s | Total session ms | Peak memory MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `resident_25_prompt_64_budget` | 25 | 64 | 482 | 4630.360 | 6.327 | 198130.965 | 4013.234 |
| `resident_50_prompt_128_budget` | 50 | 128 | 1185 | 4682.900 | 6.313 | 424606.240 | 4020.250 |

Both long-session receipts record:

- `model_loaded_once = true`;
- `tokenizer_loaded_once = true`;
- `quality_summary.passed = true`;
- deterministic repeated prompt groups pass;
- `fallback_used = false`.

The 50-prompt receipt increased peak memory by 7.016 MB over the 25-prompt
receipt in this run. That is the current local soak reference, not a fleet-wide
memory guarantee.

## Health And Regression Commands

Use `doctor` for one local health verdict:

```bash
bitnet mac doctor
```

The doctor receipt is model-free by default: it checks cache/hash readiness,
disk pressure, backend boundaries, and an advisory `checks.bitnet_ask` section
for the BitNet one-shot and fixed-prompt warm routes. It reports the
`supported-ask` catalog row, cached-model fetch/verify commands, accepted
tokenizer path and SHA, example cached-model ask and warm commands, and explicit
chat/serve/Metal-disabled claim boundaries. This readiness check does not make
dense SLM doctor fail when optional BitNet artifacts are absent. Use
`bitnet mac doctor --run-smoke` when the operator explicitly wants a live dense
SLM smoke receipt as part of the doctor flow.

Use `smoke` for a compact answer/cache receipt:

```bash
bitnet mac smoke
```

For the BitNet one-shot family, use the explicit family selector. This still
uses the one-shot `mac ask` route under the hood, defaults to the accepted
Microsoft I2_S model id and external tokenizer path, and keeps BitNet chat,
serve, full Metal, QK256, Neural Engine, MPSGraph, broad quality, and
performance claims disabled:

```bash
bitnet mac smoke --model-family bitnet
```

Committed local evidence for that BitNet smoke surface lives at
`ci/hardware/apple-m4-mac-mini/2026-05-14/bitnet-mac-smoke/bitnet-mac-smoke-runtime-receipt.json`,
with the paired answer receipt at
`ci/hardware/apple-m4-mac-mini/2026-05-14/bitnet-mac-smoke/bitnet-mac-smoke-runtime-receipt-answer.json`.
Those receipts prove one explicit accepted-artifact smoke run, not BitNet chat,
serve, full Metal inference, QK256 on Apple Silicon, or a broad performance
envelope.

For BitNet warm reuse proof, use the dedicated warm route. With no prompt flags
it loads the accepted model/tokenizer once, runs the fixed repeated-prompt proof
set, writes per-turn receipts plus an aggregate receipt, and still does not
enable BitNet chat or serve:

```bash
bitnet mac bitnet-warm
```

For operator prompt sets, repeat `--prompt` and include at least one exact
repeated prompt so the receipt can prove deterministic warm reuse:

```bash
bitnet mac bitnet-warm \
  --prompt "Answer with a single digit: 2+2=" \
  --prompt "Name the capital of France. Answer with one word." \
  --prompt "Answer with a single digit: 2+2="
```

For slow BitNet warm runs, pass `--progress` and, when scripting, an explicit
`--timeout-seconds <SECONDS>`. Progress goes to stderr and names tokenizer
verification, model verification, warm-session start, receipt write, and receipt
validation. If the run fails before the aggregate receipt is complete, the
wrapper writes a `bitnet_apple_m4_warm_session_failure` receipt to `--json-out`
with failure stage, timeout boundary, repair guidance, empty partial generation,
and the unchanged chat/serve/Metal-disabled claim boundary.

The operator-prompt route is still a warm-session proof surface, not BitNet
chat, BitNet serve, broad BitNet quality, or broad performance evidence.

Use the serve gate command after BitNet chat and service checks are available:

```bash
bitnet mac bitnet-serve-gate \
  --model-id microsoft-bitnet-b1.58-2B-4T-i2s \
  --chat-receipt ci/hardware/apple-m4-mac-mini/<date>/bitnet-chat/chat-session.json \
  --streaming-receipt ci/hardware/apple-m4-mac-mini/<date>/bitnet-serve-gate/streaming.json \
  --failure-receipt ci/hardware/apple-m4-mac-mini/<date>/bitnet-serve-gate/failure.json \
  --serve-check-receipt ci/hardware/apple-m4-mac-mini/<date>/bitnet-serve-gate/serve-check.json \
  --json-out ci/hardware/apple-m4-mac-mini/<date>/bitnet-serve-gate/gate.json
```

Only a ready gate can be consumed by `bitnet mac serve --model-family bitnet`.
That route is a local service wrapper only; it is not production hosting and
does not prove broad OpenAI compatibility.

Use the chat gate command to make missing BitNet chat evidence concrete:

```bash
bitnet mac bitnet-chat-gate \
  --model-id microsoft-bitnet-b1.58-2B-4T-i2s \
  --warm-receipt ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-productization/variable-warm-session.json \
  --failure-receipt <bitnet_apple_m4_warm_session_failure.json> \
  --streaming-receipt <bitnet_apple_m4_chat_streaming_semantics.json>
```

The receipt kind is `bitnet_apple_m4_chat_gate`. Missing timeout/failure or
streaming evidence leaves it `status=blocked` with `chat_enabled=false` and
`serve_enabled=false`; only `status=ready_to_enable` can be consumed by
`bitnet mac chat --model-family bitnet --bitnet-chat-gate-receipt <gate.json>`.

The committed aggregate warm receipt is:

```text
ci/hardware/apple-m4-mac-mini/2026-05-14/bitnet-warm/bitnet-mac-bitnet-warm-runtime-receipt.json
```

The committed operator-prompt warm receipt is:

```text
ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-productization/variable-warm-session.json
```

It records five operator prompts, ten generated tokens, generated text in
`prompts[].text`, generated token IDs in `prompts[].generated_token_ids`,
`bitnet_warm_prompt_source.source=operator_prompts`, repeated-prompt
determinism passed, `fallback_used=false`, `model_loaded_once=true`,
`tokenizer_loaded_once=true`, `total_session_ms=206419.425`, and resident
memory around 2.0 GiB for this bounded run. Treat this as variable warm-session
evidence only; BitNet chat and serve remain disabled.

Use `regression` for receipt-only drift checks against matching M4 dense SLM
envelopes:

```bash
bitnet mac regression \
  ci/hardware/apple-m4-mac-mini/2026-05-09/M4-SLM-EX-008/resident-25-64.json \
  --baseline ci/hardware/apple-m4-mac-mini/2026-05-09/M4-SLM-EX-008/resident-25-64.json
```

`bitnet mac regression` is advisory by default. Add `--fail-on-drift` for a
local operator hard failure when the matching receipt exceeds timing or memory
thresholds.

## Unsupported Claims

This envelope does not claim:

- broad BitNet local-answer or chat quality beyond the single committed
  one-shot ask receipt;
- QK256 on Apple Silicon;
- Neural Engine execution;
- MPSGraph model inference;
- full `apple-m4-metal` inference;
- broad Apple Silicon or M4 fleet performance.

Metal remains phase-scoped unless a strict full-pipeline receipt proves
otherwise. Dense Qwen success proves the M4 dense SLM runner path, not 1-bit
BitNet math.
