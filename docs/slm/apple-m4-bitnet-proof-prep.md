# Apple M4 BitNet Proof Prep

`M4-CONT-005` prepared the M4 Mac mini side of the BitNet proof path. The same
command now has two bounded modes: artifact preflight and strict receipt
validation. Receipt validation can verify a completed Apple M4 BitNet
`answer-corpus` proof. BitNet is now limited to explicit one-shot
`bitnet mac ask` and receipt-gated `bitnet mac bitnet-warm` runs with the
accepted GGUF plus external tokenizer; `bitnet mac chat` and `bitnet mac serve`
remain disabled for BitNet.

## Command Contract

The prepared M4 command is:

```bash
bitnet mac bitnet-proof \
  --model <accepted-bitnet-gguf> \
  --tokenizer-authority <authority> \
  --accepted-artifact <artifact-acceptance-receipt.json> \
  --prompt "What is 2+2? Answer briefly." \
  --max-new-tokens 16 \
  --strict \
  --json-out <preflight-receipt.json>
```

In this continuity item the command is a preflight and contract checker. It
writes an `apple_m4_bitnet_proof_preflight` receipt and fails clearly when the
artifact is missing, not accepted, or missing tokenizer authority.

After the strict Apple M4 answer-corpus proof exists, use the receipt bridge:

```bash
bitnet --device apple-m4-cpu-neon mac bitnet-proof \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --proof-receipt ci/hardware/apple-m4-mac-mini/YYYY-MM-DD/bitnet-local-answer/bitnet-answer-corpus-full-release.json \
  --strict \
  --json-out ci/hardware/apple-m4-mac-mini/YYYY-MM-DD/bitnet-local-answer/mac-bitnet-proof-receipt-check.json
```

The bridge validates model SHA, strict external llama-bpe tokenizer authority,
all-passed corpus quality, per-case `apple-m4-cpu-neon` backend/fallback fields,
generated token IDs, prompt-prefill evidence, and timing/latency fields.

## One-Shot Ask Runtime Receipt

`M4-BITNET-ASK-001` adds the first user-facing one-shot runtime receipt for the
explicit BitNet `mac ask` route:

```bash
bitnet --device apple-m4-cpu-neon mac ask \
  --model-id microsoft-bitnet-b1.58-2B-4T-i2s \
  --model-path models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json \
  --max-new-tokens 8 \
  --json-out ci/hardware/apple-m4-mac-mini/2026-05-13/bitnet-mac-ask/bitnet-mac-ask-runtime-receipt.json \
  "What is 2+2? Answer briefly."
```

Receipt:

```text
ci/hardware/apple-m4-mac-mini/2026-05-13/bitnet-mac-ask/bitnet-mac-ask-runtime-receipt.json
```

The receipt records `artifact_kind=strict_bitnet_cpu_profile`, text
`2+2 equals 4.`, generated token IDs under `tokens.generated_ids`, accepted
model SHA `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`,
strict external `llama-bpe` tokenizer authority, `requested_backend` and
`selected_backend` equal to `apple-m4-cpu-neon`, `runtime_api=cpu`,
`fallback_used=false`, and explicit BitNet chat/server disablement in
`mac_bitnet_claim_boundary`.

Timing from that receipt:

```text
model_load_ms = 4459.083
tokenizer_load_ms = 178.805
prompt_tokenize_ms = 0.202
prefill_ms = 7015.247
first_token_ms = 7536
first_token_decode_ms = 521.519
decode_total_ms = 3881.495
decode_steady_state_tok_s = 2.083
total_wall_ms = 10896
```

## Variable Warm Runtime Receipt

`M4-BITNET-PROD-002` adds the first operator-prompt warm-session runtime receipt
for the explicit BitNet `mac bitnet-warm` route:

```bash
bitnet --device apple-m4-cpu-neon mac bitnet-warm \
  --model-id microsoft-bitnet-b1.58-2B-4T-i2s \
  --model-path models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json \
  --prompt "Answer with a single digit: 2+2=" \
  --prompt "Name the capital of France. Answer with one word." \
  --prompt "Return exactly: ready" \
  --prompt "Answer with a single digit: 3+1=" \
  --prompt "Answer with a single digit: 2+2=" \
  --max-new-tokens 8 \
  --json-out ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-productization/variable-warm-session.json
```

Receipt:

```text
ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-productization/variable-warm-session.json
```

The receipt records `artifact_kind=bitnet_apple_m4_warm_session`,
`bitnet_warm_prompt_source.source=operator_prompts`, `prompt_count=5`,
`generated_tokens=10`, accepted model SHA
`4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`,
strict external tokenizer SHA
`e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7`,
`requested_backend` and `selected_backend` equal to `apple-m4-cpu-neon`,
`runtime_api=cpu`, and `fallback_used=false`.

The aggregate receipt records generated text in `prompts[].text` and generated
token IDs in `prompts[].generated_token_ids`. The bounded run produced:

| Prompt | Text | Generated token IDs |
|---|---|---|
| `Answer with a single digit: 2+2=` | `4` | `[19, 128009]` |
| `Name the capital of France. Answer with one word.` | `Paris` | `[60704, 128009]` |
| `Return exactly: ready` | `ready` | `[2359, 128009]` |
| `Answer with a single digit: 3+1=` | `4` | `[19, 128009]` |
| `Answer with a single digit: 2+2=` | `4` | `[19, 128009]` |

Repeated-prompt determinism passed for the two `2+2` prompts with stable text
and stable generated token IDs. The session loaded the model and tokenizer once,
wrote per-prompt receipts, kept `chat_enabled=false` and `serve_enabled=false`,
and made no Metal, QK256, Neural Engine, MPSGraph, MacBook, speedup, broad
BitNet quality, or broad Apple Silicon claim.

Timing from that receipt:

```text
model_load_ms = 21003.804
tokenizer_load_ms = 727.672
total_session_ms = 206419.425
resident_memory_bytes = 2140225536
```

## Warm Progress And Failure Receipts

`M4-BITNET-PROD-003` adds warm-session operator diagnostics before BitNet chat is
allowed. Use `--progress` to print stderr milestones while keeping generated
text and receipt data separate:

```bash
bitnet --device apple-m4-cpu-neon mac bitnet-warm \
  --model-id microsoft-bitnet-b1.58-2B-4T-i2s \
  --model-path models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json \
  --prompt "Answer with a single digit: 2+2=" \
  --prompt "Answer with a single digit: 2+2=" \
  --max-new-tokens 8 \
  --timeout-seconds 300 \
  --progress \
  --json-out target/apple-m4-bitnet-productization/bitnet-warm.json
```

Progress stages include tokenizer verification, model verification, warm-session
start, receipt write completion, and receipt validation. The failure receipt
taxonomy also explicitly names the slow/failure zones operators need to debug:

```text
model_load
tokenizer_load
prompt_tokenize
prefill
first_token
decode
receipt_write
receipt_validation
```

If the warm route fails before a complete aggregate receipt is produced, it
writes a `bitnet_apple_m4_warm_session_failure` receipt to `--json-out`. That
receipt records the accepted model/tokenizer expectations, prompt count and
prompt hashes, `apple-m4-cpu-neon`, `runtime_api=cpu`, `fallback_used=false`,
empty partial generation, failure stage/message/elapsed time, timeout boundary,
repair guidance, and unchanged disabled chat/serve/Metal/QK256/Neural
Engine/MPSGraph/MacBook/broad-claim boundaries.

## BitNet Chat Gate

`M4-BITNET-PROD-004` made BitNet chat enablement explicit, and
`M4-BITNET-EX-006` keeps the route gate-required. `bitnet mac chat
--model-family bitnet` fails before prompt collection unless
`--bitnet-chat-gate-receipt` points at a ready gate receipt:

```bash
bitnet --device apple-m4-cpu-neon mac bitnet-chat-gate \
  --model-id microsoft-bitnet-b1.58-2B-4T-i2s \
  --warm-receipt ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-productization/variable-warm-session.json \
  --failure-receipt <bitnet_apple_m4_warm_session_failure.json> \
  --streaming-receipt <bitnet_apple_m4_chat_streaming_semantics.json> \
  --json-out target/apple-m4-bitnet-productization/bitnet-chat-gate.json
```

The gate receipt is `bitnet_apple_m4_chat_gate`. It records the accepted
Microsoft I2_S model SHA, external tokenizer SHA and authority,
`bitnetcpp-answer` prompt authority, strict `apple-m4-cpu-neon` backend,
`fallback_used=false`, variable warm-session evidence, repeated-prompt
determinism, timeout/failure evidence, streaming-semantics evidence, and
unchanged disabled chat/serve/Metal/QK256/Neural Engine/MPSGraph/MacBook/
broad-claim boundaries. Missing evidence keeps the receipt `status=blocked`.
Only `status=ready_to_enable` can be consumed by:

```bash
bitnet --device apple-m4-cpu-neon mac chat \
  --model-family bitnet \
  --model-id microsoft-bitnet-b1.58-2B-4T-i2s \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json \
  --bitnet-chat-gate-receipt <bitnet_apple_m4_chat_gate.json> \
  --prompt "Answer with a single digit: 2+2=" \
  --prompt "Name the capital of France. Answer with one word."
```

The chat receipt kind is `bitnet_apple_m4_chat_session`. It enables only the
accepted BitNet M4 CPU/NEON chat route and keeps BitNet serve, full Metal,
QK256, Neural Engine, MPSGraph, MacBook, speedup, broad quality, and broad
Apple Silicon performance claims disabled.

## Accepted Artifact Input

The `--accepted-artifact` receipt must come from the Apple BitNet artifact
sweep. The M4 Mac mini must not manufacture this receipt and must not download
large candidate artifacts for the sweep.

Required fields:

```json
{
  "accepted": true,
  "model": {
    "sha256": "<sha256>"
  },
  "tokenizer": {
    "authority": "llama-bpe-external"
  },
  "kernel_family": "i2_s|tl1"
}
```

`result = "accepted"` or `artifact.accepted = true` are also accepted as
compatibility forms, but the receipt still needs model SHA, tokenizer authority,
and kernel family.

## Future Strict Proof Receipt

The later M4 BitNet proof item must emit a strict local-answer receipt with:

```json
{
  "model": {
    "source": "...",
    "sha256": "..."
  },
  "tokenizer": {
    "authority": "..."
  },
  "kernel_family": "i2_s|tl1",
  "requested_backend": "apple-m4-cpu-neon",
  "selected_backend": "apple-m4-cpu-neon",
  "runtime_api": "cpu",
  "fallback_used": false,
  "generation": {
    "text": "...",
    "generated_token_ids": []
  },
  "timing": {},
  "claim_boundary": {
    "bitnet_answer_quality_claimed": true,
    "full_metal_inference_claimed": false
  }
}
```

## Claim Boundary

Allowed now:

- The M4 BitNet proof command shape and receipt contract are prepared.
- Missing or unaccepted artifacts fail before proof execution.
- The explicit BitNet one-shot `bitnet mac ask` route has one committed
  Apple M4 CPU/NEON runtime receipt for the fixed short prompt above.
- The fixed-prompt `bitnet mac bitnet-warm` route has a committed Apple M4
  CPU/NEON aggregate receipt plus per-prompt receipts for repeated-prompt
  determinism.
- The `bitnet mac bitnet-warm --prompt ... --prompt ...` route accepts
  operator-provided warm prompt sets only when at least one exact prompt is
  repeated, so deterministic warm reuse remains receipt-gated before chat.
- The operator-prompt warm route has one committed Apple M4 CPU/NEON runtime
  receipt for the five-prompt bounded set above.
- BitNet warm-session failures write partial-failure receipts with repair
  guidance and an explicit progress/timeout stage taxonomy.

Not allowed now:

- Broad BitNet local-answer or chat quality works on M4.
- BitNet `mac chat` or `mac serve` works.
- Apple Metal BitNet inference works.
- QK256 works on Apple Silicon.
