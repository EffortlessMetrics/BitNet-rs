# Apple M4 BitNet Proof Prep

`M4-CONT-005` prepared the M4 Mac mini side of the BitNet proof path. The same
command now has two bounded modes: artifact preflight and strict receipt
validation. Receipt validation can verify a completed Apple M4 BitNet
`answer-corpus` proof. BitNet is now limited to explicit one-shot
`bitnet mac ask` and fixed-prompt `bitnet mac bitnet-warm` with the accepted
GGUF plus external tokenizer; `bitnet mac chat` and `bitnet mac serve` remain
disabled for BitNet.

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

Not allowed now:

- Broad BitNet local-answer or chat quality works on M4.
- BitNet `mac chat` or `mac serve` works.
- Apple Metal BitNet inference works.
- QK256 works on Apple Silicon.
