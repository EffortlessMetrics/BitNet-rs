# Apple M4 BitNet Proof Prep

`M4-CONT-005` prepared the M4 Mac mini side of the BitNet proof path. The same
command now has two bounded modes: artifact preflight and strict receipt
validation. Receipt validation can verify a completed Apple M4 BitNet
`answer-corpus` proof, but it still does not enable BitNet through `bitnet mac
ask`, `bitnet mac chat`, or `bitnet mac serve`.

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

Not allowed now:

- BitNet local-answer quality works on M4.
- The artifact is accepted by M4 continuity work.
- Apple Metal BitNet inference works.
- QK256 works on Apple Silicon.
