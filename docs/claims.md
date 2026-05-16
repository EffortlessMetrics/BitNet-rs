# Claim Ledger

Generated from `ci/claims/claim-ledger.json`.

| Claim | Status | Scope | Blocker |
| --- | --- | --- | --- |
| `a770.bitnet.trusted_partial_experience` | `diagnostic` | BitNet b1.58 i2_s on AMD 5700X + Intel Arc A770 | clean claim-grade parent benchmark, claimable A770 routes, strict resource envelope, and same-route history are not present |
| `a770.bitnet.selected_attention` | `diagnostic` | BitNet selected attention on AMD 5700X + Intel Arc A770 | value-level score implementation rule is not available and decode gates are not promoted |
| `a770.bitnet.full_residency` | `unsupported` | Full BitNet device residency on Intel Arc A770 | resident KV, attention scores, softmax, value mix, full support-op residency, and full device residency are not claimed |
| `a770.gemma_class.support` | `unsupported` | Gemma-class dense models on Intel Arc A770 | no Gemma-class model contract, dense kernels, tokenizer/template proof, quality proof, or A770 resource receipts are present |

## A770 Not Claims

- `selected_attention_residency`
- `resident_kv_decode`
- `attention_scores_residency`
- `softmax_residency`
- `attention_value_mix_residency`
- `full_support_op_residency`
- `full_device_residency`
- `completion`
