# Model status values

The model-family tracker uses named stages so docs, scaffolds, smoke tests, receipts, and benchmarks cannot be collapsed into one vague “supported” claim.

## Values

- `not_present`
- `documented`
- `design_scaffold`
- `catalog_scaffold`
- `tokenizer_scaffold`
- `prompt_template_scaffold`
- `loader_scaffold`
- `synthetic_shape_tested`
- `runtime_detected`
- `one_token_smoke_tested`
- `parity_tested`
- `receipt_backed`
- `benchmarked`
- `design_only`

## Claim boundaries

| Status | May claim | Must not claim |
| --- | --- | --- |
| `documented` | The repo contains source-backed implementation notes. | The model loads or runs. |
| `design_scaffold` | The architecture plan is recorded. | The implementation compiles, loads, or runs. |
| `loader_scaffold` | The intended loader/tensor mapping exists. | Inference works. |
| `one_token_smoke_tested` | One strict deterministic generation/classification smoke ran. | Quality, speed, long context, or full feature support. |
| `receipt_backed` | The named model/variant/backend/task is proven by receipt. | Other variants, backends, modalities, or quantizations work. |

A doc-only Gemma 4 plan is therefore never a “Gemma 4 supported” claim.

