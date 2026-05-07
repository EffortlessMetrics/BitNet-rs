# Model Status Values

Model-family claims advance only through named statuses. Each status is narrow and must not be upgraded by implication from docs, hardware visibility, generic architecture detection, or fallback execution.

| Status | Meaning |
|---|---|
| `not_present` | No tracked model-family plan exists. |
| `documented` | Source-backed notes exist, but no implementation is claimed. |
| `design_scaffold` | Architecture and lane plans exist, but no runtime behavior is claimed. |
| `catalog_scaffold` | Catalog metadata can be drafted, but loading is not claimed. |
| `tokenizer_scaffold` | Tokenizer policy or intended source is recorded, not proven. |
| `prompt_template_scaffold` | Prompt/template contract is recorded, not proven. |
| `loader_scaffold` | Intended format/tensor mapping exists, but inference is not claimed. |
| `synthetic_shape_tested` | Synthetic shape checks ran without model-quality or real-weight claims. |
| `runtime_detected` | Runtime/backend surfaced, but execution is not a model claim. |
| `one_token_smoke_tested` | One strict deterministic generation/classification smoke ran. |
| `parity_tested` | A named reference comparison passed for a narrow variant/task/backend. |
| `receipt_backed` | The named model/variant/backend/task claim has a receipt. |
| `benchmarked` | A receipt-backed benchmark exists; no broader capability is implied. |
| `design_only` | Positive planning state for targets too large or unimplemented locally. |

## Claim boundaries

- `documented` may claim that repo notes exist; it must not claim the model loads or runs.
- `design_scaffold` may claim the architecture plan is recorded; it must not claim implementation compiles, loads, or runs.
- `loader_scaffold` may claim intended loader/tensor mapping exists; it must not claim inference works.
- `one_token_smoke_tested` may claim one deterministic smoke ran; it must not claim quality, speed, long context, or full feature support.
- `receipt_backed` may claim only the named model/variant/backend/task proven by receipt; it must not claim other variants, backends, modalities, or quantizations.
