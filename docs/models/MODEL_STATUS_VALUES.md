# Model status values

Model-family statuses mirror the backend tracker style: each status has a narrow claim boundary and must not imply later proof stages.

## Values

- `not_present`: no repository artifact exists.
- `documented`: source-backed notes exist; no implementation claim.
- `design_scaffold`: architecture and implementation plan are recorded; no compile/load/run claim.
- `catalog_scaffold`: registry/catalog shape is planned or present; no loader claim.
- `tokenizer_scaffold`: tokenizer policy or mapping is planned; no tokenization parity claim.
- `prompt_template_scaffold`: prompt template policy is planned; no chat parity claim.
- `loader_scaffold`: intended loader/tensor mapping exists; no inference claim.
- `synthetic_shape_tested`: synthetic shape-only checks ran; no real model claim.
- `runtime_detected`: runtime/backend was detected; no inference claim.
- `one_token_smoke_tested`: one deterministic generation/classification smoke ran; no quality, speed, modality, or long-context claim.
- `parity_tested`: parity was checked for the named path only.
- `receipt_backed`: the named model/variant/backend/task claim is proven by receipt.
- `benchmarked`: benchmark data exists for the receipt-backed path only.
- `design_only`: intentionally documented for future work or external reference; no local execution claim.

## Claim boundaries

- `documented` may claim source-backed implementation notes exist, but must not claim the model loads or runs.
- `design_scaffold` may claim the architecture plan is recorded, but must not claim the implementation compiles, loads, or runs.
- `loader_scaffold` may claim the intended loader/tensor mapping exists, but must not claim inference works.
- `one_token_smoke_tested` may claim one strict deterministic smoke ran, but must not claim quality, speed, long context, or full feature support.
- `receipt_backed` may claim the named model/variant/backend/task is proven by receipt, but must not claim other variants, backends, modalities, or quantizations work.
