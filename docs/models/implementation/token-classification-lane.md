# Token-classification lane

Token classification is not generation.

## Required

- Encoder/bidirectional attention support.
- Token classification head.
- Per-token logits.
- Label taxonomy.
- Constrained decoder if required.
- Span output receipt.

OpenAI privacy-filter receipts must report labeled-token coverage and span decoder behavior rather than generated tokens.

