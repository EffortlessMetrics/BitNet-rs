# Token-classification lane

Token classification is not generation.

Required:

- encoder/bidirectional attention support
- token classification head
- per-token logits
- label taxonomy
- constrained decoder if required
- span output receipt

Receipts must set `generation_claim=false` and include output shape, label names or taxonomy hash, span decoder, fallback status, and speedup_claim=false unless benchmarked separately.
