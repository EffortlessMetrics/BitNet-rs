# Token Classification Lane

Token classification is not generation.

Required:
- encoder/bidirectional attention support
- token classification head
- per-token logits
- label taxonomy
- constrained decoder if required
- span output receipt

Receipts should record input token count, output shape, label taxonomy, span decoder, fallback status, and `generation_claim=false`.
