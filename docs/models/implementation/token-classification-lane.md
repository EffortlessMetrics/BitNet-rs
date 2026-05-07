# Token classification lane

Token classification is not generation.

## Required

- encoder/bidirectional attention support;
- token classification head;
- per-token logits;
- label taxonomy;
- constrained decoder if required;
- span output receipt.

OpenAI privacy-filter uses this lane. Receipts must include token count, output shape, label set, span decoder, `generation_claim=false`, fallback status, and speedup claim status.
