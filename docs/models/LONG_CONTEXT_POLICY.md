# Long-context policy

Model-card context length is not a bitnet-rs runtime claim.

Any context claim must state requested context, actual prompt length, generated/labeled tokens, KV/cache strategy, backend, fallback state, and whether the run used YaRN, sliding windows, global attention, compression, or another long-context mechanism. One-token smoke receipts at short context must set `long_context_claim=false`.
