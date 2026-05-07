# Model receipt requirements

Receipts are the only basis for working model-family claims.

## Required fields

Generative receipts must record model id, variant id, model hash or `TBD`, tokenizer source, prompt template id, requested backend, selected backend, fallback flag, task, generated token count, and explicit booleans for full-inference, multimodal, MoE, long-context, and speedup claims.

Token-classification receipts must record input token count, output shape, label taxonomy, span decoder, fallback flag, and `generation_claim=false`.

External-reference receipts must record the external runtime, host, command, artifact hash or `TBD`, and must set `local_execution_claim=false` unless the local repo executed the path.
