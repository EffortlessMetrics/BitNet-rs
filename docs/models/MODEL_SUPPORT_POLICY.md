# Model support policy

Support is evidence-based and scoped. The repository must not describe a model family as supported unless the exact support claim is backed by a status value and, for working claims, a receipt.

## Required dimensions

Every support claim must name:

- model family and variant;
- weight format and quantization;
- tokenizer source;
- prompt template id;
- requested and selected backend;
- task shape such as text generation or token classification;
- whether fallback was used;
- whether multimodal, MoE, long-context, full-inference, or speedup support is claimed.

## Non-BitNet boundary

Non-BitNet dense and MoE families must not use BitNet W1.58, QK256, or BitNet-specific kernel receipts as proof. If a non-BitNet path falls back to CPU, a reference graph, or an external runtime, the receipt must say so.
