# Tokenizer and prompt template policy

Tokenizer and prompt template behavior is model-family-specific and must be explicit.

- Tokenizer source must be `explicit`, `model`, `sibling`, or `unknown`.
- Prompt template ids must be versioned and recorded in receipts.
- Thinking modes, developer roles, tool-calling blocks, reasoning effort, and multimodal placeholders must not be inferred from another family.
- If a model requires chat-template kwargs to enable or disable thinking, the family doc must record the default and receipt setting.
- Tokenizer or prompt-template scaffolds are not inference claims.
