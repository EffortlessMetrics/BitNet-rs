# Tokenizer and prompt template policy

Tokenizer and prompt template behavior must be explicit before proof. Fallback tokenizers and inferred chat templates are allowed only when receipts say fallback was used and no full support claim is made.

## Prompt mode fields

- Thinking enabled or disabled.
- Developer role used or not used.
- Tool calling used or not used.
- Reasoning effort when the family exposes that control.
- Structured output mode when used.

Small-model defaults such as Qwen3.5 reasoning disabled by default must be encoded in the prompt contract, not guessed at runtime.

