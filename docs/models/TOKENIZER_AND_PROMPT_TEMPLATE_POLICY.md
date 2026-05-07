# Tokenizer and prompt-template policy

Tokenizer and prompt-template compatibility are separate proof dimensions. A model must not be considered supported when only weights load or when a generic family template happens to produce tokens.

## Tokenizer rules

- Record tokenizer source: explicit, model, sibling, or unknown.
- Treat tokenizer fallback as a claim boundary.
- Do not infer vocabulary size or special tokens unless source-backed.

## Prompt-template rules

- Record prompt-template ID and source.
- Record thinking mode, developer role, reasoning effort, tool calling, and structured-output switches.
- Small-model defaults must be explicit, especially when thinking is disabled by default.
- Template-only docs do not imply loader or inference support.
