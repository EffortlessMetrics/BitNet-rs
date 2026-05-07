# Tokenizer and Prompt Template Policy

Tokenizer and prompt templates are part of the model claim. A loader cannot silently choose a fallback tokenizer or generic chat template and claim family support.

Receipts must record tokenizer source, prompt-template id, thinking-mode flag, developer-role usage, tool-calling usage, and fallback status. Thinking and non-thinking modes require separate template settings where model cards distinguish them.
