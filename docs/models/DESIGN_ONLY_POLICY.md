# Design-only policy

`design_only` is an explicit positive tracker state for models that are too large or too incomplete for local proof today. It allows source-backed architecture, prompt, tokenizer, loader, and external-reference plans while denying local execution claims.

Design-only entries must state:

- No loader claim.
- No inference claim.
- No speedup claim.
- No local residency claim.
- Which future receipt would be required to advance.

