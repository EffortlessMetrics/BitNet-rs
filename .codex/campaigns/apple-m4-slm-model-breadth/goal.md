# apple-m4-slm-model-breadth

Goal: expand the Apple M4 dense SLM path beyond the current Qwen2.5 supported
set by adding exact, pinned, storage-conscious dense instruct models only after
reference output sanity, Rust M4 quality, tokenizer authority, cache metadata,
receipt validation, and deterministic behavior gates pass.

Keep Qwen2.5 Q8_0 as the default until a future item explicitly changes that
with receipts. Never commit model binaries.
