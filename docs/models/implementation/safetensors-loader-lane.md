# safetensors loader lane

This lane covers future safetensors loading for model families that publish Hugging Face-style artifacts.

## Required gates

- Config-to-architecture mapping.
- Tensor name map.
- Tokenizer and chat-template source.
- Weight dtype and quantization policy.
- Hashes for config, tokenizer, and shards.
- Shape-only checks before inference claims.

A safetensors loader scaffold is not inference support. Remote code requirements must be marked `remote_code_tbd` until audited.
