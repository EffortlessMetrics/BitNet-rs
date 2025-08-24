# How to Configure BitNet.rs

This guide explains how to configure BitNet.rs using configuration files and environment variables.

## Configuration File

Create a `bitnet.toml` configuration file to manage your settings:

```toml
[model]
default_model = "microsoft/bitnet-b1_58-large"
cache_dir = "~/.cache/bitnet"

[inference]
device = "auto"  # "cpu", "cuda", or "auto"
max_batch_size = 8
kv_cache_size = 2048

[generation]
max_new_tokens = 512
temperature = 0.7
top_p = 0.9
top_k = 50
repetition_penalty = 1.0
```

## Environment Variables

BitNet.rs respects these environment variables for configuration overrides:

- `BITNET_MODEL_CACHE`: Model cache directory
- `BITNET_DEVICE`: Default device ("cpu", "cuda", "auto")
- `BITNET_LOG_LEVEL`: Log level ("trace", "debug", "info", "warn", "error")
- `CUDA_VISIBLE_DEVICES`: GPU device selection
