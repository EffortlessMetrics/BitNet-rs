# Long-context policy

Source-backed context length is a model capability, not a bitnet-rs runtime claim. A one-token proof at 2,048 tokens does not imply 128K, 200K, 256K, or 1M context support.

Long-context receipts must record requested context, achieved context, KV/cache policy, RoPE/YaRN/scaling configuration, fallback status, and whether the run was synthetic or full inference.

