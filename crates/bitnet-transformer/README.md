# bitnet-transformer

BitNet transformer architecture implementation.

Contains the core neural network layers: `RotaryEmbedding`, `MultiHeadAttention`, `FeedForward`, `TransformerBlock`, `TransformerModel`, and `KVCache`.

RoPE cache table generation is delegated to the `bitnet-rope` microcrate.
