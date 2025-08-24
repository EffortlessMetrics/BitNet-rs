# How to Optimize Performance

This guide provides steps to optimize the performance of BitNet.rs.

## CPU Optimization

1. **Enable CPU features**:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

2. **Tune thread count**:
```bash
export RAYON_NUM_THREADS=8
bitnet-cli inference --model model.gguf --prompt "Hello"
```

## GPU Optimization

1. **Enable mixed precision**:
```rust
let config = InferenceConfig {
    use_mixed_precision: true,
    ..Default::default()
};
```

2. **Optimize batch size**:
```rust
let config = InferenceConfig {
    max_batch_size: 16,  // Adjust based on GPU memory
    ..Default::default()
};
```
