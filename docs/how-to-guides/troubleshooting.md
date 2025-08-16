# Troubleshooting

This guide provides solutions to common issues you might encounter while using BitNet Rust.

### Common Issues

1. **CUDA not found**:
   - Install CUDA 11.8 or later
   - Set `CUDA_HOME` environment variable
   - Build with `--no-default-features --features cli` for CPU-only

2. **Model loading fails**:
   - Check model format compatibility
   - Verify model file integrity
   - Ensure sufficient disk space and memory

3. **Poor performance**:
   - Enable native CPU features with `RUSTFLAGS="-C target-cpu=native"`
   - Use GPU acceleration if available
   - Adjust batch size and thread count

### Debug Mode

Enable debug logging:

```bash
RUST_LOG=debug bitnet-cli inference --model model.gguf --prompt "Hello"
```

### Memory Issues

Monitor memory usage:

```bash
bitnet-cli benchmark --model model.gguf --monitor-memory
```

### Getting Help

If you can't find a solution here, please check out our other resources:

- [GitHub Issues](https://github.com/your-org/bitnet-rust/issues)
- [Discord Community](https://discord.gg/bitnet-rust)
