# Intel GPU Performance Tuning Guide

Performance tuning reference for BitNet-rs inference on Intel GPUs via OpenCL.

## Profiling Tools

| Tool | Purpose |
|------|---------|
| `intel_gpu_top` | Real-time GPU utilization monitoring |
| `BITNET_OPENCL_PROFILE=1` | Kernel event timing (enqueue → complete) |
| `clinfo` | Device capability query (compute units, memory, extensions) |

```bash
# Check available OpenCL devices
clinfo --list

# Monitor GPU utilization during inference
intel_gpu_top &
cargo run -p bitnet-cli --no-default-features --features cpu,oneapi,full-cli -- run \
  --model models/model.gguf --tokenizer models/tokenizer.json \
  --prompt "Hello" --max-tokens 16

# Enable kernel profiling
BITNET_OPENCL_PROFILE=1 cargo run -p bitnet-cli --no-default-features \
  --features cpu,oneapi,full-cli -- run --model models/model.gguf \
  --tokenizer models/tokenizer.json --prompt "Hello" --max-tokens 16
```

## Work-Group Sizing

Intel Arc GPUs use Xe-cores with 16 EUs (Execution Units) each:
- **A770**: 32 Xe-cores × 16 EUs = 512 EUs
- **A750**: 28 Xe-cores × 16 EUs = 448 EUs
- **A380**: 8 Xe-cores × 16 EUs = 128 EUs

**Optimal local work size**: multiples of 16 (subgroup/SIMD width).

```bash
# Override local work size for tuning
BITNET_OPENCL_LOCAL_SIZE=16x16 cargo run ...

# Common configurations to benchmark
BITNET_OPENCL_LOCAL_SIZE=8x8    # Conservative, good for small kernels
BITNET_OPENCL_LOCAL_SIZE=16x16  # Default, matches subgroup size
BITNET_OPENCL_LOCAL_SIZE=32x8   # Wide, good for matmul with large M
```

## Memory Optimization

### Zero-Copy Access (Integrated GPUs)

For Intel iGPUs, use `CL_MEM_ALLOC_HOST_PTR` to enable zero-copy access
between CPU and GPU, avoiding explicit transfers:

```
# Enable zero-copy on iGPU (auto-detected when available)
BITNET_OPENCL_ZERO_COPY=1
```

### Buffer Pool

Repeated OpenCL buffer allocation is expensive. The buffer pool in
`bitnet-kernels` reuses allocations across inference steps:

```
# Cap GPU memory usage (default: 75% of device memory)
BITNET_GPU_MEM_LIMIT_MB=4096
```

### Memory Bandwidth

| Device | Bandwidth | Notes |
|--------|-----------|-------|
| Arc A770 (16GB) | 560 GB/s | Full PCIe x16 Gen4 |
| Arc A750 (8GB) | 512 GB/s | PCIe x16 Gen4 |
| Intel iGPU (shared) | System RAM speed | Zero-copy preferred |

## Kernel Optimization Checklist

1. **Use `float4` vectorization** where possible — 4× throughput on Intel GPUs
2. **Minimize host↔device transfers** — batch operations before transferring results
3. **Prefer fused kernels** — SiLU+gate, QK+softmax reduce memory round-trips
4. **Tile matmul with `__local` memory** — fits in SLM (shared local memory)
5. **Pre-compile kernels to SPIR-V** — avoids JIT compilation latency on first run

## Benchmarking

```bash
# Run OpenCL-specific benchmarks
RUSTFLAGS="-C target-cpu=native" cargo bench --features bench,oneapi -- opencl

# Compare CPU vs GPU for specific operations
RUSTFLAGS="-C target-cpu=native" cargo bench --features bench,oneapi -- matmul
RUSTFLAGS="-C target-cpu=native" cargo bench --features bench,oneapi -- quantize

# Quick smoke test with timing
time BITNET_OPENCL_PROFILE=1 cargo run -p bitnet-cli --release \
  --no-default-features --features cpu,oneapi,full-cli -- run \
  --model models/model.gguf --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" --max-tokens 16
```

## Known Limitations

- **QK256 scalar kernels**: ~0.1 tok/s — use the optimized tiled version instead
- **First kernel launch is slow**: JIT compilation overhead — pre-compile to SPIR-V
  with `BITNET_OPENCL_SPIRV_CACHE=~/.cache/bitnet/spirv`
- **PCIe bandwidth**: Discrete GPUs limited by PCIe x16 transfer speed for
  host↔device copies; prefer keeping tensors on-device
- **Driver version**: Requires Intel compute-runtime ≥ 23.22 for full OpenCL 3.0

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `BITNET_OPENCL_PROFILE` | `0` | Enable kernel event timing |
| `BITNET_OPENCL_LOCAL_SIZE` | auto | Work-group size override (e.g., `16x16`) |
| `BITNET_OPENCL_ZERO_COPY` | auto | Force zero-copy buffer allocation |
| `BITNET_OPENCL_SPIRV_CACHE` | none | Directory for pre-compiled SPIR-V kernels |
| `BITNET_GPU_MEM_LIMIT_MB` | 75% VRAM | Maximum GPU memory allocation |
