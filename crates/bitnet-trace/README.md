# bitnet-trace

Tensor activation tracing for BitNet.rs cross-validation debugging.

## Overview

`bitnet-trace` provides utilities for capturing and hashing tensor activations during inference, enabling systematic cross-validation against reference implementations. It computes Blake3 hashes and RMS (root mean square) statistics of tensors, writing the results to JSON trace files.

## Features

- **Environment-controlled**: Only traces when `BITNET_TRACE_DIR` is set
- **Cryptographic hashing**: Uses Blake3 for fast, secure tensor fingerprinting
- **Statistical metadata**: Captures RMS, shape, dtype, and element count
- **Zero overhead when disabled**: Returns immediately if tracing is not enabled
- **Automatic directory creation**: Creates trace directory if it doesn't exist

## Usage

### Basic Example

```rust
use candle_core::{Tensor, Device};
use bitnet_trace::dump_trace;

fn main() -> candle_core::Result<()> {
    let tensor = Tensor::randn(0f32, 1f32, (32, 128), &Device::Cpu)?;

    // Only traces if BITNET_TRACE_DIR is set
    dump_trace("layer_output", &tensor)?;

    Ok(())
}
```

### Enable Tracing

Set the `BITNET_TRACE_DIR` environment variable to enable tracing:

```bash
export BITNET_TRACE_DIR=/tmp/bitnet-traces
cargo run -p bitnet-cli -- run --model model.gguf --prompt "test"
```

Trace files will be written to `$BITNET_TRACE_DIR/{name}.trace`.

### Trace File Format

Trace files are JSON documents with the following structure:

```json
{
  "name": "blk0/attn_norm",
  "shape": [1, 2560],
  "dtype": "F32",
  "blake3": "abc123...",
  "rms": 0.9982,
  "num_elements": 2560
}
```

Fields:
- `name`: Tensor identifier (path separators replaced with underscores in filename)
- `shape`: Tensor dimensions
- `dtype`: Data type (captured before conversion to F32)
- `blake3`: 64-character hex Blake3 hash of tensor data (as F32 little-endian bytes)
- `rms`: Root mean square of tensor values
- `num_elements`: Total number of elements

## Cross-Validation Workflow

1. **Capture Rust traces**:
   ```bash
   export BITNET_TRACE_DIR=/tmp/rust-traces
   cargo run -p bitnet-cli -- run --model model.gguf --prompt "test"
   ```

2. **Capture reference traces** (e.g., from C++ implementation):
   ```bash
   export BITNET_TRACE_DIR=/tmp/cpp-traces
   ./reference_impl --model model.gguf --prompt "test"
   ```

3. **Compare trace files**:
   ```bash
   # Compare Blake3 hashes
   diff <(jq -r '.blake3' /tmp/rust-traces/*.trace | sort) \
        <(jq -r '.blake3' /tmp/cpp-traces/*.trace | sort)

   # Compare RMS values
   diff <(jq -r '.rms' /tmp/rust-traces/*.trace | sort) \
        <(jq -r '.rms' /tmp/cpp-traces/*.trace | sort)
   ```

## Design Patterns

This crate follows similar design patterns to `KernelRecorder` in `bitnet-inference`:

- **Environment-controlled**: Uses environment variable to enable/disable tracing
- **Graceful degradation**: Returns `Ok(())` silently when disabled
- **Sequential operation**: No thread synchronization needed (traces written during sequential inference)
- **Minimal overhead**: Fast Blake3 hashing, single tensor conversion to F32

## Performance Considerations

- **Disabled by default**: Zero overhead when `BITNET_TRACE_DIR` is not set
- **Fast hashing**: Blake3 is optimized for high throughput
- **Single conversion**: Tensor is converted to F32 once for both hashing and RMS
- **Sequential I/O**: Trace files written sequentially during inference

## Dependencies

- `blake3`: Fast cryptographic hashing
- `candle-core`: Tensor operations
- `serde`: Serialization framework
- `serde_json`: JSON output format

## License

MIT OR Apache-2.0
