# QK256 (GGML I2_S) Loader Implementation

## Summary

This document describes the implementation of QK256 (GGML I2_S with 256-element blocks) support in the BitNet-rs GGUF loader. The implementation enables loading QK256-quantized weights from GGUF files and storing them as U8 tensors for later use by the linear layer.

## Implementation Details

### 1. QK256 Detection (gguf_simple.rs:783-834)

The loader detects QK256 format by analyzing the tensor's available bytes:

```rust
// Calculate expected bytes for different formats
let blocks_32 = nelems.div_ceil(32);    // BitNet I2_S (32 elem/block)
let blocks_256 = nelems.div_ceil(256);  // GGML I2_S (256 elem/block)
let ggml_need = blocks_256 * 64;        // 256 elem/block, 64 B/block

// Detect QK256 format by available bytes
if available.abs_diff(ggml_need) <= tolerance {
    // This is QK256 format - store as U8 tensor
}
```

**Detection criteria:**
- Tensor type is `GgufTensorType::I2_S`
- Available bytes match QK256 format: `blocks_256 * 64 ± tolerance`
- Tolerance allows for alignment padding (128 bytes)

### 2. U8 Tensor Creation (gguf_simple.rs:790-833)

Once QK256 format is detected, the loader creates a U8 Candle tensor:

```rust
// Calculate dimensions for QK256 storage
let (rows, cols) = if info.shape.len() == 2 {
    (info.shape[0] as usize, info.shape[1] as usize)
} else if info.shape.len() == 1 {
    (1, info.shape[0] as usize)  // 1D → single row
} else {
    return Err(...);  // Unsupported shape
};

let blocks_per_row = (cols + 255) / 256;     // ceil(cols/256)
let row_stride_bytes = blocks_per_row * 64;  // 64 bytes per block
let needed_bytes = rows * row_stride_bytes;

// Create U8 tensor [rows, row_stride_bytes]
let qs_u8 = CandleTensor::from_vec(
    tensor_data[..needed_bytes].to_vec(),
    &[rows, row_stride_bytes],
    device,
)?
.to_dtype(DType::U8)?;
```

**Tensor dimensions:**
- **Shape:** `[rows, row_stride_bytes]`
- **row_stride_bytes:** `ceil(cols / 256) * 64`
- **Data type:** `DType::U8`

### 3. Derived Key Storage (gguf_simple.rs:160-180)

QK256 tensors are stored with derived keys to distinguish them from regular tensors:

```rust
// Check if this is a QK256 U8 tensor
let is_qk256 = tensor.dtype() == DType::U8
    && tensor.dims().len() == 2
    && tensor.dims()[1] % 64 == 0
    && info.tensor_type == GgufTensorType::I2_S;

if is_qk256 {
    // Store under derived key: "{original_name}.qk256_qs"
    let qk_key = format!("{}.qk256_qs", info.name);
    tensor_map.insert(qk_key, tensor);
    // Do NOT insert under original key
} else {
    // Regular tensor - store under original key
    tensor_map.insert(info.name.clone(), tensor);
}
```

**Key naming convention:**
- **Original tensor name:** `blk.0.attn_q.weight`
- **Derived QK256 key:** `blk.0.attn_q.weight.qk256_qs`

**Detection logic:**
- `DType::U8` - Indicates raw quantized bytes
- 2D shape with `dim[1] % 64 == 0` - QK256 block alignment
- `GgufTensorType::I2_S` - I2_S quantization type

### 4. Type System Enhancement (types.rs:606)

Added `PartialEq` to `GgufTensorType` for comparison:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub enum GgufTensorType {
    F32, F16, F64,
    I2_S,  // Can now be compared
    // ... other types
}
```

## QK256 Format Specification

### Block Structure

Each QK256 block contains 256 elements packed into 64 bytes:

```
Block Layout (64 bytes):
- Byte 0-1:   Scale (F16)
- Byte 2-3:   Min value (F16)
- Byte 4-67:  Packed 2-bit weights (256 elements / 4 per byte = 64 bytes)
```

**Note:** The exact layout details are implementation-specific and will be handled by the QK256 dequantization kernel.

### Matrix Storage

For a weight matrix with shape `[rows, cols]`:

```
Storage dimensions:
- blocks_per_row = ceil(cols / 256)
- row_stride_bytes = blocks_per_row * 64
- U8 tensor shape: [rows, row_stride_bytes]
```

**Example:**
```
Matrix: 2048 × 2048
blocks_per_row = ceil(2048 / 256) = 8
row_stride_bytes = 8 * 64 = 512 bytes
U8 tensor: [2048, 512] = 1,048,576 bytes
```

## Integration with Linear Layer

The linear layer can detect QK256 weights by checking for the `.qk256_qs` suffix:

```rust
// In linear layer implementation
if let Some(qk256_tensor) = tensors.get(&format!("{}.qk256_qs", weight_name)) {
    // Use QK256 dequantization kernel
    let weights = dequantize_qk256(qk256_tensor)?;
} else if let Some(regular_tensor) = tensors.get(weight_name) {
    // Use regular tensor directly
    let weights = regular_tensor.clone();
}
```

## Testing

### Unit Tests

Created `crates/bitnet-models/tests/qk256_detection.rs` with tests for:
1. **Detection logic verification** - Documents the QK256 detection criteria
2. **Calculation logic verification** - Tests dimension calculations for various matrix sizes

### Test Cases

```rust
// Test case: 2048 × 2048 matrix
rows: 2048, cols: 2048
blocks_per_row = (2048 + 255) / 256 = 8
row_stride_bytes = 8 * 64 = 512
needed_bytes = 2048 * 512 = 1,048,576

// Test case: 11008 × 2048 matrix (FFN intermediate)
rows: 11008, cols: 2048
blocks_per_row = (2048 + 255) / 256 = 8
row_stride_bytes = 8 * 64 = 512
needed_bytes = 11008 * 512 = 5,636,096
```

## Expected Behavior

### Before Implementation

```
Loading QK256 GGUF model...
ERROR: I2_S 'blk.0.attn_q.weight': GGML/llama.cpp format detected (QK_K=256, 64B/block, no scale tensor).
       This format requires GGML-compatible kernels and is not yet supported in pure Rust.
       Please use the FFI backend with --features ffi or set BITNET_CPP_DIR.
```

### After Implementation

```
Loading QK256 GGUF model...
INFO: I2_S 'blk.0.attn_q.weight': GGML/llama.cpp format detected (QK_K=256, 64B/block). Storing as U8 tensor for QK256 kernel.
DEBUG: I2_S 'blk.0.attn_q.weight': Created QK256 U8 tensor with shape [2048, 512] (1048576 bytes)
DEBUG: Storing QK256 tensor under derived key: 'blk.0.attn_q.weight.qk256_qs' (shape: [2048, 512])
```

## Files Modified

1. **crates/bitnet-models/src/gguf_simple.rs**
   - Modified QK256 detection (lines 783-834)
   - Updated tensor loading loop (lines 160-180)

2. **crates/bitnet-models/src/formats/gguf/types.rs**
   - Added `PartialEq` to `GgufTensorType` (line 606)

3. **crates/bitnet-models/tests/qk256_detection.rs** (new file)
   - Unit tests for QK256 detection and calculation logic

## Next Steps

1. **Implement QK256 dequantization kernel** in `crates/bitnet-models/src/quant/i2s_qk256.rs`
2. **Wire linear layer** to detect and use QK256 tensors
3. **Add integration tests** with real QK256 GGUF models
4. **Validate parity** with GGML reference implementation

## References

- GGML I2_S format: https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c
- BitNet quantization: `docs/reference/quantization-support.md`
- GGUF loader: `docs/architecture/model-loading.md`
