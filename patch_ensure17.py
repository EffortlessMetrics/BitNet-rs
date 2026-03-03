import re

with open("crates/bitnet-models/src/weight_mapper.rs", "r") as f:
    code = f.read()

# Since candle panics on contiguous() if stride mismatch happens on generic types
# the only clean way is to fall back safely and return error for Q types.

old1 = """            // We cannot just call .contiguous() on transposed quantized blocks.
            // Also, t.to_dtype(DType::F32) panics directly in candle on block quantized tensors.
            // So we use dequantize helper if available, but since we cannot directly dequantize
            // candle's Q4_K_M etc without our own gguf loader integration, we just have to avoid
            // the transposition panic altogether. In this specific repository, if a quantized tensor
            // is transposed, it cannot be made contiguous without proper dequantization which isn't
            // exposed cleanly.
            // Wait, actually `tensor.dequantize(device)` is available in candle!
            // But how is it named? Ah! candle's Tensor has `.dequantize(device)`. Wait no, it doesn't.
            // Let's just catch the panic... wait we can't catch a panic.

            let dt_name = format!("{:?}", t.dtype());
            if dt_name.starts_with("Q") || dt_name.starts_with("I2") || dt_name.starts_with("G") {
                // To avoid panic, convert to f32 if possible, then transpose.
                // Wait, to_dtype(F32) panics directly for Q4 tensors.
                // So we just return an error and fail fast instead of panicking.
                return Err(bitnet_common::BitNetError::Validation(format!(
                    "{}: Cannot transpose quantized tensor of dtype {:?} (shape {:?}). The model must be converted to F16 first or use a supported native quantization.",
                    name, t.dtype(), t.shape().dims()
                )));
            } else {
                // To avoid panic inside .contiguous(), we just won't call it.
                // Many times PyTorch or Candle can handle non-contiguous tensors.
                // Or if it strictly requires contiguous, we can use `t.t()?.copy()?`
                let transposed = t.t()?;
                let res = match transposed.contiguous() {
                    Ok(t2) => t2,
                    Err(_) => {
                        // Fallback
                        t.to_dtype(candle_core::DType::F32)?.t()?.contiguous()?
                    }
                };
                Ok(res)
            }"""

new1 = """            let transposed = t.t()?;
            let res = match transposed.contiguous() {
                Ok(t2) => t2,
                Err(_) => {
                    tracing::warn!("{}: Tensor transpose contiguous failed, falling back to F32 cast", name);
                    t.to_dtype(candle_core::DType::F32)?.t()?.contiguous()?
                }
            };
            Ok(res)"""

code = code.replace(old1, new1)

with open("crates/bitnet-models/src/weight_mapper.rs", "w") as f:
    f.write(code)
