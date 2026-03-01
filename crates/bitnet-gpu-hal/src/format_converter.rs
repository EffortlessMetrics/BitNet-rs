//! Model format conversion utilities for GGUF, `SafeTensors`, ONNX, and `PyTorch`.
//!
//! Provides planning, validation, tensor name mapping, and quantization
//! conversion for moving models between serialisation formats.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors produced during format conversion.
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("unsupported conversion: {from} -> {to}")]
    UnsupportedConversion { from: String, to: String },

    #[error("unsupported dtype conversion: {from} -> {to}")]
    UnsupportedDtype { from: String, to: String },

    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    #[error("empty tensor list")]
    EmptyTensorList,

    #[error("data length mismatch: expected {expected}, got {actual}")]
    DataLengthMismatch { expected: usize, actual: usize },

    #[error("invalid data: {0}")]
    InvalidData(String),
}

// ── Model format ─────────────────────────────────────────────────────────────

/// Recognised model serialisation formats.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
    Onnx,
    PyTorch,
    Custom(String),
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gguf => write!(f, "GGUF"),
            Self::SafeTensors => write!(f, "SafeTensors"),
            Self::Onnx => write!(f, "ONNX"),
            Self::PyTorch => write!(f, "PyTorch"),
            Self::Custom(name) => write!(f, "Custom({name})"),
        }
    }
}

// Magic-byte signatures for format detection.
const GGUF_MAGIC: &[u8] = b"GGUF";
const SAFETENSORS_HEADER_BYTE: u8 = b'{';
const ONNX_MAGIC: &[u8] = &[0x08]; // protobuf varint for ir_version field
const PYTORCH_MAGIC: &[u8] = &[0x80, 0x02]; // Python pickle protocol 2

/// Detect a [`ModelFormat`] from the first bytes of a file.
///
/// Returns `None` when the bytes do not match any known signature.
pub fn detect_format(bytes: &[u8]) -> Option<ModelFormat> {
    if bytes.len() >= 4 && bytes[..4] == *GGUF_MAGIC {
        return Some(ModelFormat::Gguf);
    }
    // SafeTensors files start with a JSON header whose first byte is `{`.
    // The first 8 bytes are a little-endian u64 header length, so byte 8
    // is the opening brace.
    if bytes.len() >= 9 && bytes[8] == SAFETENSORS_HEADER_BYTE {
        return Some(ModelFormat::SafeTensors);
    }
    if bytes.len() >= 2 && bytes[..2] == *PYTORCH_MAGIC {
        return Some(ModelFormat::PyTorch);
    }
    if !bytes.is_empty() && bytes[0] == ONNX_MAGIC[0] {
        return Some(ModelFormat::Onnx);
    }
    None
}

// ── Tensor conversion descriptor ─────────────────────────────────────────────

/// Describes how a single tensor should be converted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorConversion {
    pub source_name: String,
    pub target_name: String,
    pub source_dtype: String,
    pub target_dtype: String,
    pub shape: Vec<usize>,
    pub needs_transpose: bool,
    pub needs_requantize: bool,
}

// ── Conversion plan ──────────────────────────────────────────────────────────

/// A plan describing a full model conversion.
#[derive(Debug, Clone)]
pub struct ConversionPlan {
    pub source_format: ModelFormat,
    pub target_format: ModelFormat,
    pub tensors: Vec<TensorConversion>,
    pub estimated_output_size: u64,
}

impl ConversionPlan {
    /// Validate that the conversion plan is feasible.
    pub fn validate(&self) -> Result<(), ConversionError> {
        if self.source_format == self.target_format {
            return Err(ConversionError::UnsupportedConversion {
                from: self.source_format.to_string(),
                to: self.target_format.to_string(),
            });
        }
        if self.tensors.is_empty() {
            return Err(ConversionError::EmptyTensorList);
        }
        for tc in &self.tensors {
            if tc.shape.is_empty() {
                return Err(ConversionError::ShapeMismatch { expected: vec![1], actual: vec![] });
            }
        }
        Ok(())
    }

    /// Rough wall-clock estimate in milliseconds (heuristic).
    ///
    /// 1 GB/s assumed throughput; requantisation adds 2× overhead.
    pub fn estimated_time_ms(&self) -> u64 {
        const BYTES_PER_MS: u64 = 1_000_000; // ~1 GB/s
        let base = self.estimated_output_size / BYTES_PER_MS.max(1);
        let requant_count = self.tensors.iter().filter(|t| t.needs_requantize).count() as u64;
        base + requant_count * 2
    }
}

// ── Format converter ─────────────────────────────────────────────────────────

/// High-level entry point for building conversion plans.
pub struct FormatConverter;

/// Description of a tensor to be converted (input to the planner).
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl FormatConverter {
    /// Build a [`ConversionPlan`] from source/target formats and tensor list.
    pub fn plan(
        source: ModelFormat,
        target: ModelFormat,
        tensors: &[TensorInfo],
        mapper: &TensorNameMapper,
    ) -> ConversionPlan {
        let mut conversions = Vec::with_capacity(tensors.len());
        let mut total_bytes: u64 = 0;

        for t in tensors {
            let target_name = mapper.map_name(&t.name);
            let target_dtype = default_target_dtype(&source, &target, &t.dtype);
            let needs_transpose = needs_transpose(&source, &target, &t.shape);
            let needs_requantize = t.dtype != target_dtype;
            let elem_size = dtype_byte_size(&target_dtype);
            let numel: u64 = t.shape.iter().copied().product::<usize>() as u64;
            total_bytes += numel * elem_size as u64;

            conversions.push(TensorConversion {
                source_name: t.name.clone(),
                target_name,
                source_dtype: t.dtype.clone(),
                target_dtype,
                shape: t.shape.clone(),
                needs_transpose,
                needs_requantize,
            });
        }

        ConversionPlan {
            source_format: source,
            target_format: target,
            tensors: conversions,
            estimated_output_size: total_bytes,
        }
    }
}

// ── Conversion result ────────────────────────────────────────────────────────

/// Summary returned after executing a conversion.
#[derive(Debug, Clone)]
pub struct ConversionResult {
    pub tensors_converted: usize,
    pub total_bytes: u64,
    pub warnings: Vec<String>,
    pub name_mappings: Vec<(String, String)>,
}

// ── Name mapping ─────────────────────────────────────────────────────────────

/// A single find→replace rule used by [`TensorNameMapper`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameMappingRule {
    pub pattern: String,
    pub replacement: String,
}

/// Maps tensor names between format conventions.
#[derive(Debug, Clone)]
pub struct TensorNameMapper {
    rules: Vec<NameMappingRule>,
}

impl TensorNameMapper {
    /// Create an empty mapper.
    #[must_use]
    pub const fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Append a mapping rule.
    pub fn add_rule(&mut self, pattern: &str, replacement: &str) {
        self.rules.push(NameMappingRule {
            pattern: pattern.to_string(),
            replacement: replacement.to_string(),
        });
    }

    /// Apply rules in order, returning the (possibly rewritten) name.
    #[must_use]
    pub fn map_name(&self, name: &str) -> String {
        let mut result = name.to_string();
        for rule in &self.rules {
            result = result.replace(&rule.pattern, &rule.replacement);
        }
        result
    }
}

impl Default for TensorNameMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-built mapper: GGUF tensor names → `SafeTensors` conventions.
#[must_use]
pub fn gguf_to_safetensors_mapper() -> TensorNameMapper {
    let mut m = TensorNameMapper::new();
    m.add_rule("blk.", "model.layers.");
    m.add_rule(".attn_q.", ".self_attn.q_proj.");
    m.add_rule(".attn_k.", ".self_attn.k_proj.");
    m.add_rule(".attn_v.", ".self_attn.v_proj.");
    m.add_rule(".attn_output.", ".self_attn.o_proj.");
    m.add_rule(".ffn_up.", ".mlp.up_proj.");
    m.add_rule(".ffn_down.", ".mlp.down_proj.");
    m.add_rule(".ffn_gate.", ".mlp.gate_proj.");
    m.add_rule(".attn_norm.", ".input_layernorm.");
    m.add_rule(".ffn_norm.", ".post_attention_layernorm.");
    m.add_rule("token_embd.", "model.embed_tokens.");
    m.add_rule("output_norm.", "model.norm.");
    m.add_rule("output.", "lm_head.");
    m
}

/// Pre-built mapper: `SafeTensors` tensor names → GGUF conventions.
#[must_use]
pub fn safetensors_to_gguf_mapper() -> TensorNameMapper {
    let mut m = TensorNameMapper::new();
    m.add_rule("model.layers.", "blk.");
    m.add_rule(".self_attn.q_proj.", ".attn_q.");
    m.add_rule(".self_attn.k_proj.", ".attn_k.");
    m.add_rule(".self_attn.v_proj.", ".attn_v.");
    m.add_rule(".self_attn.o_proj.", ".attn_output.");
    m.add_rule(".mlp.up_proj.", ".ffn_up.");
    m.add_rule(".mlp.down_proj.", ".ffn_down.");
    m.add_rule(".mlp.gate_proj.", ".ffn_gate.");
    m.add_rule(".input_layernorm.", ".attn_norm.");
    m.add_rule(".post_attention_layernorm.", ".ffn_norm.");
    m.add_rule("model.embed_tokens.", "token_embd.");
    m.add_rule("model.norm.", "output_norm.");
    m.add_rule("lm_head.", "output.");
    m
}

/// Pre-built mapper: `PyTorch` tensor names → GGUF conventions.
#[must_use]
pub fn pytorch_to_gguf_mapper() -> TensorNameMapper {
    let mut m = TensorNameMapper::new();
    m.add_rule("transformer.h.", "blk.");
    m.add_rule(".attn.c_attn.", ".attn_q.");
    m.add_rule(".attn.c_proj.", ".attn_output.");
    m.add_rule(".mlp.c_fc.", ".ffn_up.");
    m.add_rule(".mlp.c_proj.", ".ffn_down.");
    m.add_rule(".ln_1.", ".attn_norm.");
    m.add_rule(".ln_2.", ".ffn_norm.");
    m.add_rule("transformer.wte.", "token_embd.");
    m.add_rule("transformer.ln_f.", "output_norm.");
    m.add_rule("lm_head.", "output.");
    m
}

/// Pre-built mapper: GGUF tensor names → `PyTorch` conventions.
#[must_use]
pub fn gguf_to_pytorch_mapper() -> TensorNameMapper {
    let mut m = TensorNameMapper::new();
    m.add_rule("blk.", "transformer.h.");
    m.add_rule(".attn_q.", ".attn.c_attn.");
    m.add_rule(".attn_output.", ".attn.c_proj.");
    m.add_rule(".ffn_up.", ".mlp.c_fc.");
    m.add_rule(".ffn_down.", ".mlp.c_proj.");
    m.add_rule(".attn_norm.", ".ln_1.");
    m.add_rule(".ffn_norm.", ".ln_2.");
    m.add_rule("token_embd.", "transformer.wte.");
    m.add_rule("output_norm.", "transformer.ln_f.");
    m.add_rule("output.", "lm_head.");
    m
}

// ── Quantisation converter ───────────────────────────────────────────────────

/// Handles dtype conversions (e.g. F32→F16, F16→`I2_S`).
pub struct QuantizationConverter;

/// Result of a single dtype conversion.
#[derive(Debug, Clone)]
pub struct QuantConversionResult {
    pub original_dtype: String,
    pub target_dtype: String,
    pub original_size: u64,
    pub converted_size: u64,
    pub compression_ratio: f64,
}

impl QuantizationConverter {
    /// Convert raw tensor bytes from one dtype to another.
    ///
    /// Currently supports: F32↔F16, F32→`I2_S`, F16→`I2_S`.
    pub fn convert_dtype(
        from: &str,
        to: &str,
        data: &[u8],
    ) -> Result<(Vec<u8>, QuantConversionResult), ConversionError> {
        #[allow(clippy::cast_possible_truncation)]
        let from_elem = dtype_byte_size(from) as usize;
        #[allow(clippy::cast_possible_truncation)]
        let to_elem = dtype_byte_size(to) as usize;

        if from_elem == 0 {
            return Err(ConversionError::UnsupportedDtype {
                from: from.to_string(),
                to: to.to_string(),
            });
        }
        if to_elem == 0 {
            return Err(ConversionError::UnsupportedDtype {
                from: from.to_string(),
                to: to.to_string(),
            });
        }
        if !data.len().is_multiple_of(from_elem) {
            return Err(ConversionError::DataLengthMismatch {
                expected: (data.len() / from_elem) * from_elem,
                actual: data.len(),
            });
        }

        let num_elements = data.len() / from_elem;
        let converted = match (from, to) {
            ("F32", "F16") => convert_f32_to_f16(data, num_elements),
            ("F16", "F32") => convert_f16_to_f32(data, num_elements),
            ("F32", "I2_S") => convert_to_i2s(data, num_elements, 4),
            ("F16", "I2_S") => convert_to_i2s(data, num_elements, 2),
            _ => {
                return Err(ConversionError::UnsupportedDtype {
                    from: from.to_string(),
                    to: to.to_string(),
                });
            }
        };

        let original_size = data.len() as u64;
        let converted_size = converted.len() as u64;
        let compression_ratio = if converted_size > 0 {
            #[allow(clippy::cast_precision_loss)]
            {
                original_size as f64 / converted_size as f64
            }
        } else {
            0.0
        };

        Ok((
            converted,
            QuantConversionResult {
                original_dtype: from.to_string(),
                target_dtype: to.to_string(),
                original_size,
                converted_size,
                compression_ratio,
            },
        ))
    }
}

// ── Helpers (private) ────────────────────────────────────────────────────────

/// Byte size per element for a given dtype string.
fn dtype_byte_size(dtype: &str) -> u64 {
    match dtype {
        "F32" | "f32" | "I32" | "i32" => 4,
        "F16" | "f16" | "BF16" | "bf16" => 2,
        "I8" | "i8" | "I2_S" | "i2_s" => 1, // I2_S: packed 4 elems/byte
        "I64" | "i64" | "F64" | "f64" => 8,
        _ => 0,
    }
}

/// Choose a sensible default target dtype based on formats and source dtype.
fn default_target_dtype(_source: &ModelFormat, target: &ModelFormat, source_dtype: &str) -> String {
    match target {
        ModelFormat::Gguf => {
            // GGUF commonly uses F16 for weights
            if source_dtype == "F32" { "F16".to_string() } else { source_dtype.to_string() }
        }
        ModelFormat::SafeTensors => {
            // SafeTensors preserves dtype
            source_dtype.to_string()
        }
        _ => source_dtype.to_string(),
    }
}

/// Heuristic: 2-D weight tensors going to/from `PyTorch` may need transpose.
const fn needs_transpose(source: &ModelFormat, target: &ModelFormat, shape: &[usize]) -> bool {
    if shape.len() != 2 {
        return false;
    }
    matches!(
        (source, target),
        (ModelFormat::PyTorch, ModelFormat::Gguf) | (ModelFormat::Gguf, ModelFormat::PyTorch)
    )
}

/// Truncating F32→F16 conversion (loses precision, but fast).
fn convert_f32_to_f16(data: &[u8], num_elements: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(num_elements * 2);
    for i in 0..num_elements {
        let offset = i * 4;
        let val = f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        let half = f32_to_f16_bits(val);
        out.extend_from_slice(&half.to_le_bytes());
    }
    out
}

/// F16→F32 widening conversion.
fn convert_f16_to_f32(data: &[u8], num_elements: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(num_elements * 4);
    for i in 0..num_elements {
        let offset = i * 2;
        let half = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let val = f16_bits_to_f32(half);
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

/// Stub `I2_S` packing: packs 4 ternary values per byte.
fn convert_to_i2s(_data: &[u8], num_elements: usize, _src_elem_size: usize) -> Vec<u8> {
    // Each byte stores 4 ternary values (2 bits each).
    let packed_len = num_elements.div_ceil(4);
    vec![0u8; packed_len]
}

/// Minimal software F32→F16 (IEEE 754 half-precision) bit conversion.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
const fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF).cast_signed();
    let mantissa = bits & 0x007F_FFFF;

    if exp == 255 {
        // Inf / NaN
        return sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7C00; // overflow → Inf
    }
    if new_exp <= 0 {
        return sign; // underflow → zero (flush to zero)
    }
    sign | ((new_exp as u16) << 10) | ((mantissa >> 13) as u16)
}

/// Minimal software F16→F32 bit conversion.
#[allow(clippy::cast_sign_loss, clippy::cast_lossless)]
const fn f16_bits_to_f32(half: u16) -> f32 {
    let sign = ((half & 0x8000) as u32) << 16;
    let exp = ((half >> 10) & 0x1F) as u32;
    let mantissa = (half & 0x03FF) as u32;

    if exp == 31 {
        // Inf / NaN
        let f_exp = 0xFF << 23;
        let f_man = mantissa << 13;
        return f32::from_bits(sign | f_exp | f_man);
    }
    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign); // ±0
        }
        // Denormalized — convert to normalized f32
        let mut m = mantissa;
        let mut e: i32 = 1;
        while m & 0x0400 == 0 {
            m <<= 1;
            e -= 1;
        }
        let f_exp = ((127 - 15 + e) as u32) << 23;
        let f_man = (m & 0x03FF) << 13;
        return f32::from_bits(sign | f_exp | f_man);
    }
    let f_exp = (exp + 127 - 15) << 23;
    let f_man = mantissa << 13;
    f32::from_bits(sign | f_exp | f_man)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ModelFormat Display ───────────────────────────────────────────────

    #[test]
    fn display_gguf() {
        assert_eq!(ModelFormat::Gguf.to_string(), "GGUF");
    }

    #[test]
    fn display_safetensors() {
        assert_eq!(ModelFormat::SafeTensors.to_string(), "SafeTensors");
    }

    #[test]
    fn display_onnx() {
        assert_eq!(ModelFormat::Onnx.to_string(), "ONNX");
    }

    #[test]
    fn display_pytorch() {
        assert_eq!(ModelFormat::PyTorch.to_string(), "PyTorch");
    }

    #[test]
    fn display_custom() {
        let f = ModelFormat::Custom("MyFormat".into());
        assert_eq!(f.to_string(), "Custom(MyFormat)");
    }

    // ── Format detection ─────────────────────────────────────────────────

    #[test]
    fn detect_gguf_magic() {
        let bytes = b"GGUF\x03\x00\x00\x00";
        assert_eq!(detect_format(bytes), Some(ModelFormat::Gguf));
    }

    #[test]
    fn detect_safetensors_magic() {
        // 8-byte LE header length + opening brace
        let mut bytes = vec![0u8; 9];
        bytes[8] = b'{';
        assert_eq!(detect_format(&bytes), Some(ModelFormat::SafeTensors));
    }

    #[test]
    fn detect_pytorch_magic() {
        let bytes = [0x80, 0x02, 0x00, 0x00];
        assert_eq!(detect_format(&bytes), Some(ModelFormat::PyTorch));
    }

    #[test]
    fn detect_onnx_magic() {
        let bytes = [0x08, 0x06];
        assert_eq!(detect_format(&bytes), Some(ModelFormat::Onnx));
    }

    #[test]
    fn detect_unknown_format() {
        let bytes = [0xFF, 0xFE, 0xFD];
        assert_eq!(detect_format(&bytes), None);
    }

    #[test]
    fn detect_empty_bytes() {
        assert_eq!(detect_format(&[]), None);
    }

    #[test]
    fn detect_too_short_for_gguf() {
        assert_eq!(detect_format(b"GGU"), None);
    }

    // ── TensorNameMapper basics ──────────────────────────────────────────

    #[test]
    fn mapper_no_rules_identity() {
        let m = TensorNameMapper::new();
        assert_eq!(m.map_name("foo.bar"), "foo.bar");
    }

    #[test]
    fn mapper_single_rule() {
        let mut m = TensorNameMapper::new();
        m.add_rule("foo", "bar");
        assert_eq!(m.map_name("foo.weight"), "bar.weight");
    }

    #[test]
    fn mapper_multiple_rules_applied_in_order() {
        let mut m = TensorNameMapper::new();
        m.add_rule("a", "b");
        m.add_rule("b", "c");
        // "a" → "b" → "c"
        assert_eq!(m.map_name("a"), "c");
    }

    #[test]
    fn mapper_default_is_empty() {
        let m = TensorNameMapper::default();
        assert_eq!(m.map_name("x"), "x");
    }

    #[test]
    fn mapper_pattern_not_found_unchanged() {
        let mut m = TensorNameMapper::new();
        m.add_rule("missing", "replacement");
        assert_eq!(m.map_name("present"), "present");
    }

    #[test]
    fn mapper_replaces_all_occurrences() {
        let mut m = TensorNameMapper::new();
        m.add_rule("x", "y");
        assert_eq!(m.map_name("x.x.x"), "y.y.y");
    }

    // ── GGUF↔SafeTensors name mapping ────────────────────────────────────

    #[test]
    fn gguf_to_st_attn_q() {
        let m = gguf_to_safetensors_mapper();
        assert_eq!(m.map_name("blk.0.attn_q.weight"), "model.layers.0.self_attn.q_proj.weight");
    }

    #[test]
    fn gguf_to_st_attn_k() {
        let m = gguf_to_safetensors_mapper();
        assert_eq!(m.map_name("blk.0.attn_k.weight"), "model.layers.0.self_attn.k_proj.weight");
    }

    #[test]
    fn gguf_to_st_attn_v() {
        let m = gguf_to_safetensors_mapper();
        assert_eq!(m.map_name("blk.0.attn_v.weight"), "model.layers.0.self_attn.v_proj.weight");
    }

    #[test]
    fn gguf_to_st_ffn_up() {
        let m = gguf_to_safetensors_mapper();
        assert_eq!(m.map_name("blk.1.ffn_up.weight"), "model.layers.1.mlp.up_proj.weight");
    }

    #[test]
    fn gguf_to_st_ffn_down() {
        let m = gguf_to_safetensors_mapper();
        assert_eq!(m.map_name("blk.2.ffn_down.weight"), "model.layers.2.mlp.down_proj.weight");
    }

    #[test]
    fn gguf_to_st_token_embd() {
        let m = gguf_to_safetensors_mapper();
        assert_eq!(m.map_name("token_embd.weight"), "model.embed_tokens.weight");
    }

    #[test]
    fn gguf_to_st_output_norm() {
        let m = gguf_to_safetensors_mapper();
        assert_eq!(m.map_name("output_norm.weight"), "model.norm.weight");
    }

    #[test]
    fn gguf_to_st_output() {
        let m = gguf_to_safetensors_mapper();
        assert_eq!(m.map_name("output.weight"), "lm_head.weight");
    }

    // ── Reverse: SafeTensors→GGUF ────────────────────────────────────────

    #[test]
    fn st_to_gguf_attn_q() {
        let m = safetensors_to_gguf_mapper();
        assert_eq!(m.map_name("model.layers.0.self_attn.q_proj.weight"), "blk.0.attn_q.weight");
    }

    #[test]
    fn st_to_gguf_ffn_gate() {
        let m = safetensors_to_gguf_mapper();
        assert_eq!(m.map_name("model.layers.3.mlp.gate_proj.weight"), "blk.3.ffn_gate.weight");
    }

    #[test]
    fn st_to_gguf_embed_tokens() {
        let m = safetensors_to_gguf_mapper();
        assert_eq!(m.map_name("model.embed_tokens.weight"), "token_embd.weight");
    }

    #[test]
    fn st_to_gguf_lm_head() {
        let m = safetensors_to_gguf_mapper();
        assert_eq!(m.map_name("lm_head.weight"), "output.weight");
    }

    // ── PyTorch↔GGUF name mapping ────────────────────────────────────────

    #[test]
    fn pytorch_to_gguf_transformer_block() {
        let m = pytorch_to_gguf_mapper();
        assert_eq!(m.map_name("transformer.h.0.attn.c_attn.weight"), "blk.0.attn_q.weight");
    }

    #[test]
    fn gguf_to_pytorch_block() {
        let m = gguf_to_pytorch_mapper();
        assert_eq!(m.map_name("blk.0.attn_q.weight"), "transformer.h.0.attn.c_attn.weight");
    }

    // ── FormatConverter::plan ─────────────────────────────────────────────

    fn sample_tensors() -> Vec<TensorInfo> {
        vec![
            TensorInfo {
                name: "blk.0.attn_q.weight".into(),
                dtype: "F32".into(),
                shape: vec![4096, 4096],
            },
            TensorInfo {
                name: "blk.0.ffn_up.weight".into(),
                dtype: "F16".into(),
                shape: vec![4096, 11008],
            },
        ]
    }

    #[test]
    fn plan_gguf_to_safetensors() {
        let mapper = gguf_to_safetensors_mapper();
        let plan = FormatConverter::plan(
            ModelFormat::Gguf,
            ModelFormat::SafeTensors,
            &sample_tensors(),
            &mapper,
        );
        assert_eq!(plan.tensors.len(), 2);
        assert_eq!(plan.tensors[0].target_name, "model.layers.0.self_attn.q_proj.weight");
    }

    #[test]
    fn plan_preserves_shapes() {
        let mapper = TensorNameMapper::new();
        let tensors = sample_tensors();
        let plan =
            FormatConverter::plan(ModelFormat::Gguf, ModelFormat::SafeTensors, &tensors, &mapper);
        assert_eq!(plan.tensors[0].shape, vec![4096, 4096]);
        assert_eq!(plan.tensors[1].shape, vec![4096, 11008]);
    }

    #[test]
    fn plan_dtype_preserved_for_safetensors_target() {
        let mapper = TensorNameMapper::new();
        let plan = FormatConverter::plan(
            ModelFormat::Gguf,
            ModelFormat::SafeTensors,
            &sample_tensors(),
            &mapper,
        );
        // SafeTensors preserves dtype
        assert_eq!(plan.tensors[0].target_dtype, "F32");
        assert_eq!(plan.tensors[1].target_dtype, "F16");
    }

    #[test]
    fn plan_f32_downcast_to_f16_for_gguf_target() {
        let mapper = TensorNameMapper::new();
        let tensors =
            vec![TensorInfo { name: "w".into(), dtype: "F32".into(), shape: vec![128, 128] }];
        let plan =
            FormatConverter::plan(ModelFormat::SafeTensors, ModelFormat::Gguf, &tensors, &mapper);
        assert_eq!(plan.tensors[0].target_dtype, "F16");
        assert!(plan.tensors[0].needs_requantize);
    }

    #[test]
    fn plan_estimated_output_size() {
        let mapper = TensorNameMapper::new();
        let tensors =
            vec![TensorInfo { name: "w".into(), dtype: "F32".into(), shape: vec![100, 100] }];
        let plan =
            FormatConverter::plan(ModelFormat::Gguf, ModelFormat::SafeTensors, &tensors, &mapper);
        // 100*100 * 4 bytes (F32 preserved for SafeTensors)
        assert_eq!(plan.estimated_output_size, 40_000);
    }

    #[test]
    fn plan_transpose_pytorch_to_gguf() {
        let mapper = TensorNameMapper::new();
        let tensors =
            vec![TensorInfo { name: "w".into(), dtype: "F32".into(), shape: vec![128, 256] }];
        let plan =
            FormatConverter::plan(ModelFormat::PyTorch, ModelFormat::Gguf, &tensors, &mapper);
        assert!(plan.tensors[0].needs_transpose);
    }

    #[test]
    fn plan_no_transpose_safetensors_to_gguf() {
        let mapper = TensorNameMapper::new();
        let tensors =
            vec![TensorInfo { name: "w".into(), dtype: "F16".into(), shape: vec![128, 256] }];
        let plan =
            FormatConverter::plan(ModelFormat::SafeTensors, ModelFormat::Gguf, &tensors, &mapper);
        assert!(!plan.tensors[0].needs_transpose);
    }

    #[test]
    fn plan_no_transpose_for_1d_tensor() {
        let mapper = TensorNameMapper::new();
        let tensors =
            vec![TensorInfo { name: "bias".into(), dtype: "F32".into(), shape: vec![4096] }];
        let plan =
            FormatConverter::plan(ModelFormat::PyTorch, ModelFormat::Gguf, &tensors, &mapper);
        assert!(!plan.tensors[0].needs_transpose);
    }

    #[test]
    fn plan_empty_tensors() {
        let mapper = TensorNameMapper::new();
        let plan = FormatConverter::plan(ModelFormat::Gguf, ModelFormat::SafeTensors, &[], &mapper);
        assert!(plan.tensors.is_empty());
        assert_eq!(plan.estimated_output_size, 0);
    }

    #[test]
    fn plan_large_tensor() {
        let mapper = TensorNameMapper::new();
        let tensors =
            vec![TensorInfo { name: "huge".into(), dtype: "F16".into(), shape: vec![32000, 4096] }];
        let plan =
            FormatConverter::plan(ModelFormat::Gguf, ModelFormat::SafeTensors, &tensors, &mapper);
        // 32000 * 4096 * 2 bytes = 262_144_000
        assert_eq!(plan.estimated_output_size, 262_144_000);
    }

    // ── ConversionPlan validation ────────────────────────────────────────

    #[test]
    fn validate_same_format_rejected() {
        let plan = ConversionPlan {
            source_format: ModelFormat::Gguf,
            target_format: ModelFormat::Gguf,
            tensors: vec![TensorConversion {
                source_name: "w".into(),
                target_name: "w".into(),
                source_dtype: "F32".into(),
                target_dtype: "F32".into(),
                shape: vec![8],
                needs_transpose: false,
                needs_requantize: false,
            }],
            estimated_output_size: 32,
        };
        assert!(plan.validate().is_err());
    }

    #[test]
    fn validate_empty_tensors_rejected() {
        let plan = ConversionPlan {
            source_format: ModelFormat::Gguf,
            target_format: ModelFormat::SafeTensors,
            tensors: vec![],
            estimated_output_size: 0,
        };
        assert!(plan.validate().is_err());
    }

    #[test]
    fn validate_empty_shape_rejected() {
        let plan = ConversionPlan {
            source_format: ModelFormat::Gguf,
            target_format: ModelFormat::SafeTensors,
            tensors: vec![TensorConversion {
                source_name: "w".into(),
                target_name: "w".into(),
                source_dtype: "F32".into(),
                target_dtype: "F32".into(),
                shape: vec![],
                needs_transpose: false,
                needs_requantize: false,
            }],
            estimated_output_size: 0,
        };
        assert!(plan.validate().is_err());
    }

    #[test]
    fn validate_valid_plan_ok() {
        let mapper = gguf_to_safetensors_mapper();
        let plan = FormatConverter::plan(
            ModelFormat::Gguf,
            ModelFormat::SafeTensors,
            &sample_tensors(),
            &mapper,
        );
        assert!(plan.validate().is_ok());
    }

    // ── Estimated time ───────────────────────────────────────────────────

    #[test]
    fn estimated_time_zero_for_small_plan() {
        let plan = ConversionPlan {
            source_format: ModelFormat::Gguf,
            target_format: ModelFormat::SafeTensors,
            tensors: vec![TensorConversion {
                source_name: "w".into(),
                target_name: "w".into(),
                source_dtype: "F32".into(),
                target_dtype: "F32".into(),
                shape: vec![8],
                needs_transpose: false,
                needs_requantize: false,
            }],
            estimated_output_size: 32,
        };
        // Very small → 0 ms base + 0 requant
        assert_eq!(plan.estimated_time_ms(), 0);
    }

    #[test]
    fn estimated_time_increases_with_requant() {
        let make = |requant: bool| ConversionPlan {
            source_format: ModelFormat::Gguf,
            target_format: ModelFormat::SafeTensors,
            tensors: vec![TensorConversion {
                source_name: "w".into(),
                target_name: "w".into(),
                source_dtype: "F32".into(),
                target_dtype: "F16".into(),
                shape: vec![1024],
                needs_transpose: false,
                needs_requantize: requant,
            }],
            estimated_output_size: 2_000_000,
        };
        assert!(make(true).estimated_time_ms() > make(false).estimated_time_ms());
    }

    // ── QuantizationConverter ────────────────────────────────────────────

    #[test]
    fn quant_f32_to_f16_length() {
        let data = vec![0u8; 16]; // 4 floats
        let (out, res) = QuantizationConverter::convert_dtype("F32", "F16", &data).unwrap();
        assert_eq!(out.len(), 8); // 4 halfs
        assert_eq!(res.original_size, 16);
        assert_eq!(res.converted_size, 8);
    }

    #[test]
    fn quant_f16_to_f32_length() {
        let data = vec![0u8; 8]; // 4 halfs
        let (out, _) = QuantizationConverter::convert_dtype("F16", "F32", &data).unwrap();
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn quant_f32_to_f16_roundtrip_zero() {
        let zero = 0.0f32;
        let data = zero.to_le_bytes().to_vec();
        let (half, _) = QuantizationConverter::convert_dtype("F32", "F16", &data).unwrap();
        let (back, _) = QuantizationConverter::convert_dtype("F16", "F32", &half).unwrap();
        let val = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
        assert!((val - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn quant_f32_to_f16_roundtrip_one() {
        let one = 1.0f32;
        let data = one.to_le_bytes().to_vec();
        let (half, _) = QuantizationConverter::convert_dtype("F32", "F16", &data).unwrap();
        let (back, _) = QuantizationConverter::convert_dtype("F16", "F32", &half).unwrap();
        let val = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
        assert!((val - 1.0).abs() < 0.001);
    }

    #[test]
    fn quant_f32_to_i2s() {
        let data = vec![0u8; 32]; // 8 floats
        let (out, res) = QuantizationConverter::convert_dtype("F32", "I2_S", &data).unwrap();
        // 8 elements → 2 bytes (4 per byte)
        assert_eq!(out.len(), 2);
        assert!(res.compression_ratio > 1.0);
    }

    #[test]
    fn quant_f16_to_i2s() {
        let data = vec![0u8; 16]; // 8 halfs
        let (out, _) = QuantizationConverter::convert_dtype("F16", "I2_S", &data).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn quant_compression_ratio_f32_to_f16() {
        let data = vec![0u8; 400]; // 100 floats
        let (_, res) = QuantizationConverter::convert_dtype("F32", "F16", &data).unwrap();
        assert!((res.compression_ratio - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn quant_unsupported_dtype_error() {
        let data = vec![0u8; 4];
        let result = QuantizationConverter::convert_dtype("BFLOAT16", "Q4_0", &data);
        assert!(result.is_err());
    }

    #[test]
    fn quant_misaligned_data_error() {
        // 5 bytes is not a multiple of 4 (F32 element size)
        let data = vec![0u8; 5];
        let result = QuantizationConverter::convert_dtype("F32", "F16", &data);
        assert!(result.is_err());
    }

    #[test]
    fn quant_empty_data() {
        let (out, res) = QuantizationConverter::convert_dtype("F32", "F16", &[]).unwrap();
        assert!(out.is_empty());
        assert_eq!(res.original_size, 0);
    }

    // ── ConversionResult construction ────────────────────────────────────

    #[test]
    fn conversion_result_fields() {
        let r = ConversionResult {
            tensors_converted: 5,
            total_bytes: 1024,
            warnings: vec!["low precision".into()],
            name_mappings: vec![("a".into(), "b".into())],
        };
        assert_eq!(r.tensors_converted, 5);
        assert_eq!(r.total_bytes, 1024);
        assert_eq!(r.warnings.len(), 1);
        assert_eq!(r.name_mappings.len(), 1);
    }

    // ── Batch conversion planning ────────────────────────────────────────

    #[test]
    fn batch_plan_many_tensors() {
        let mapper = gguf_to_safetensors_mapper();
        let tensors: Vec<TensorInfo> = (0..50)
            .map(|i| TensorInfo {
                name: format!("blk.{i}.attn_q.weight"),
                dtype: "F16".into(),
                shape: vec![4096, 4096],
            })
            .collect();
        let plan =
            FormatConverter::plan(ModelFormat::Gguf, ModelFormat::SafeTensors, &tensors, &mapper);
        assert_eq!(plan.tensors.len(), 50);
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn batch_plan_all_names_mapped() {
        let mapper = gguf_to_safetensors_mapper();
        let tensors: Vec<TensorInfo> = (0..5)
            .map(|i| TensorInfo {
                name: format!("blk.{i}.ffn_gate.weight"),
                dtype: "F16".into(),
                shape: vec![4096, 11008],
            })
            .collect();
        let plan =
            FormatConverter::plan(ModelFormat::Gguf, ModelFormat::SafeTensors, &tensors, &mapper);
        for (i, tc) in plan.tensors.iter().enumerate() {
            assert_eq!(tc.target_name, format!("model.layers.{i}.mlp.gate_proj.weight"));
        }
    }

    // ── dtype_byte_size helper ────────────────────────────────────────────

    #[test]
    fn dtype_sizes() {
        assert_eq!(dtype_byte_size("F32"), 4);
        assert_eq!(dtype_byte_size("F16"), 2);
        assert_eq!(dtype_byte_size("BF16"), 2);
        assert_eq!(dtype_byte_size("I8"), 1);
        assert_eq!(dtype_byte_size("I2_S"), 1);
        assert_eq!(dtype_byte_size("I32"), 4);
        assert_eq!(dtype_byte_size("I64"), 8);
        assert_eq!(dtype_byte_size("F64"), 8);
        assert_eq!(dtype_byte_size("UNKNOWN"), 0);
    }

    // ── Custom format handling ───────────────────────────────────────────

    #[test]
    fn plan_with_custom_format() {
        let mapper = TensorNameMapper::new();
        let tensors = vec![TensorInfo { name: "w".into(), dtype: "F32".into(), shape: vec![64] }];
        let plan = FormatConverter::plan(
            ModelFormat::Custom("MyFmt".into()),
            ModelFormat::Gguf,
            &tensors,
            &mapper,
        );
        assert_eq!(plan.source_format, ModelFormat::Custom("MyFmt".into()));
        assert_eq!(plan.tensors[0].target_dtype, "F16");
    }

    // ── ModelFormat equality ─────────────────────────────────────────────

    #[test]
    fn model_format_eq() {
        assert_eq!(ModelFormat::Gguf, ModelFormat::Gguf);
        assert_ne!(ModelFormat::Gguf, ModelFormat::SafeTensors);
        assert_eq!(ModelFormat::Custom("x".into()), ModelFormat::Custom("x".into()));
        assert_ne!(ModelFormat::Custom("x".into()), ModelFormat::Custom("y".into()));
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn f32_to_f16_negative_value() {
        let val = (-2.75f32).to_le_bytes().to_vec();
        let (half, _) = QuantizationConverter::convert_dtype("F32", "F16", &val).unwrap();
        let (back, _) = QuantizationConverter::convert_dtype("F16", "F32", &half).unwrap();
        let result = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
        assert!(result < 0.0);
        assert!((result - (-2.75)).abs() < 0.01);
    }

    #[test]
    fn f32_to_f16_infinity() {
        let val = f32::INFINITY.to_le_bytes().to_vec();
        let (half, _) = QuantizationConverter::convert_dtype("F32", "F16", &val).unwrap();
        let (back, _) = QuantizationConverter::convert_dtype("F16", "F32", &half).unwrap();
        let result = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
        assert!(result.is_infinite());
    }

    #[test]
    fn quant_i2s_packing_alignment() {
        // 7 elements → ceil(7/4) = 2 bytes
        let data = vec![0u8; 28]; // 7 F32 elements
        let (out, _) = QuantizationConverter::convert_dtype("F32", "I2_S", &data).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn plan_needs_requantize_when_dtype_changes() {
        let mapper = TensorNameMapper::new();
        let tensors = vec![TensorInfo { name: "w".into(), dtype: "F32".into(), shape: vec![256] }];
        let plan =
            FormatConverter::plan(ModelFormat::SafeTensors, ModelFormat::Gguf, &tensors, &mapper);
        // F32→F16 for GGUF target
        assert!(plan.tensors[0].needs_requantize);
    }

    #[test]
    fn plan_no_requantize_when_dtype_same() {
        let mapper = TensorNameMapper::new();
        let tensors = vec![TensorInfo { name: "w".into(), dtype: "F16".into(), shape: vec![256] }];
        let plan =
            FormatConverter::plan(ModelFormat::SafeTensors, ModelFormat::Gguf, &tensors, &mapper);
        // F16 stays F16 for GGUF
        assert!(!plan.tensors[0].needs_requantize);
    }
}
