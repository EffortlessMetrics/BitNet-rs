//! GGUF model weight loading for OpenCL.
//!
//! Parses GGUF headers, extracts tensor metadata, and manages weight transfer
//! to GPU buffers with format conversion for Intel Arc A770. Provides CPU
//! reference implementations — the actual OpenCL buffer operations are layered
//! on top by the runtime backend.
//!
//! # Supported conversions
//!
//! | Source format | OpenCL target | Notes |
//! |---------------|---------------|-------|
//! | F32 | F32 | Pass-through |
//! | F32 | F16 | Truncate mantissa |
//! | F16 | F16 | Pass-through |
//! | F16 | F32 | Widen |
//! | I2_S (packed) | I2_S packed | Pass-through |
//! | QK256 | F32 (dequant) | Block-wise dequantize |

use std::fmt;

// ---------------------------------------------------------------------------
// GgufFieldType
// ---------------------------------------------------------------------------

/// GGUF metadata field types (§4.1 of the GGUF spec).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgufFieldType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    F16,
    Bool,
    String,
    Array,
}

impl GgufFieldType {
    /// Parse from the on-disk u32 tag.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::U8),
            1 => Some(Self::I8),
            2 => Some(Self::U16),
            3 => Some(Self::I16),
            4 => Some(Self::U32),
            5 => Some(Self::I32),
            6 => Some(Self::F32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::F16),
            _ => None,
        }
    }

    /// Convert back to the on-disk u32 tag.
    pub fn to_u32(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 3,
            Self::U32 => 4,
            Self::I32 => 5,
            Self::F32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Array => 9,
            Self::F16 => 10,
        }
    }
}

impl fmt::Display for GgufFieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// GgufHeader
// ---------------------------------------------------------------------------

/// Parsed GGUF file header (magic + version + tensor/KV counts).
///
/// This is a lightweight representation for the OpenCL loader — it does not
/// replicate the full security-hardened parser in `bitnet-models`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
}

/// GGUF magic bytes: `GGUF` in little-endian.
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// Minimum supported GGUF version.
pub const GGUF_VERSION_MIN: u32 = 2;

/// Maximum supported GGUF version.
pub const GGUF_VERSION_MAX: u32 = 3;

/// Maximum tensor count we accept (safety bound).
const MAX_TENSOR_COUNT: u64 = 100_000;

/// Minimum header size in bytes (magic + version + tensor_count + kv_count).
const HEADER_SIZE: usize = 4 + 4 + 8 + 8; // 24 bytes

impl GgufHeader {
    /// Parse a header from a raw byte slice at the given offset.
    ///
    /// Returns `(header, bytes_consumed)` on success.
    pub fn parse(data: &[u8]) -> Result<(Self, usize), GgufLoadError> {
        if data.len() < HEADER_SIZE {
            return Err(GgufLoadError::HeaderTooShort { have: data.len(), need: HEADER_SIZE });
        }

        let magic = [data[0], data[1], data[2], data[3]];
        if magic != GGUF_MAGIC {
            return Err(GgufLoadError::BadMagic(magic));
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if !(GGUF_VERSION_MIN..=GGUF_VERSION_MAX).contains(&version) {
            return Err(GgufLoadError::UnsupportedVersion(version));
        }

        let n_tensors = u64::from_le_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]);
        if n_tensors > MAX_TENSOR_COUNT {
            return Err(GgufLoadError::TensorCountExceeded {
                count: n_tensors,
                limit: MAX_TENSOR_COUNT,
            });
        }

        let n_kv = u64::from_le_bytes([
            data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
        ]);

        Ok((Self { magic, version, n_tensors, n_kv }, HEADER_SIZE))
    }

    /// Write this header into a new byte vector (little-endian).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_SIZE);
        buf.extend_from_slice(&self.magic);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.n_tensors.to_le_bytes());
        buf.extend_from_slice(&self.n_kv.to_le_bytes());
        buf
    }
}

// ---------------------------------------------------------------------------
// Tensor dtype (local, minimal)
// ---------------------------------------------------------------------------

/// Quantization / element type for tensors in the GGUF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum TensorDtype {
    F32,
    F16,
    I2_S,
    QK256,
}

impl TensorDtype {
    /// Bytes per element (for non-blocked types) or block size in bytes.
    pub fn element_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            // I2_S: 4 ternary values per byte → 0.25 B/element; report block of 4.
            Self::I2_S => 1,
            // QK256: 256-element block → 2-bit values + scale/min → ~66 bytes/block.
            Self::QK256 => 66,
        }
    }

    /// Whether this type uses blocked (sub-byte) packing.
    pub fn is_blocked(self) -> bool {
        matches!(self, Self::I2_S | Self::QK256)
    }
}

impl fmt::Display for TensorDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "F32"),
            Self::F16 => write!(f, "F16"),
            Self::I2_S => write!(f, "I2_S"),
            Self::QK256 => write!(f, "QK256"),
        }
    }
}

// ---------------------------------------------------------------------------
// GgufTensorInfo
// ---------------------------------------------------------------------------

/// Metadata for a single tensor inside a GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: TensorDtype,
    /// Byte offset from the start of the tensor data section.
    pub offset: u64,
    /// Total size in bytes for the raw tensor data.
    pub size_bytes: u64,
}

impl GgufTensorInfo {
    pub fn new(
        name: impl Into<String>,
        dims: Vec<u64>,
        dtype: TensorDtype,
        offset: u64,
        size_bytes: u64,
    ) -> Self {
        Self { name: name.into(), dims, dtype, offset, size_bytes }
    }

    /// Total number of elements across all dimensions.
    pub fn n_elements(&self) -> u64 {
        if self.dims.is_empty() {
            return 0;
        }
        self.dims.iter().copied().product()
    }
}

// ---------------------------------------------------------------------------
// TensorLayout
// ---------------------------------------------------------------------------

/// Memory layout for a tensor on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorLayout {
    RowMajor,
    ColumnMajor,
    Blocked(usize),
}

impl TensorLayout {
    /// Returns `true` if this layout is a blocked layout.
    pub fn is_blocked(self) -> bool {
        matches!(self, Self::Blocked(_))
    }

    /// Returns the block size, or 1 for non-blocked layouts.
    pub fn block_size(self) -> usize {
        match self {
            Self::Blocked(bs) => bs,
            _ => 1,
        }
    }

    /// Infer the natural layout for a given dtype.
    pub fn for_dtype(dtype: TensorDtype) -> Self {
        match dtype {
            TensorDtype::F32 | TensorDtype::F16 => Self::RowMajor,
            TensorDtype::I2_S => Self::Blocked(4),
            TensorDtype::QK256 => Self::Blocked(256),
        }
    }
}

impl fmt::Display for TensorLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowMajor => write!(f, "RowMajor"),
            Self::ColumnMajor => write!(f, "ColumnMajor"),
            Self::Blocked(bs) => write!(f, "Blocked({})", bs),
        }
    }
}

// ---------------------------------------------------------------------------
// FormatConversion
// ---------------------------------------------------------------------------

/// Describes a format conversion step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FormatConversion {
    pub src: TensorDtype,
    pub dst: TensorDtype,
}

impl FormatConversion {
    pub fn new(src: TensorDtype, dst: TensorDtype) -> Self {
        Self { src, dst }
    }

    /// Whether this is a no-op (same type).
    pub fn is_passthrough(self) -> bool {
        self.src == self.dst
    }
}

impl fmt::Display for FormatConversion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} → {}", self.src, self.dst)
    }
}

// ---------------------------------------------------------------------------
// FormatConverter — CPU reference implementations
// ---------------------------------------------------------------------------

/// CPU-side format converter for GGUF weights.
pub struct FormatConverter;

impl FormatConverter {
    /// Convert F32 → F16 (IEEE 754 half-precision, truncating).
    pub fn f32_to_f16(input: &[f32]) -> Vec<u16> {
        input.iter().map(|&v| Self::f32_bits_to_f16(v)).collect()
    }

    /// Convert F16 → F32.
    pub fn f16_to_f32(input: &[u16]) -> Vec<f32> {
        input.iter().map(|&v| Self::f16_bits_to_f32(v)).collect()
    }

    /// Pack ternary values (−1, 0, +1) into I2_S bytes (4 values per byte).
    pub fn pack_i2s(values: &[i8]) -> Vec<u8> {
        let packed_len = values.len().div_ceil(4);
        let mut packed = vec![0u8; packed_len];
        for (i, &v) in values.iter().enumerate() {
            let encoded: u8 = match v {
                -1 => 0b00,
                0 => 0b01,
                1 => 0b10,
                _ => 0b01, // treat out-of-range as zero
            };
            packed[i / 4] |= encoded << ((i % 4) * 2);
        }
        packed
    }

    /// Unpack I2_S bytes into ternary values.
    pub fn unpack_i2s(packed: &[u8], count: usize) -> Vec<i8> {
        (0..count)
            .map(|i| {
                let bits = (packed[i / 4] >> ((i % 4) * 2)) & 0x03;
                (bits as i8) - 1
            })
            .collect()
    }

    /// Dequantize a QK256 block into F32 values.
    ///
    /// QK256 block layout: `[scale: f16 (2B)] [min: f16 (2B)] [quants: 64B]`
    /// → 256 elements.
    pub fn dequantize_qk256_block(block: &[u8]) -> Vec<f32> {
        if block.len() < 68 {
            return vec![0.0f32; 256];
        }
        let scale = Self::f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = Self::f16_bits_to_f32(u16::from_le_bytes([block[2], block[3]]));

        let mut out = Vec::with_capacity(256);
        for i in 0..256 {
            let byte_idx = 4 + i / 4;
            let shift = (i % 4) * 2;
            let q = if byte_idx < block.len() {
                ((block[byte_idx] >> shift) & 0x03) as f32
            } else {
                0.0
            };
            out.push(q * scale + min);
        }
        out
    }

    // -- internal IEEE-754 helpers --

    fn f32_bits_to_f16(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x007F_FFFF;

        if exp == 0xFF {
            // Inf / NaN
            return sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 };
        }
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            return sign | 0x7C00; // overflow → Inf
        }
        if new_exp <= 0 {
            // Subnormal or zero
            if new_exp < -10 {
                return sign; // too small → 0
            }
            let shift = (14 - new_exp) as u32;
            let m = ((mantissa | 0x0080_0000) >> shift) as u16;
            return sign | m;
        }
        sign | ((new_exp as u16) << 10) | ((mantissa >> 13) as u16)
    }

    fn f16_bits_to_f32(half: u16) -> f32 {
        let sign = ((half & 0x8000) as u32) << 16;
        let exp = ((half >> 10) & 0x1F) as u32;
        let mantissa = (half & 0x03FF) as u32;

        if exp == 0 {
            if mantissa == 0 {
                return f32::from_bits(sign); // ±0
            }
            // Subnormal
            let mut m = mantissa;
            let mut e = 1u32;
            while m & 0x0400 == 0 {
                m <<= 1;
                e += 1;
            }
            let m = (m & 0x03FF) << 13;
            let e = (127 - 15 + 1 - e) << 23;
            return f32::from_bits(sign | e | m);
        }
        if exp == 31 {
            let m = if mantissa != 0 { 0x007F_FFFF } else { 0 };
            return f32::from_bits(sign | 0x7F80_0000 | m);
        }
        let e = (exp + 127 - 15) << 23;
        let m = mantissa << 13;
        f32::from_bits(sign | e | m)
    }
}

// ---------------------------------------------------------------------------
// WeightLoadPlan
// ---------------------------------------------------------------------------

/// An entry in a weight-load plan.
#[derive(Debug, Clone)]
pub struct WeightLoadEntry {
    pub tensor: GgufTensorInfo,
    pub conversion: Option<FormatConversion>,
    pub layout: TensorLayout,
    /// Destination size in bytes (after conversion).
    pub dst_size_bytes: u64,
}

/// Ordered plan for loading model weights to the GPU.
#[derive(Debug, Clone)]
pub struct WeightLoadPlan {
    entries: Vec<WeightLoadEntry>,
    total_src_bytes: u64,
    total_dst_bytes: u64,
}

impl WeightLoadPlan {
    /// Create a new empty plan.
    pub fn new() -> Self {
        Self { entries: Vec::new(), total_src_bytes: 0, total_dst_bytes: 0 }
    }

    /// Build a plan from a list of tensor infos with optional target dtype.
    pub fn build(tensors: &[GgufTensorInfo], target_dtype: Option<TensorDtype>) -> Self {
        let mut plan = Self::new();
        for t in tensors {
            let conversion = target_dtype.and_then(|dst| {
                if dst == t.dtype { None } else { Some(FormatConversion::new(t.dtype, dst)) }
            });

            let dst_size = match conversion {
                Some(c) => Self::converted_size(t, c.dst),
                None => t.size_bytes,
            };

            let layout = TensorLayout::for_dtype(conversion.map(|c| c.dst).unwrap_or(t.dtype));

            plan.total_src_bytes += t.size_bytes;
            plan.total_dst_bytes += dst_size;

            plan.entries.push(WeightLoadEntry {
                tensor: t.clone(),
                conversion,
                layout,
                dst_size_bytes: dst_size,
            });
        }
        plan
    }

    fn converted_size(tensor: &GgufTensorInfo, dst: TensorDtype) -> u64 {
        let n_elem = tensor.n_elements();
        match dst {
            TensorDtype::F32 => n_elem * 4,
            TensorDtype::F16 => n_elem * 2,
            TensorDtype::I2_S => n_elem.div_ceil(4),
            TensorDtype::QK256 => {
                // 256-element blocks, ~66 bytes per block
                let n_blocks = n_elem.div_ceil(256);
                n_blocks * 66
            }
        }
    }

    /// Number of tensors in the plan.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the plan has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over plan entries.
    pub fn entries(&self) -> &[WeightLoadEntry] {
        &self.entries
    }

    /// Total source (on-disk) bytes.
    pub fn total_src_bytes(&self) -> u64 {
        self.total_src_bytes
    }

    /// Total destination (GPU) bytes after conversion.
    pub fn total_dst_bytes(&self) -> u64 {
        self.total_dst_bytes
    }

    /// Whether any entry needs format conversion.
    pub fn needs_conversion(&self) -> bool {
        self.entries.iter().any(|e| e.conversion.is_some())
    }

    /// Sort entries by offset for sequential I/O.
    pub fn sort_by_offset(&mut self) {
        self.entries.sort_by_key(|e| e.tensor.offset);
    }

    /// Sort entries by size descending (largest first for better GPU packing).
    pub fn sort_by_size_desc(&mut self) {
        self.entries.sort_by(|a, b| b.tensor.size_bytes.cmp(&a.tensor.size_bytes));
    }
}

impl Default for WeightLoadPlan {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WeightLoadProgress
// ---------------------------------------------------------------------------

/// Tracks weight loading progress.
#[derive(Debug, Clone)]
pub struct WeightLoadProgress {
    pub total_tensors: usize,
    pub loaded_tensors: usize,
    pub total_bytes: u64,
    pub bytes_transferred: u64,
    /// Names of tensors that failed to load.
    pub errors: Vec<String>,
}

impl WeightLoadProgress {
    /// Create progress tracker for a plan.
    pub fn new(plan: &WeightLoadPlan) -> Self {
        Self {
            total_tensors: plan.len(),
            loaded_tensors: 0,
            total_bytes: plan.total_dst_bytes(),
            bytes_transferred: 0,
            errors: Vec::new(),
        }
    }

    /// Record that a tensor was loaded successfully.
    pub fn record_success(&mut self, bytes: u64) {
        self.loaded_tensors += 1;
        self.bytes_transferred += bytes;
    }

    /// Record a tensor load failure.
    pub fn record_failure(&mut self, tensor_name: &str) {
        self.errors.push(tensor_name.to_string());
    }

    /// Fraction complete in `[0.0, 1.0]`.
    pub fn fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            return if self.total_tensors == 0 { 1.0 } else { 0.0 };
        }
        self.bytes_transferred as f64 / self.total_bytes as f64
    }

    /// Percentage complete `[0, 100]`.
    pub fn percent(&self) -> u32 {
        (self.fraction() * 100.0).min(100.0) as u32
    }

    /// Whether all tensors have been processed (success or failure).
    pub fn is_complete(&self) -> bool {
        self.loaded_tensors + self.errors.len() >= self.total_tensors
    }

    /// Whether all tensors loaded without errors.
    pub fn is_success(&self) -> bool {
        self.is_complete() && self.errors.is_empty()
    }
}

// ---------------------------------------------------------------------------
// GgufLoadError
// ---------------------------------------------------------------------------

/// Errors specific to GGUF loading for OpenCL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufLoadError {
    HeaderTooShort { have: usize, need: usize },
    BadMagic([u8; 4]),
    UnsupportedVersion(u32),
    TensorCountExceeded { count: u64, limit: u64 },
}

impl fmt::Display for GgufLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HeaderTooShort { have, need } => {
                write!(f, "header too short: have {} bytes, need {}", have, need)
            }
            Self::BadMagic(m) => {
                write!(f, "bad GGUF magic: {:02X} {:02X} {:02X} {:02X}", m[0], m[1], m[2], m[3])
            }
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported GGUF version: {}", v)
            }
            Self::TensorCountExceeded { count, limit } => {
                write!(f, "tensor count {} exceeds limit {}", count, limit)
            }
        }
    }
}

impl std::error::Error for GgufLoadError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===== GgufFieldType =====

    #[test]
    fn field_type_round_trip_all_variants() {
        let variants = [
            (0, GgufFieldType::U8),
            (1, GgufFieldType::I8),
            (2, GgufFieldType::U16),
            (3, GgufFieldType::I16),
            (4, GgufFieldType::U32),
            (5, GgufFieldType::I32),
            (6, GgufFieldType::F32),
            (7, GgufFieldType::Bool),
            (8, GgufFieldType::String),
            (9, GgufFieldType::Array),
            (10, GgufFieldType::F16),
        ];
        for (tag, expected) in &variants {
            let parsed = GgufFieldType::from_u32(*tag).unwrap();
            assert_eq!(parsed, *expected);
            assert_eq!(parsed.to_u32(), *tag);
        }
    }

    #[test]
    fn field_type_unknown_returns_none() {
        assert!(GgufFieldType::from_u32(255).is_none());
        assert!(GgufFieldType::from_u32(11).is_none());
    }

    #[test]
    fn field_type_display() {
        assert_eq!(GgufFieldType::F32.to_string(), "F32");
        assert_eq!(GgufFieldType::String.to_string(), "String");
    }

    // ===== GgufHeader =====

    fn make_header_bytes(magic: &[u8; 4], ver: u32, tensors: u64, kv: u64) -> Vec<u8> {
        let mut buf = Vec::with_capacity(24);
        buf.extend_from_slice(magic);
        buf.extend_from_slice(&ver.to_le_bytes());
        buf.extend_from_slice(&tensors.to_le_bytes());
        buf.extend_from_slice(&kv.to_le_bytes());
        buf
    }

    #[test]
    fn header_parse_v2_valid() {
        let data = make_header_bytes(b"GGUF", 2, 10, 5);
        let (hdr, consumed) = GgufHeader::parse(&data).unwrap();
        assert_eq!(hdr.magic, *b"GGUF");
        assert_eq!(hdr.version, 2);
        assert_eq!(hdr.n_tensors, 10);
        assert_eq!(hdr.n_kv, 5);
        assert_eq!(consumed, 24);
    }

    #[test]
    fn header_parse_v3_valid() {
        let data = make_header_bytes(b"GGUF", 3, 100, 20);
        let (hdr, _) = GgufHeader::parse(&data).unwrap();
        assert_eq!(hdr.version, 3);
        assert_eq!(hdr.n_tensors, 100);
    }

    #[test]
    fn header_bad_magic() {
        let data = make_header_bytes(b"GGML", 2, 1, 0);
        let err = GgufHeader::parse(&data).unwrap_err();
        assert!(matches!(err, GgufLoadError::BadMagic(_)));
    }

    #[test]
    fn header_too_short() {
        let data = vec![0u8; 10];
        let err = GgufHeader::parse(&data).unwrap_err();
        assert!(matches!(err, GgufLoadError::HeaderTooShort { .. }));
    }

    #[test]
    fn header_empty_data() {
        let err = GgufHeader::parse(&[]).unwrap_err();
        assert!(matches!(err, GgufLoadError::HeaderTooShort { have: 0, need: 24 }));
    }

    #[test]
    fn header_version_1_rejected() {
        let data = make_header_bytes(b"GGUF", 1, 1, 0);
        let err = GgufHeader::parse(&data).unwrap_err();
        assert!(matches!(err, GgufLoadError::UnsupportedVersion(1)));
    }

    #[test]
    fn header_version_99_rejected() {
        let data = make_header_bytes(b"GGUF", 99, 1, 0);
        let err = GgufHeader::parse(&data).unwrap_err();
        assert!(matches!(err, GgufLoadError::UnsupportedVersion(99)));
    }

    #[test]
    fn header_tensor_count_exceeded() {
        let data = make_header_bytes(b"GGUF", 3, 200_000, 0);
        let err = GgufHeader::parse(&data).unwrap_err();
        assert!(matches!(err, GgufLoadError::TensorCountExceeded { .. }));
    }

    #[test]
    fn header_zero_tensors_ok() {
        let data = make_header_bytes(b"GGUF", 3, 0, 0);
        let (hdr, _) = GgufHeader::parse(&data).unwrap();
        assert_eq!(hdr.n_tensors, 0);
    }

    #[test]
    fn header_round_trip() {
        let data = make_header_bytes(b"GGUF", 3, 42, 7);
        let (hdr, _) = GgufHeader::parse(&data).unwrap();
        let serialized = hdr.to_bytes();
        assert_eq!(serialized, data);
    }

    #[test]
    fn header_exactly_24_bytes() {
        let data = make_header_bytes(b"GGUF", 2, 1, 0);
        assert_eq!(data.len(), 24);
        assert!(GgufHeader::parse(&data).is_ok());
    }

    #[test]
    fn header_extra_trailing_bytes() {
        let mut data = make_header_bytes(b"GGUF", 3, 5, 2);
        data.extend_from_slice(&[0xDE, 0xAD]);
        let (hdr, consumed) = GgufHeader::parse(&data).unwrap();
        assert_eq!(consumed, 24);
        assert_eq!(hdr.n_tensors, 5);
    }

    // ===== GgufTensorInfo =====

    #[test]
    fn tensor_info_n_elements_2d() {
        let t = GgufTensorInfo::new("w", vec![4096, 4096], TensorDtype::F32, 0, 67108864);
        assert_eq!(t.n_elements(), 4096 * 4096);
    }

    #[test]
    fn tensor_info_n_elements_1d() {
        let t = GgufTensorInfo::new("b", vec![512], TensorDtype::F16, 0, 1024);
        assert_eq!(t.n_elements(), 512);
    }

    #[test]
    fn tensor_info_n_elements_empty_dims() {
        let t = GgufTensorInfo::new("empty", vec![], TensorDtype::F32, 0, 0);
        assert_eq!(t.n_elements(), 0);
    }

    #[test]
    fn tensor_info_n_elements_3d() {
        let t = GgufTensorInfo::new("x", vec![2, 3, 4], TensorDtype::F32, 0, 96);
        assert_eq!(t.n_elements(), 24);
    }

    #[test]
    fn tensor_info_name_and_dtype() {
        let t = GgufTensorInfo::new("layer.0.weight", vec![128, 256], TensorDtype::I2_S, 100, 4096);
        assert_eq!(t.name, "layer.0.weight");
        assert_eq!(t.dtype, TensorDtype::I2_S);
        assert_eq!(t.offset, 100);
    }

    // ===== TensorDtype =====

    #[test]
    fn dtype_element_sizes() {
        assert_eq!(TensorDtype::F32.element_size(), 4);
        assert_eq!(TensorDtype::F16.element_size(), 2);
        assert_eq!(TensorDtype::I2_S.element_size(), 1);
        assert_eq!(TensorDtype::QK256.element_size(), 66);
    }

    #[test]
    fn dtype_is_blocked() {
        assert!(!TensorDtype::F32.is_blocked());
        assert!(!TensorDtype::F16.is_blocked());
        assert!(TensorDtype::I2_S.is_blocked());
        assert!(TensorDtype::QK256.is_blocked());
    }

    #[test]
    fn dtype_display() {
        assert_eq!(TensorDtype::F32.to_string(), "F32");
        assert_eq!(TensorDtype::QK256.to_string(), "QK256");
    }

    // ===== TensorLayout =====

    #[test]
    fn layout_for_dtype() {
        assert_eq!(TensorLayout::for_dtype(TensorDtype::F32), TensorLayout::RowMajor);
        assert_eq!(TensorLayout::for_dtype(TensorDtype::F16), TensorLayout::RowMajor);
        assert_eq!(TensorLayout::for_dtype(TensorDtype::I2_S), TensorLayout::Blocked(4));
        assert_eq!(TensorLayout::for_dtype(TensorDtype::QK256), TensorLayout::Blocked(256));
    }

    #[test]
    fn layout_is_blocked() {
        assert!(!TensorLayout::RowMajor.is_blocked());
        assert!(!TensorLayout::ColumnMajor.is_blocked());
        assert!(TensorLayout::Blocked(32).is_blocked());
    }

    #[test]
    fn layout_block_size() {
        assert_eq!(TensorLayout::RowMajor.block_size(), 1);
        assert_eq!(TensorLayout::ColumnMajor.block_size(), 1);
        assert_eq!(TensorLayout::Blocked(256).block_size(), 256);
    }

    #[test]
    fn layout_display() {
        assert_eq!(TensorLayout::RowMajor.to_string(), "RowMajor");
        assert_eq!(TensorLayout::Blocked(32).to_string(), "Blocked(32)");
    }

    // ===== FormatConversion =====

    #[test]
    fn conversion_passthrough() {
        let c = FormatConversion::new(TensorDtype::F32, TensorDtype::F32);
        assert!(c.is_passthrough());
    }

    #[test]
    fn conversion_f32_to_f16() {
        let c = FormatConversion::new(TensorDtype::F32, TensorDtype::F16);
        assert!(!c.is_passthrough());
        assert_eq!(c.to_string(), "F32 → F16");
    }

    // ===== FormatConverter — F32 ↔ F16 =====

    #[test]
    fn f32_to_f16_zero() {
        let result = FormatConverter::f32_to_f16(&[0.0]);
        assert_eq!(result, vec![0x0000]);
    }

    #[test]
    fn f32_to_f16_one() {
        let result = FormatConverter::f32_to_f16(&[1.0]);
        assert_eq!(result, vec![0x3C00]);
    }

    #[test]
    fn f32_to_f16_neg_one() {
        let result = FormatConverter::f32_to_f16(&[-1.0]);
        assert_eq!(result, vec![0xBC00]);
    }

    #[test]
    fn f16_to_f32_round_trip() {
        let original = vec![0.0f32, 1.0, -1.0, 0.5, -0.5, 65504.0];
        let half = FormatConverter::f32_to_f16(&original);
        let back = FormatConverter::f16_to_f32(&half);
        for (a, b) in original.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-3, "mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn f32_to_f16_infinity() {
        let half = FormatConverter::f32_to_f16(&[f32::INFINITY]);
        assert_eq!(half, vec![0x7C00]);
    }

    #[test]
    fn f32_to_f16_neg_infinity() {
        let half = FormatConverter::f32_to_f16(&[f32::NEG_INFINITY]);
        assert_eq!(half, vec![0xFC00]);
    }

    #[test]
    fn f32_to_f16_nan() {
        let half = FormatConverter::f32_to_f16(&[f32::NAN]);
        // NaN preserves sign=0, exp=0x1F, mantissa!=0
        assert_eq!(half[0] & 0x7C00, 0x7C00);
        assert_ne!(half[0] & 0x03FF, 0);
    }

    #[test]
    fn f32_to_f16_multiple_values() {
        let input = vec![0.0, 1.0, 2.0, 0.25];
        let half = FormatConverter::f32_to_f16(&input);
        assert_eq!(half.len(), 4);
        let back = FormatConverter::f16_to_f32(&half);
        for (a, b) in input.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ===== FormatConverter — I2_S =====

    #[test]
    fn i2s_pack_unpack_round_trip() {
        let values = vec![-1i8, 0, 1, -1, 0, 1, 1, 0];
        let packed = FormatConverter::pack_i2s(&values);
        let unpacked = FormatConverter::unpack_i2s(&packed, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn i2s_pack_single_value() {
        let packed = FormatConverter::pack_i2s(&[-1]);
        assert_eq!(packed.len(), 1);
        let unpacked = FormatConverter::unpack_i2s(&packed, 1);
        assert_eq!(unpacked, vec![-1]);
    }

    #[test]
    fn i2s_pack_four_values() {
        let values = vec![1i8, 0, -1, 1];
        let packed = FormatConverter::pack_i2s(&values);
        assert_eq!(packed.len(), 1);
        let unpacked = FormatConverter::unpack_i2s(&packed, 4);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn i2s_pack_non_multiple_of_four() {
        let values = vec![-1i8, 0, 1];
        let packed = FormatConverter::pack_i2s(&values);
        assert_eq!(packed.len(), 1);
        let unpacked = FormatConverter::unpack_i2s(&packed, 3);
        assert_eq!(unpacked, values);
    }

    // ===== FormatConverter — QK256 =====

    #[test]
    fn qk256_dequantize_zeros() {
        // scale=0, min=0, all quants=0 → all zeros
        let block = vec![0u8; 68];
        let out = FormatConverter::dequantize_qk256_block(&block);
        assert_eq!(out.len(), 256);
        for v in &out {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn qk256_dequantize_short_block() {
        let block = vec![0u8; 4]; // too short — returns zeros
        let out = FormatConverter::dequantize_qk256_block(&block);
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn qk256_dequantize_scale_only() {
        // scale=1.0 (f16=0x3C00), min=0.0, quants all set to 0b01 (=1)
        let mut block = vec![0u8; 68];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        // quant byte with all 0b01 patterns: 0b01_01_01_01 = 0x55
        for b in &mut block[4..68] {
            *b = 0x55;
        }
        let out = FormatConverter::dequantize_qk256_block(&block);
        for v in &out {
            assert!((v - 1.0).abs() < 1e-3, "expected ~1.0, got {}", v);
        }
    }

    // ===== WeightLoadPlan =====

    #[test]
    fn plan_empty() {
        let plan = WeightLoadPlan::new();
        assert!(plan.is_empty());
        assert_eq!(plan.len(), 0);
        assert_eq!(plan.total_src_bytes(), 0);
        assert_eq!(plan.total_dst_bytes(), 0);
        assert!(!plan.needs_conversion());
    }

    #[test]
    fn plan_build_no_conversion() {
        let tensors = vec![
            GgufTensorInfo::new("a", vec![1024], TensorDtype::F32, 0, 4096),
            GgufTensorInfo::new("b", vec![512], TensorDtype::F32, 4096, 2048),
        ];
        let plan = WeightLoadPlan::build(&tensors, None);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.total_src_bytes(), 6144);
        assert!(!plan.needs_conversion());
    }

    #[test]
    fn plan_build_with_conversion() {
        let tensors = vec![GgufTensorInfo::new("w", vec![1024], TensorDtype::F32, 0, 4096)];
        let plan = WeightLoadPlan::build(&tensors, Some(TensorDtype::F16));
        assert!(plan.needs_conversion());
        let entry = &plan.entries()[0];
        assert!(entry.conversion.is_some());
        // F32 → F16: 1024 elements × 2 bytes = 2048
        assert_eq!(entry.dst_size_bytes, 2048);
    }

    #[test]
    fn plan_build_same_dtype_no_conversion() {
        let tensors = vec![GgufTensorInfo::new("w", vec![256], TensorDtype::F16, 0, 512)];
        let plan = WeightLoadPlan::build(&tensors, Some(TensorDtype::F16));
        assert!(!plan.needs_conversion());
    }

    #[test]
    fn plan_sort_by_offset() {
        let tensors = vec![
            GgufTensorInfo::new("b", vec![100], TensorDtype::F32, 200, 400),
            GgufTensorInfo::new("a", vec![100], TensorDtype::F32, 0, 400),
            GgufTensorInfo::new("c", vec![100], TensorDtype::F32, 100, 400),
        ];
        let mut plan = WeightLoadPlan::build(&tensors, None);
        plan.sort_by_offset();
        let names: Vec<&str> = plan.entries().iter().map(|e| e.tensor.name.as_str()).collect();
        assert_eq!(names, vec!["a", "c", "b"]);
    }

    #[test]
    fn plan_sort_by_size_desc() {
        let tensors = vec![
            GgufTensorInfo::new("small", vec![10], TensorDtype::F32, 0, 40),
            GgufTensorInfo::new("big", vec![1000], TensorDtype::F32, 40, 4000),
            GgufTensorInfo::new("med", vec![100], TensorDtype::F32, 4040, 400),
        ];
        let mut plan = WeightLoadPlan::build(&tensors, None);
        plan.sort_by_size_desc();
        let names: Vec<&str> = plan.entries().iter().map(|e| e.tensor.name.as_str()).collect();
        assert_eq!(names, vec!["big", "med", "small"]);
    }

    #[test]
    fn plan_single_tensor() {
        let tensors = vec![GgufTensorInfo::new("only", vec![64], TensorDtype::F16, 0, 128)];
        let plan = WeightLoadPlan::build(&tensors, None);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.total_src_bytes(), 128);
    }

    #[test]
    fn plan_i2s_to_f32_conversion_size() {
        // 1024 I2_S elements → 1024 × 4 = 4096 bytes as F32
        let tensors = vec![GgufTensorInfo::new(
            "w",
            vec![1024],
            TensorDtype::I2_S,
            0,
            256, // 1024 / 4 = 256 bytes packed
        )];
        let plan = WeightLoadPlan::build(&tensors, Some(TensorDtype::F32));
        assert_eq!(plan.entries()[0].dst_size_bytes, 4096);
    }

    // ===== WeightLoadProgress =====

    #[test]
    fn progress_initial() {
        let plan = WeightLoadPlan::build(
            &[GgufTensorInfo::new("a", vec![100], TensorDtype::F32, 0, 400)],
            None,
        );
        let prog = WeightLoadProgress::new(&plan);
        assert_eq!(prog.total_tensors, 1);
        assert_eq!(prog.loaded_tensors, 0);
        assert_eq!(prog.percent(), 0);
        assert!(!prog.is_complete());
    }

    #[test]
    fn progress_after_one_load() {
        let plan = WeightLoadPlan::build(
            &[
                GgufTensorInfo::new("a", vec![100], TensorDtype::F32, 0, 400),
                GgufTensorInfo::new("b", vec![100], TensorDtype::F32, 400, 400),
            ],
            None,
        );
        let mut prog = WeightLoadProgress::new(&plan);
        prog.record_success(400);
        assert_eq!(prog.loaded_tensors, 1);
        assert_eq!(prog.percent(), 50);
        assert!(!prog.is_complete());
    }

    #[test]
    fn progress_complete() {
        let plan = WeightLoadPlan::build(
            &[GgufTensorInfo::new("a", vec![100], TensorDtype::F32, 0, 400)],
            None,
        );
        let mut prog = WeightLoadProgress::new(&plan);
        prog.record_success(400);
        assert!(prog.is_complete());
        assert!(prog.is_success());
        assert_eq!(prog.percent(), 100);
    }

    #[test]
    fn progress_with_failure() {
        let plan = WeightLoadPlan::build(
            &[GgufTensorInfo::new("a", vec![100], TensorDtype::F32, 0, 400)],
            None,
        );
        let mut prog = WeightLoadProgress::new(&plan);
        prog.record_failure("a");
        assert!(prog.is_complete());
        assert!(!prog.is_success());
        assert_eq!(prog.errors.len(), 1);
    }

    #[test]
    fn progress_empty_plan() {
        let plan = WeightLoadPlan::new();
        let prog = WeightLoadProgress::new(&plan);
        assert!(prog.is_complete());
        assert!(prog.is_success());
        assert_eq!(prog.fraction(), 1.0);
    }

    #[test]
    fn progress_fraction_accuracy() {
        let plan = WeightLoadPlan::build(
            &[
                GgufTensorInfo::new("a", vec![100], TensorDtype::F32, 0, 400),
                GgufTensorInfo::new("b", vec![300], TensorDtype::F32, 400, 1200),
            ],
            None,
        );
        let mut prog = WeightLoadProgress::new(&plan);
        prog.record_success(400);
        assert!((prog.fraction() - 0.25).abs() < 1e-9);
    }

    // ===== Edge cases =====

    #[test]
    fn header_corrupted_all_zeros() {
        let data = vec![0u8; 24];
        let err = GgufHeader::parse(&data).unwrap_err();
        assert!(matches!(err, GgufLoadError::BadMagic(_)));
    }

    #[test]
    fn header_corrupted_all_ff() {
        let data = vec![0xFFu8; 24];
        let err = GgufHeader::parse(&data).unwrap_err();
        assert!(matches!(err, GgufLoadError::BadMagic(_)));
    }

    #[test]
    fn huge_tensor_count_rejected() {
        let data = make_header_bytes(b"GGUF", 3, u64::MAX, 0);
        let err = GgufHeader::parse(&data).unwrap_err();
        assert!(matches!(err, GgufLoadError::TensorCountExceeded { .. }));
    }

    #[test]
    fn error_display() {
        let e = GgufLoadError::BadMagic([0x47, 0x47, 0x4D, 0x4C]);
        assert!(e.to_string().contains("bad GGUF magic"));
        let e = GgufLoadError::UnsupportedVersion(1);
        assert!(e.to_string().contains("unsupported GGUF version: 1"));
    }

    // ===== Property-style tests for format conversions =====

    #[test]
    fn f32_f16_round_trip_many_values() {
        // Test a range of representable F16 values
        let values: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.1).collect();
        let half = FormatConverter::f32_to_f16(&values);
        let back = FormatConverter::f16_to_f32(&half);
        for (a, b) in values.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.01, "round-trip mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn i2s_round_trip_all_patterns() {
        // Exhaustive: every possible 4-element pattern
        for a in [-1i8, 0, 1] {
            for b in [-1, 0, 1] {
                for c in [-1, 0, 1] {
                    for d in [-1, 0, 1] {
                        let values = vec![a, b, c, d];
                        let packed = FormatConverter::pack_i2s(&values);
                        let unpacked = FormatConverter::unpack_i2s(&packed, 4);
                        assert_eq!(unpacked, values, "failed for {:?}", values);
                    }
                }
            }
        }
    }

    #[test]
    fn i2s_round_trip_lengths_1_to_16() {
        for len in 1..=16 {
            let values: Vec<i8> = (0..len).map(|i| [-1, 0, 1][i % 3]).collect();
            let packed = FormatConverter::pack_i2s(&values);
            let unpacked = FormatConverter::unpack_i2s(&packed, len);
            assert_eq!(unpacked, values, "failed for len={}", len);
        }
    }

    #[test]
    fn f32_f16_zero_preserves_sign() {
        let pos = FormatConverter::f32_to_f16(&[0.0]);
        let neg = FormatConverter::f32_to_f16(&[-0.0]);
        assert_eq!(pos[0], 0x0000);
        assert_eq!(neg[0], 0x8000);
    }
}
