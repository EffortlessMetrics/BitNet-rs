//! Weight compression/decompression kernels for efficient model storage.
//!
//! Provides multiple compression strategies for ternary ({-1, 0, 1}) weight
//! tensors commonly found in BitNet models.  Each algorithm trades CPU time
//! for reduced memory footprint:
//!
//! | Format         | Best for                              |
//! |----------------|---------------------------------------|
//! | `BitPacking`   | Dense ternary weights (2 bits/value)  |
//! | `RunLength`    | Highly sparse weights (many zeros)    |
//! | `DeltaEncoding`| Slowly-varying or sorted patterns     |
//! | `Huffman`      | Skewed distributions (e.g., 90% zero) |
//! | `None`         | Already-compact or tiny tensors       |
//!
//! [`adaptive_compress`] automatically benchmarks all formats and picks the
//! smallest result for a given input.
//!
//! # CPU fallback
//!
//! All public functions have pure-Rust implementations that work on any
//! platform.  GPU-specific CUDA kernel source strings are gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use bitnet_common::{KernelError, Result};

// ── Compression format enum ───────────────────────────────────────────

/// Supported weight compression formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionFormat {
    /// No compression — raw storage.
    None,
    /// Run-length encoding for sparse ternary weights.
    RunLength,
    /// Huffman-style variable-length coding (simplified canonical form).
    Huffman,
    /// Delta encoding for slowly-varying weight patterns.
    DeltaEncoding,
    /// 2-bit packing of ternary {-1, 0, +1} values (4 values per byte).
    BitPacking,
}

/// Header prepended to every compressed payload so that the decompressor
/// can identify format and original length without out-of-band metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionHeader {
    /// Format used for this payload.
    pub format: CompressionFormat,
    /// Number of logical elements before compression.
    pub original_len: usize,
}

impl CompressionHeader {
    /// Serialise into a fixed 8-byte prefix: `[format_tag(u8), padding(3),
    /// original_len(u32 LE)]`.
    fn to_bytes(self) -> [u8; 8] {
        let mut buf = [0u8; 8];
        buf[0] = match self.format {
            CompressionFormat::None => 0,
            CompressionFormat::RunLength => 1,
            CompressionFormat::Huffman => 2,
            CompressionFormat::DeltaEncoding => 3,
            CompressionFormat::BitPacking => 4,
        };
        let len_bytes = (self.original_len as u32).to_le_bytes();
        buf[4..8].copy_from_slice(&len_bytes);
        buf
    }

    /// Deserialise from the first 8 bytes of a buffer.
    fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < 8 {
            return Err(KernelError::InvalidArguments {
                reason: "compressed buffer too short for header".into(),
            }
            .into());
        }
        let format = match buf[0] {
            0 => CompressionFormat::None,
            1 => CompressionFormat::RunLength,
            2 => CompressionFormat::Huffman,
            3 => CompressionFormat::DeltaEncoding,
            4 => CompressionFormat::BitPacking,
            t => {
                return Err(KernelError::InvalidArguments {
                    reason: format!("unknown compression tag {t}"),
                }
                .into());
            }
        };
        let original_len = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
        Ok(Self { format, original_len })
    }
}

// ── Bit-packing (ternary 2-bit) ──────────────────────────────────────

/// Encode a single ternary value into 2 bits: +1 → 0b01, 0 → 0b00,
/// −1 → 0b10.
#[inline(always)]
fn ternary_to_bits(v: i8) -> u8 {
    match v {
        1 => 0b01,
        -1 => 0b10,
        _ => 0b00,
    }
}

/// Decode 2 bits back to a ternary value.
#[inline(always)]
fn bits_to_ternary(bits: u8) -> i8 {
    match bits & 0x03 {
        0b01 => 1,
        0b10 => -1,
        _ => 0,
    }
}

/// Pack ternary {-1, 0, +1} values into 2 bits each (4 values per byte,
/// LSB-first).
///
/// # Errors
///
/// Returns an error if any value is outside {-1, 0, +1}.
pub fn bitpack_ternary(values: &[i8]) -> Result<Vec<u8>> {
    for (i, &v) in values.iter().enumerate() {
        if !(-1..=1).contains(&v) {
            return Err(KernelError::InvalidArguments {
                reason: format!("value {v} at index {i} is not ternary (must be -1, 0, or 1)"),
            }
            .into());
        }
    }
    let packed_len = values.len().div_ceil(4);
    let mut packed = vec![0u8; packed_len];
    for (i, &v) in values.iter().enumerate() {
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        packed[byte_idx] |= ternary_to_bits(v) << bit_off;
    }
    Ok(packed)
}

/// Unpack 2-bit ternary values from a packed buffer.
///
/// Returns exactly `num_values` elements.
///
/// # Errors
///
/// Returns an error if the packed buffer is too short.
pub fn unpack_ternary(packed: &[u8], num_values: usize) -> Result<Vec<i8>> {
    let required = num_values.div_ceil(4);
    if packed.len() < required {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "packed buffer too short: need {required} bytes for \
                 {num_values} values, got {}",
                packed.len()
            ),
        }
        .into());
    }
    let mut out = Vec::with_capacity(num_values);
    for i in 0..num_values {
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        out.push(bits_to_ternary((packed[byte_idx] >> bit_off) & 0x03));
    }
    Ok(out)
}

// ── Delta encoding / decoding ─────────────────────────────────────────

/// Delta-encode a sequence of i8 values.
///
/// The first element is stored verbatim; subsequent elements store the
/// difference from their predecessor, clamped to i8 range.
pub fn delta_encode(values: &[i8]) -> Vec<i8> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut encoded = Vec::with_capacity(values.len());
    encoded.push(values[0]);
    for i in 1..values.len() {
        let diff = (values[i] as i16) - (values[i - 1] as i16);
        encoded.push(diff.clamp(i8::MIN as i16, i8::MAX as i16) as i8);
    }
    encoded
}

/// Delta-decode a sequence previously encoded with [`delta_encode`].
pub fn delta_decode(encoded: &[i8]) -> Vec<i8> {
    if encoded.is_empty() {
        return Vec::new();
    }
    let mut decoded = Vec::with_capacity(encoded.len());
    decoded.push(encoded[0]);
    for i in 1..encoded.len() {
        let prev = decoded[i - 1] as i16;
        let val = (prev + encoded[i] as i16).clamp(i8::MIN as i16, i8::MAX as i16);
        decoded.push(val as i8);
    }
    decoded
}

// ── Run-length encoding / decoding ────────────────────────────────────

/// A single run in run-length encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    /// The repeated value.
    pub value: i8,
    /// How many consecutive times the value appears (≥ 1).
    pub count: u32,
}

/// Run-length encode a sequence of i8 values.
///
/// Consecutive identical values are collapsed into `(value, count)` pairs.
pub fn run_length_encode(values: &[i8]) -> Vec<Run> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut current = values[0];
    let mut count = 1u32;
    for &v in &values[1..] {
        if v == current {
            count += 1;
        } else {
            runs.push(Run { value: current, count });
            current = v;
            count = 1;
        }
    }
    runs.push(Run { value: current, count });
    runs
}

/// Decode a run-length encoded sequence back into i8 values.
pub fn run_length_decode(runs: &[Run]) -> Vec<i8> {
    let total: usize = runs.iter().map(|r| r.count as usize).sum();
    let mut out = Vec::with_capacity(total);
    for r in runs {
        for _ in 0..r.count {
            out.push(r.value);
        }
    }
    out
}

// ── Serialised RLE (byte stream) ──────────────────────────────────────

/// Serialise runs into a compact byte stream: `[value: i8, count: u32 LE]`
/// per run (5 bytes each).
fn rle_to_bytes(runs: &[Run]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(runs.len() * 5);
    for r in runs {
        buf.push(r.value as u8);
        buf.extend_from_slice(&r.count.to_le_bytes());
    }
    buf
}

/// Deserialise runs from a byte stream produced by [`rle_to_bytes`].
fn rle_from_bytes(data: &[u8]) -> Result<Vec<Run>> {
    if !data.len().is_multiple_of(5) {
        return Err(KernelError::InvalidArguments {
            reason: format!("RLE data length {} is not a multiple of 5", data.len()),
        }
        .into());
    }
    let n = data.len() / 5;
    let mut runs = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 5;
        let value = data[off] as i8;
        let count =
            u32::from_le_bytes([data[off + 1], data[off + 2], data[off + 3], data[off + 4]]);
        runs.push(Run { value, count });
    }
    Ok(runs)
}

// ── Simplified Huffman (canonical 3-symbol) ───────────────────────────
//
// For ternary weights with only three possible values we use a fixed
// canonical coding based on symbol frequency:
//
//   Most frequent  → 1 bit  (0)
//   Second         → 2 bits (10)
//   Third          → 2 bits (11)
//
// The header stores the symbol order (3 bytes) so the decoder can
// reconstruct the mapping.

/// Huffman-encode ternary values into a packed bitstream.
///
/// Returns `(symbol_order, bit_count, packed_bytes)`.
fn huffman_encode_ternary(values: &[i8]) -> (Vec<i8>, usize, Vec<u8>) {
    if values.is_empty() {
        return (vec![0, 1, -1], 0, Vec::new());
    }
    // Count frequencies.
    let mut freq = [0usize; 3]; // index: 0 → val 0, 1 → val 1, 2 → val -1
    for &v in values {
        match v {
            0 => freq[0] += 1,
            1 => freq[1] += 1,
            _ => freq[2] += 1,
        }
    }
    // Sort symbols by descending frequency.
    let mut order: Vec<(usize, i8)> = vec![(freq[0], 0i8), (freq[1], 1i8), (freq[2], -1i8)];
    order.sort_by(|a, b| b.0.cmp(&a.0));
    let symbols: Vec<i8> = order.iter().map(|&(_, s)| s).collect();

    // Encode: most-frequent → 0 (1 bit), second → 10 (2 bits), third → 11 (2 bits).
    let mut bits = Vec::with_capacity(values.len() * 2);
    for &v in values {
        if v == symbols[0] {
            bits.push(false);
        } else if v == symbols[1] {
            bits.push(true);
            bits.push(false);
        } else {
            bits.push(true);
            bits.push(true);
        }
    }
    let bit_count = bits.len();
    let byte_count = bit_count.div_ceil(8);
    let mut packed = vec![0u8; byte_count];
    for (i, &b) in bits.iter().enumerate() {
        if b {
            packed[i / 8] |= 1 << (i % 8);
        }
    }
    (symbols, bit_count, packed)
}

/// Huffman-decode a bitstream produced by [`huffman_encode_ternary`].
fn huffman_decode_ternary(
    symbols: &[i8],
    bit_count: usize,
    packed: &[u8],
    num_values: usize,
) -> Result<Vec<i8>> {
    if symbols.len() < 3 {
        return Err(KernelError::InvalidArguments {
            reason: "huffman symbol table must have 3 entries".into(),
        }
        .into());
    }
    let mut out = Vec::with_capacity(num_values);
    let mut bit_idx = 0usize;

    let read_bit = |idx: usize| -> bool {
        if idx >= bit_count {
            return false;
        }
        (packed[idx / 8] >> (idx % 8)) & 1 == 1
    };

    while out.len() < num_values && bit_idx < bit_count {
        if !read_bit(bit_idx) {
            out.push(symbols[0]);
            bit_idx += 1;
        } else {
            bit_idx += 1;
            if read_bit(bit_idx) {
                out.push(symbols[2]);
            } else {
                out.push(symbols[1]);
            }
            bit_idx += 1;
        }
    }
    Ok(out)
}

/// Serialise Huffman payload: `[sym0, sym1, sym2, bit_count(u32 LE),
/// packed_bytes…]`.
fn huffman_to_bytes(symbols: &[i8], bit_count: usize, packed: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(3 + 4 + packed.len());
    for &s in &symbols[..3] {
        buf.push(s as u8);
    }
    buf.extend_from_slice(&(bit_count as u32).to_le_bytes());
    buf.extend_from_slice(packed);
    buf
}

/// Deserialise Huffman payload.
fn huffman_from_bytes(data: &[u8]) -> Result<(Vec<i8>, usize, Vec<u8>)> {
    if data.len() < 7 {
        return Err(
            KernelError::InvalidArguments { reason: "huffman payload too short".into() }.into()
        );
    }
    let symbols = vec![data[0] as i8, data[1] as i8, data[2] as i8];
    let bit_count = u32::from_le_bytes([data[3], data[4], data[5], data[6]]) as usize;
    let packed = data[7..].to_vec();
    Ok((symbols, bit_count, packed))
}

// ── High-level compress / decompress ──────────────────────────────────

/// Compress a weight tensor using the specified format.
///
/// Returns a self-describing byte buffer (header + payload) that can be
/// passed to [`decompress_weights`].
///
/// # Errors
///
/// Returns an error if any value is outside {-1, 0, +1} when using
/// `BitPacking` or `Huffman`, or on internal serialisation failures.
pub fn compress_weights(weights: &[i8], format: CompressionFormat) -> Result<Vec<u8>> {
    let header = CompressionHeader { format, original_len: weights.len() };
    let mut buf = header.to_bytes().to_vec();

    match format {
        CompressionFormat::None => {
            buf.extend(weights.iter().map(|&v| v as u8));
        }
        CompressionFormat::BitPacking => {
            let packed = bitpack_ternary(weights)?;
            buf.extend_from_slice(&packed);
        }
        CompressionFormat::RunLength => {
            let runs = run_length_encode(weights);
            buf.extend_from_slice(&rle_to_bytes(&runs));
        }
        CompressionFormat::DeltaEncoding => {
            let encoded = delta_encode(weights);
            buf.extend(encoded.iter().map(|&v| v as u8));
        }
        CompressionFormat::Huffman => {
            let (symbols, bit_count, packed) = huffman_encode_ternary(weights);
            buf.extend_from_slice(&huffman_to_bytes(&symbols, bit_count, &packed));
        }
    }
    Ok(buf)
}

/// Decompress a weight tensor previously compressed with
/// [`compress_weights`].
///
/// # Errors
///
/// Returns an error on corrupted or truncated buffers.
pub fn decompress_weights(compressed: &[u8]) -> Result<Vec<i8>> {
    let header = CompressionHeader::from_bytes(compressed)?;
    let payload = &compressed[8..];

    match header.format {
        CompressionFormat::None => {
            if payload.len() < header.original_len {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "none-compressed payload too short: need {}, got {}",
                        header.original_len,
                        payload.len()
                    ),
                }
                .into());
            }
            Ok(payload[..header.original_len].iter().map(|&b| b as i8).collect())
        }
        CompressionFormat::BitPacking => unpack_ternary(payload, header.original_len),
        CompressionFormat::RunLength => {
            let runs = rle_from_bytes(payload)?;
            let decoded = run_length_decode(&runs);
            if decoded.len() < header.original_len {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "RLE decoded {} elements, expected {}",
                        decoded.len(),
                        header.original_len
                    ),
                }
                .into());
            }
            Ok(decoded[..header.original_len].to_vec())
        }
        CompressionFormat::DeltaEncoding => {
            if payload.len() < header.original_len {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "delta payload too short: need {}, got {}",
                        header.original_len,
                        payload.len()
                    ),
                }
                .into());
            }
            let encoded: Vec<i8> =
                payload[..header.original_len].iter().map(|&b| b as i8).collect();
            Ok(delta_decode(&encoded))
        }
        CompressionFormat::Huffman => {
            let (symbols, bit_count, packed) = huffman_from_bytes(payload)?;
            huffman_decode_ternary(&symbols, bit_count, &packed, header.original_len)
        }
    }
}

// ── Compression ratio ─────────────────────────────────────────────────

/// Compute the compression ratio achieved for the given weights and
/// format.
///
/// Returns `original_size / compressed_size`.  A ratio > 1.0 means the
/// compressed form is smaller.
///
/// # Errors
///
/// Forwards any error from [`compress_weights`].
pub fn compression_ratio(weights: &[i8], format: CompressionFormat) -> Result<f64> {
    if weights.is_empty() {
        return Ok(1.0);
    }
    let compressed = compress_weights(weights, format)?;
    let original_bytes = weights.len(); // 1 byte per i8
    Ok(original_bytes as f64 / compressed.len() as f64)
}

// ── Adaptive compression ──────────────────────────────────────────────

/// Auto-select the best compression format for the given weights.
///
/// Tries every [`CompressionFormat`] variant and returns the compressed
/// buffer that achieves the smallest size.
///
/// # Errors
///
/// Returns an error only if *all* formats fail (should not happen for
/// valid ternary input).
pub fn adaptive_compress(weights: &[i8]) -> Result<Vec<u8>> {
    let formats = [
        CompressionFormat::None,
        CompressionFormat::BitPacking,
        CompressionFormat::RunLength,
        CompressionFormat::DeltaEncoding,
        CompressionFormat::Huffman,
    ];

    let mut best: Option<Vec<u8>> = Option::None;
    let mut best_len = usize::MAX;
    let mut last_err = Option::None;

    for fmt in &formats {
        match compress_weights(weights, *fmt) {
            Ok(compressed) => {
                if compressed.len() < best_len {
                    best_len = compressed.len();
                    best = Some(compressed);
                }
            }
            Err(e) => {
                last_err = Some(e);
            }
        }
    }

    best.ok_or_else(|| {
        last_err.unwrap_or_else(|| {
            KernelError::InvalidArguments { reason: "adaptive_compress: all formats failed".into() }
                .into()
        })
    })
}

// ── CUDA kernel source ────────────────────────────────────────────────

/// CUDA C source for ternary bit-pack kernel.
///
/// Kernel `bitpack_ternary_kernel` packs 4 ternary values per byte using
/// grid-stride loop. Each thread processes 4 consecutive elements.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const BITPACK_TERNARY_KERNEL_SRC: &str = r#"
extern "C" __global__ void bitpack_ternary_kernel(
    const signed char* __restrict__ input,
    unsigned char* __restrict__ output,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int packed_n = (n + 3) / 4;

    for (int i = tid; i < packed_n; i += stride) {
        unsigned char byte = 0;
        for (int j = 0; j < 4; j++) {
            int idx = i * 4 + j;
            if (idx < n) {
                signed char v = input[idx];
                unsigned char bits = (v == 1) ? 0x01 : (v == -1) ? 0x02 : 0x00;
                byte |= (bits << (j * 2));
            }
        }
        output[i] = byte;
    }
}
"#;

/// CUDA C source for ternary bit-unpack kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const UNPACK_TERNARY_KERNEL_SRC: &str = r#"
extern "C" __global__ void unpack_ternary_kernel(
    const unsigned char* __restrict__ input,
    signed char* __restrict__ output,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        int byte_idx = i / 4;
        int bit_off  = (i % 4) * 2;
        unsigned char bits = (input[byte_idx] >> bit_off) & 0x03;
        signed char v = (bits == 0x01) ? 1 : (bits == 0x02) ? -1 : 0;
        output[i] = v;
    }
}
"#;

/// CUDA C source for delta-decode kernel.
///
/// Uses an inclusive prefix-sum (scan) over the delta-encoded stream.
/// This simple version is single-block; production use should employ a
/// multi-block Blelloch scan.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const DELTA_DECODE_KERNEL_SRC: &str = r#"
extern "C" __global__ void delta_decode_kernel(
    signed char* __restrict__ data,
    int n)
{
    // Simple sequential prefix-sum (single thread).
    // For production, replace with parallel scan.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 1; i < n; i++) {
            int val = (int)data[i - 1] + (int)data[i];
            if (val > 127)  val = 127;
            if (val < -128) val = -128;
            data[i] = (signed char)val;
        }
    }
}
"#;

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────

    fn assert_ternary(values: &[i8]) {
        for (i, &v) in values.iter().enumerate() {
            assert!(v == -1 || v == 0 || v == 1, "non-ternary value {v} at index {i}");
        }
    }

    // ── CompressionFormat / Header ────────────────────────────────

    #[test]
    fn header_roundtrip_all_formats() {
        let formats = [
            CompressionFormat::None,
            CompressionFormat::RunLength,
            CompressionFormat::Huffman,
            CompressionFormat::DeltaEncoding,
            CompressionFormat::BitPacking,
        ];
        for fmt in &formats {
            let hdr = CompressionHeader { format: *fmt, original_len: 42 };
            let bytes = hdr.to_bytes();
            let decoded = CompressionHeader::from_bytes(&bytes).unwrap();
            assert_eq!(decoded.format, *fmt);
            assert_eq!(decoded.original_len, 42);
        }
    }

    #[test]
    fn header_rejects_unknown_tag() {
        let mut bytes = [0u8; 8];
        bytes[0] = 255;
        assert!(CompressionHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn header_rejects_short_buffer() {
        assert!(CompressionHeader::from_bytes(&[0u8; 4]).is_err());
    }

    // ── bitpack_ternary / unpack_ternary ──────────────────────────

    #[test]
    fn bitpack_roundtrip_basic() {
        let vals = vec![1, -1, 0, 1, -1, 0, 0, 1];
        let packed = bitpack_ternary(&vals).unwrap();
        let unpacked = unpack_ternary(&packed, vals.len()).unwrap();
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn bitpack_roundtrip_non_multiple_of_four() {
        let vals = vec![1, 0, -1];
        let packed = bitpack_ternary(&vals).unwrap();
        assert_eq!(packed.len(), 1); // ceil(3/4) = 1
        let unpacked = unpack_ternary(&packed, 3).unwrap();
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn bitpack_single_value() {
        for v in [-1i8, 0, 1] {
            let packed = bitpack_ternary(&[v]).unwrap();
            let unpacked = unpack_ternary(&packed, 1).unwrap();
            assert_eq!(unpacked, vec![v]);
        }
    }

    #[test]
    fn bitpack_empty() {
        let packed = bitpack_ternary(&[]).unwrap();
        assert!(packed.is_empty());
        let unpacked = unpack_ternary(&[], 0).unwrap();
        assert!(unpacked.is_empty());
    }

    #[test]
    fn bitpack_all_zeros() {
        let vals = vec![0i8; 16];
        let packed = bitpack_ternary(&vals).unwrap();
        assert!(packed.iter().all(|&b| b == 0));
        let unpacked = unpack_ternary(&packed, 16).unwrap();
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn bitpack_all_ones() {
        let vals = vec![1i8; 16];
        let packed = bitpack_ternary(&vals).unwrap();
        let unpacked = unpack_ternary(&packed, 16).unwrap();
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn bitpack_all_neg_ones() {
        let vals = vec![-1i8; 16];
        let packed = bitpack_ternary(&vals).unwrap();
        let unpacked = unpack_ternary(&packed, 16).unwrap();
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn bitpack_rejects_out_of_range() {
        assert!(bitpack_ternary(&[2]).is_err());
        assert!(bitpack_ternary(&[-2]).is_err());
        assert!(bitpack_ternary(&[0, 1, -1, 5]).is_err());
    }

    #[test]
    fn unpack_rejects_short_buffer() {
        assert!(unpack_ternary(&[0], 5).is_err()); // 5 values need 2 bytes
    }

    #[test]
    fn bitpack_encoding_values() {
        // Verify the specific bit patterns.
        let vals = vec![1, -1, 0, 0];
        let packed = bitpack_ternary(&vals).unwrap();
        assert_eq!(packed.len(), 1);
        let byte = packed[0];
        // 1 → 0b01 at bits 0-1, -1 → 0b10 at bits 2-3, 0 → 0b00 at 4-5, 0 → 0b00 at 6-7
        assert_eq!(byte, 0b00_00_10_01);
    }

    #[test]
    fn bitpack_large_roundtrip() {
        let vals: Vec<i8> = (0..1024)
            .map(|i| match i % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();
        let packed = bitpack_ternary(&vals).unwrap();
        assert_eq!(packed.len(), 256); // 1024 / 4
        let unpacked = unpack_ternary(&packed, vals.len()).unwrap();
        assert_eq!(unpacked, vals);
    }

    // ── delta_encode / delta_decode ───────────────────────────────

    #[test]
    fn delta_roundtrip_basic() {
        let vals = vec![0i8, 1, 1, 0, -1, -1, 0];
        let encoded = delta_encode(&vals);
        let decoded = delta_decode(&encoded);
        assert_eq!(decoded, vals);
    }

    #[test]
    fn delta_empty() {
        assert!(delta_encode(&[]).is_empty());
        assert!(delta_decode(&[]).is_empty());
    }

    #[test]
    fn delta_single_element() {
        let vals = vec![1i8];
        let encoded = delta_encode(&vals);
        assert_eq!(encoded, vec![1]);
        let decoded = delta_decode(&encoded);
        assert_eq!(decoded, vals);
    }

    #[test]
    fn delta_constant_sequence() {
        let vals = vec![1i8; 8];
        let encoded = delta_encode(&vals);
        // First element = 1, rest should be 0.
        assert_eq!(encoded[0], 1);
        assert!(encoded[1..].iter().all(|&d| d == 0));
        let decoded = delta_decode(&encoded);
        assert_eq!(decoded, vals);
    }

    #[test]
    fn delta_alternating() {
        let vals = vec![1i8, -1, 1, -1, 1, -1];
        let encoded = delta_encode(&vals);
        let decoded = delta_decode(&encoded);
        assert_eq!(decoded, vals);
    }

    #[test]
    fn delta_ascending() {
        let vals: Vec<i8> = (0..10).map(|i| (i as i8).min(5)).collect();
        let encoded = delta_encode(&vals);
        let decoded = delta_decode(&encoded);
        assert_eq!(decoded, vals);
    }

    #[test]
    fn delta_ternary_pattern() {
        let vals = vec![0i8, 0, 1, 1, 1, 0, -1, -1, 0];
        let encoded = delta_encode(&vals);
        let decoded = delta_decode(&encoded);
        assert_eq!(decoded, vals);
    }

    // ── run_length_encode / run_length_decode ─────────────────────

    #[test]
    fn rle_roundtrip_basic() {
        let vals = vec![0i8, 0, 0, 1, 1, -1];
        let runs = run_length_encode(&vals);
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0], Run { value: 0, count: 3 });
        assert_eq!(runs[1], Run { value: 1, count: 2 });
        assert_eq!(runs[2], Run { value: -1, count: 1 });
        let decoded = run_length_decode(&runs);
        assert_eq!(decoded, vals);
    }

    #[test]
    fn rle_empty() {
        let runs = run_length_encode(&[]);
        assert!(runs.is_empty());
        let decoded = run_length_decode(&[]);
        assert!(decoded.is_empty());
    }

    #[test]
    fn rle_single_value() {
        let runs = run_length_encode(&[1]);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0], Run { value: 1, count: 1 });
        let decoded = run_length_decode(&runs);
        assert_eq!(decoded, vec![1]);
    }

    #[test]
    fn rle_all_same() {
        let vals = vec![0i8; 100];
        let runs = run_length_encode(&vals);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].count, 100);
        let decoded = run_length_decode(&runs);
        assert_eq!(decoded, vals);
    }

    #[test]
    fn rle_all_different() {
        let vals = vec![1i8, -1, 0, 1, -1, 0];
        let runs = run_length_encode(&vals);
        assert_eq!(runs.len(), 6); // No consecutive duplicates.
        let decoded = run_length_decode(&runs);
        assert_eq!(decoded, vals);
    }

    #[test]
    fn rle_sparse_zeros() {
        let mut vals = vec![0i8; 50];
        vals.push(1);
        vals.extend(vec![0i8; 50]);
        let runs = run_length_encode(&vals);
        assert_eq!(runs.len(), 3);
        let decoded = run_length_decode(&runs);
        assert_eq!(decoded, vals);
    }

    #[test]
    fn rle_serialisation_roundtrip() {
        let runs = vec![
            Run { value: 0, count: 100 },
            Run { value: 1, count: 5 },
            Run { value: -1, count: 3 },
        ];
        let bytes = rle_to_bytes(&runs);
        assert_eq!(bytes.len(), 15); // 3 runs × 5 bytes
        let decoded = rle_from_bytes(&bytes).unwrap();
        assert_eq!(decoded, runs);
    }

    #[test]
    fn rle_serialisation_rejects_bad_length() {
        assert!(rle_from_bytes(&[0, 1, 2]).is_err());
    }

    // ── Huffman encode / decode ───────────────────────────────────

    #[test]
    fn huffman_roundtrip_uniform() {
        let vals: Vec<i8> = (0..30)
            .map(|i| match i % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            })
            .collect();
        let (symbols, bit_count, packed) = huffman_encode_ternary(&vals);
        let decoded = huffman_decode_ternary(&symbols, bit_count, &packed, vals.len()).unwrap();
        assert_eq!(decoded, vals);
    }

    #[test]
    fn huffman_roundtrip_skewed() {
        // 90% zeros, 8% ones, 2% neg-ones.
        let mut vals = vec![0i8; 90];
        vals.extend(vec![1i8; 8]);
        vals.extend(vec![-1i8; 2]);
        let (symbols, bit_count, packed) = huffman_encode_ternary(&vals);
        let decoded = huffman_decode_ternary(&symbols, bit_count, &packed, vals.len()).unwrap();
        assert_eq!(decoded, vals);
    }

    #[test]
    fn huffman_empty() {
        let (_, bit_count, packed) = huffman_encode_ternary(&[]);
        assert_eq!(bit_count, 0);
        assert!(packed.is_empty());
    }

    #[test]
    fn huffman_single() {
        let vals = vec![1i8];
        let (symbols, bit_count, packed) = huffman_encode_ternary(&vals);
        let decoded = huffman_decode_ternary(&symbols, bit_count, &packed, 1).unwrap();
        assert_eq!(decoded, vals);
    }

    #[test]
    fn huffman_serialisation_roundtrip() {
        let vals = vec![0i8, 0, 1, -1, 0, 0, 0, 1];
        let (symbols, bit_count, packed) = huffman_encode_ternary(&vals);
        let bytes = huffman_to_bytes(&symbols, bit_count, &packed);
        let (s2, bc2, p2) = huffman_from_bytes(&bytes).unwrap();
        assert_eq!(s2, symbols);
        assert_eq!(bc2, bit_count);
        assert_eq!(p2, packed);
    }

    #[test]
    fn huffman_deserialisation_rejects_short() {
        assert!(huffman_from_bytes(&[0, 1, 2]).is_err());
    }

    // ── compress_weights / decompress_weights ─────────────────────

    #[test]
    fn compress_none_roundtrip() {
        let vals = vec![1i8, 0, -1, 1, 0, -1];
        let compressed = compress_weights(&vals, CompressionFormat::None).unwrap();
        let decompressed = decompress_weights(&compressed).unwrap();
        assert_eq!(decompressed, vals);
    }

    #[test]
    fn compress_bitpacking_roundtrip() {
        let vals = vec![1i8, -1, 0, 0, 1, -1, 1, 0];
        let compressed = compress_weights(&vals, CompressionFormat::BitPacking).unwrap();
        let decompressed = decompress_weights(&compressed).unwrap();
        assert_eq!(decompressed, vals);
    }

    #[test]
    fn compress_rle_roundtrip() {
        let vals = vec![0i8, 0, 0, 1, 1, -1, -1, -1];
        let compressed = compress_weights(&vals, CompressionFormat::RunLength).unwrap();
        let decompressed = decompress_weights(&compressed).unwrap();
        assert_eq!(decompressed, vals);
    }

    #[test]
    fn compress_delta_roundtrip() {
        let vals = vec![0i8, 0, 1, 1, 0, -1, -1, 0];
        let compressed = compress_weights(&vals, CompressionFormat::DeltaEncoding).unwrap();
        let decompressed = decompress_weights(&compressed).unwrap();
        assert_eq!(decompressed, vals);
    }

    #[test]
    fn compress_huffman_roundtrip() {
        let vals = vec![0i8, 0, 0, 1, 0, 0, -1, 0];
        let compressed = compress_weights(&vals, CompressionFormat::Huffman).unwrap();
        let decompressed = decompress_weights(&compressed).unwrap();
        assert_eq!(decompressed, vals);
    }

    #[test]
    fn compress_empty_all_formats() {
        for fmt in [
            CompressionFormat::None,
            CompressionFormat::BitPacking,
            CompressionFormat::RunLength,
            CompressionFormat::DeltaEncoding,
            CompressionFormat::Huffman,
        ] {
            let compressed = compress_weights(&[], fmt).unwrap();
            let decompressed = decompress_weights(&compressed).unwrap();
            assert!(decompressed.is_empty(), "format {fmt:?} failed on empty");
        }
    }

    #[test]
    fn decompress_rejects_truncated_buffer() {
        assert!(decompress_weights(&[0, 0, 0]).is_err());
    }

    #[test]
    fn decompress_none_rejects_short_payload() {
        let hdr = CompressionHeader { format: CompressionFormat::None, original_len: 10 };
        let mut buf = hdr.to_bytes().to_vec();
        buf.extend([0u8; 5]); // Only 5 bytes, need 10.
        assert!(decompress_weights(&buf).is_err());
    }

    #[test]
    fn decompress_delta_rejects_short_payload() {
        let hdr = CompressionHeader { format: CompressionFormat::DeltaEncoding, original_len: 10 };
        let mut buf = hdr.to_bytes().to_vec();
        buf.extend([0u8; 3]);
        assert!(decompress_weights(&buf).is_err());
    }

    // ── compression_ratio ─────────────────────────────────────────

    #[test]
    fn ratio_empty_returns_one() {
        let r = compression_ratio(&[], CompressionFormat::BitPacking).unwrap();
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ratio_bitpacking_compresses() {
        let vals = vec![0i8; 256];
        let r = compression_ratio(&vals, CompressionFormat::BitPacking).unwrap();
        // 256 bytes → 8-byte header + 64 packed bytes = 72, ratio ≈ 3.56
        assert!(r > 3.0, "bitpacking ratio should be > 3, got {r}");
    }

    #[test]
    fn ratio_rle_sparse_compresses() {
        let vals = vec![0i8; 1000];
        let r = compression_ratio(&vals, CompressionFormat::RunLength).unwrap();
        // 1000 bytes → header(8) + 1 run(5) = 13, ratio ≈ 76.9
        assert!(r > 50.0, "RLE ratio should be > 50 for all-same, got {r}");
    }

    #[test]
    fn ratio_none_does_not_compress() {
        let vals = vec![1i8, -1, 0, 1];
        let r = compression_ratio(&vals, CompressionFormat::None).unwrap();
        // 4 bytes → header(8) + 4 = 12, ratio < 1
        assert!(r < 1.0, "None format should not compress, got {r}");
    }

    #[test]
    fn ratio_delta_constant_compresses_via_size() {
        let vals = vec![1i8; 64];
        let r_delta = compression_ratio(&vals, CompressionFormat::DeltaEncoding).unwrap();
        let r_none = compression_ratio(&vals, CompressionFormat::None).unwrap();
        // Delta is same byte count as None for i8 (both store original_len
        // bytes), but the zero deltas may help downstream compression.
        // Here we just verify both complete without error.
        assert!(r_delta > 0.0);
        assert!(r_none > 0.0);
    }

    // ── adaptive_compress ─────────────────────────────────────────

    #[test]
    fn adaptive_returns_smallest() {
        let vals = vec![0i8; 256];
        let compressed = adaptive_compress(&vals).unwrap();
        let decompressed = decompress_weights(&compressed).unwrap();
        assert_eq!(decompressed, vals);
        // RLE should win for all-zero: header(8) + 1 run(5) = 13
        assert!(
            compressed.len() < 50,
            "adaptive should pick a compact format, got {} bytes",
            compressed.len()
        );
    }

    #[test]
    fn adaptive_empty() {
        let compressed = adaptive_compress(&[]).unwrap();
        let decompressed = decompress_weights(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn adaptive_diverse_pattern() {
        let vals: Vec<i8> = (0..100)
            .map(|i| match i % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();
        let compressed = adaptive_compress(&vals).unwrap();
        let decompressed = decompress_weights(&compressed).unwrap();
        assert_eq!(decompressed, vals);
    }

    #[test]
    fn adaptive_decompresses_via_header() {
        // Verify the header self-describes the format so
        // decompress_weights works without external metadata.
        let vals = vec![1i8, -1, 0, 1, 0, -1, 1, 1];
        let compressed = adaptive_compress(&vals).unwrap();
        let hdr = CompressionHeader::from_bytes(&compressed).unwrap();
        // Any valid format is acceptable.
        assert!(
            [
                CompressionFormat::None,
                CompressionFormat::BitPacking,
                CompressionFormat::RunLength,
                CompressionFormat::DeltaEncoding,
                CompressionFormat::Huffman,
            ]
            .contains(&hdr.format)
        );
        let decompressed = decompress_weights(&compressed).unwrap();
        assert_eq!(decompressed, vals);
    }

    // ── Cross-format round-trip stress ────────────────────────────

    #[test]
    fn all_formats_roundtrip_ternary_pattern() {
        let vals: Vec<i8> = (0..128)
            .map(|i| match i % 5 {
                0 | 1 => 0,
                2 | 3 => 1,
                _ => -1,
            })
            .collect();
        for fmt in [
            CompressionFormat::None,
            CompressionFormat::BitPacking,
            CompressionFormat::RunLength,
            CompressionFormat::DeltaEncoding,
            CompressionFormat::Huffman,
        ] {
            let compressed = compress_weights(&vals, fmt).unwrap();
            let decompressed = decompress_weights(&compressed).unwrap();
            assert_eq!(decompressed, vals, "roundtrip failed for {fmt:?}");
        }
    }

    #[test]
    fn all_formats_roundtrip_single_value() {
        for v in [-1i8, 0, 1] {
            for fmt in [
                CompressionFormat::None,
                CompressionFormat::BitPacking,
                CompressionFormat::RunLength,
                CompressionFormat::DeltaEncoding,
                CompressionFormat::Huffman,
            ] {
                let compressed = compress_weights(&[v], fmt).unwrap();
                let decompressed = decompress_weights(&compressed).unwrap();
                assert_eq!(
                    decompressed,
                    vec![v],
                    "single-value roundtrip failed for {v} / {fmt:?}"
                );
            }
        }
    }

    #[test]
    fn all_formats_roundtrip_large_sparse() {
        let mut vals = vec![0i8; 500];
        vals[100] = 1;
        vals[200] = -1;
        vals[300] = 1;
        for fmt in [
            CompressionFormat::None,
            CompressionFormat::BitPacking,
            CompressionFormat::RunLength,
            CompressionFormat::DeltaEncoding,
            CompressionFormat::Huffman,
        ] {
            let compressed = compress_weights(&vals, fmt).unwrap();
            let decompressed = decompress_weights(&compressed).unwrap();
            assert_eq!(decompressed, vals, "sparse roundtrip failed for {fmt:?}");
        }
    }

    #[test]
    fn all_formats_roundtrip_dense_mixed() {
        let vals: Vec<i8> = (0..200).map(|i| [1, -1, 0, 1, -1][i % 5]).collect();
        for fmt in [
            CompressionFormat::None,
            CompressionFormat::BitPacking,
            CompressionFormat::RunLength,
            CompressionFormat::DeltaEncoding,
            CompressionFormat::Huffman,
        ] {
            let compressed = compress_weights(&vals, fmt).unwrap();
            let decompressed = decompress_weights(&compressed).unwrap();
            assert_eq!(decompressed, vals, "dense-mixed roundtrip failed for {fmt:?}");
        }
    }

    // ── Edge cases ────────────────────────────────────────────────

    #[test]
    fn bitpack_exact_multiple_of_four() {
        let vals = vec![1i8, -1, 0, 1, 0, -1, 1, -1];
        assert_eq!(vals.len() % 4, 0);
        let packed = bitpack_ternary(&vals).unwrap();
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_ternary(&packed, vals.len()).unwrap();
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn bitpack_one_more_than_multiple_of_four() {
        let vals = vec![1i8, -1, 0, 1, 0];
        assert_eq!(vals.len() % 4, 1);
        let packed = bitpack_ternary(&vals).unwrap();
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_ternary(&packed, vals.len()).unwrap();
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn delta_full_range_ternary() {
        // Worst-case delta: alternating -1 and 1 (delta = ±2).
        let vals: Vec<i8> = (0..64).map(|i| if i % 2 == 0 { -1 } else { 1 }).collect();
        let encoded = delta_encode(&vals);
        let decoded = delta_decode(&encoded);
        assert_eq!(decoded, vals);
    }

    #[test]
    fn rle_long_runs() {
        let mut vals = Vec::new();
        vals.extend(vec![1i8; 10_000]);
        vals.extend(vec![0i8; 10_000]);
        vals.extend(vec![-1i8; 10_000]);
        let runs = run_length_encode(&vals);
        assert_eq!(runs.len(), 3);
        let decoded = run_length_decode(&runs);
        assert_eq!(decoded, vals);
    }

    #[test]
    fn huffman_all_same_symbol() {
        let vals = vec![0i8; 64];
        let (symbols, bit_count, packed) = huffman_encode_ternary(&vals);
        // Most frequent = 0, each encoded as 1 bit → 64 bits.
        assert_eq!(bit_count, 64);
        let decoded = huffman_decode_ternary(&symbols, bit_count, &packed, 64).unwrap();
        assert_eq!(decoded, vals);
    }

    #[test]
    fn huffman_two_symbols_only() {
        let vals: Vec<i8> = (0..32).map(|i| if i % 2 == 0 { 0 } else { 1 }).collect();
        let (symbols, bit_count, packed) = huffman_encode_ternary(&vals);
        let decoded = huffman_decode_ternary(&symbols, bit_count, &packed, vals.len()).unwrap();
        assert_eq!(decoded, vals);
    }

    #[test]
    fn compression_preserves_ternary_values() {
        let vals: Vec<i8> = (0..256)
            .map(|i| match i % 7 {
                0 | 1 | 2 => 0,
                3 | 4 => 1,
                _ => -1,
            })
            .collect();
        for fmt in [
            CompressionFormat::None,
            CompressionFormat::BitPacking,
            CompressionFormat::RunLength,
            CompressionFormat::DeltaEncoding,
            CompressionFormat::Huffman,
        ] {
            let compressed = compress_weights(&vals, fmt).unwrap();
            let decompressed = decompress_weights(&compressed).unwrap();
            assert_ternary(&decompressed);
            assert_eq!(decompressed, vals, "ternary check failed for {fmt:?}");
        }
    }

    #[test]
    fn adaptive_picks_rle_for_sparse() {
        let vals = vec![0i8; 1000];
        let compressed = adaptive_compress(&vals).unwrap();
        let hdr = CompressionHeader::from_bytes(&compressed).unwrap();
        // RLE should be the best for all-same: 8 + 5 = 13 bytes
        assert_eq!(hdr.format, CompressionFormat::RunLength);
    }

    #[test]
    fn adaptive_picks_compact_for_dense() {
        // Dense ternary with no long runs → bitpacking or huffman should
        // beat None and RLE.
        let vals: Vec<i8> = (0..256).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let compressed = adaptive_compress(&vals).unwrap();
        let hdr = CompressionHeader::from_bytes(&compressed).unwrap();
        assert!(
            matches!(hdr.format, CompressionFormat::BitPacking | CompressionFormat::Huffman),
            "expected BitPacking or Huffman, got {:?}",
            hdr.format
        );
        assert!(compressed.len() < vals.len());
        let decompressed = decompress_weights(&compressed).unwrap();
        assert_eq!(decompressed, vals);
    }

    #[test]
    fn rle_preserves_alternating_long_runs() {
        let mut vals = Vec::new();
        for v in [0i8, 1, -1] {
            vals.extend(vec![v; 200]);
        }
        let runs = run_length_encode(&vals);
        assert_eq!(runs.len(), 3);
        for r in &runs {
            assert_eq!(r.count, 200);
        }
        let decoded = run_length_decode(&runs);
        assert_eq!(decoded, vals);
    }

    #[test]
    fn bitpack_two_values() {
        let vals = vec![1i8, -1];
        let packed = bitpack_ternary(&vals).unwrap();
        let unpacked = unpack_ternary(&packed, 2).unwrap();
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn delta_encode_preserves_length() {
        for len in [0, 1, 2, 7, 8, 15, 16, 100] {
            let vals: Vec<i8> = (0..len).map(|i| (i % 3 - 1) as i8).collect();
            let encoded = delta_encode(&vals);
            assert_eq!(
                encoded.len(),
                vals.len(),
                "delta_encode changed length for input of size {len}"
            );
        }
    }

    #[test]
    fn huffman_decoder_rejects_short_symbol_table() {
        assert!(huffman_decode_ternary(&[0, 1], 0, &[], 0).is_err());
    }

    #[test]
    fn compress_decompress_rle_rejects_bad_decoded_len() {
        // Manually craft a buffer with header claiming 100 elements but
        // payload encoding only 1.
        let hdr = CompressionHeader { format: CompressionFormat::RunLength, original_len: 100 };
        let mut buf = hdr.to_bytes().to_vec();
        let runs = vec![Run { value: 0, count: 1 }];
        buf.extend_from_slice(&rle_to_bytes(&runs));
        assert!(decompress_weights(&buf).is_err());
    }

    #[test]
    fn all_formats_roundtrip_power_of_two_sizes() {
        for exp in 0..8u32 {
            let len = 2usize.pow(exp);
            let vals: Vec<i8> = (0..len).map(|i| (i % 3) as i8 - 1).collect();
            for fmt in [
                CompressionFormat::None,
                CompressionFormat::BitPacking,
                CompressionFormat::RunLength,
                CompressionFormat::DeltaEncoding,
                CompressionFormat::Huffman,
            ] {
                let compressed = compress_weights(&vals, fmt).unwrap();
                let decompressed = decompress_weights(&compressed).unwrap();
                assert_eq!(
                    decompressed, vals,
                    "power-of-two roundtrip failed for len={len}, {fmt:?}"
                );
            }
        }
    }
}
