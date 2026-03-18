//! Model format conversion for optimized GPU tensor layouts.
//!
//! Provides CPU reference implementations for converting tensors between
//! memory layouts (row-major, column-major, blocked, tiled) and packing
//! multiple tensors into aligned GPU buffers. Designed for Intel Arc A770
//! alignment requirements (64-byte / 128-byte CL line).
//!
//! No OpenCL runtime is required — all operations run on the CPU.

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// TensorLayout
// ---------------------------------------------------------------------------

/// Memory layout for a 2-D tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorLayout {
    /// Standard C-order: elements in the same row are contiguous.
    RowMajor,
    /// Fortran-order: elements in the same column are contiguous.
    ColumnMajor,
    /// Blocked layout with `(block_r, block_c)` tile sizes.
    /// Data is stored as row-major blocks of row-major elements.
    BlockedRC(usize, usize),
    /// Interleaved: row 0 element 0, row 1 element 0, row 0 element 1, …
    /// Groups of `factor` rows are interleaved element-wise.
    Interleaved,
}

impl fmt::Display for TensorLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowMajor => write!(f, "RowMajor"),
            Self::ColumnMajor => write!(f, "ColumnMajor"),
            Self::BlockedRC(br, bc) => write!(f, "Blocked({br}×{bc})"),
            Self::Interleaved => write!(f, "Interleaved"),
        }
    }
}

// ---------------------------------------------------------------------------
// AlignmentRequirements
// ---------------------------------------------------------------------------

/// Device-specific alignment constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlignmentRequirements {
    /// Minimum buffer start alignment in bytes.
    pub buffer_align: usize,
    /// Cache-line size in bytes (optimal access granularity).
    pub cache_line: usize,
}

impl AlignmentRequirements {
    /// Intel Arc A770 defaults: 64-byte buffer alignment, 128-byte cache line.
    pub const A770: Self = Self { buffer_align: 64, cache_line: 128 };

    /// Generic GPU defaults: 256-byte buffer alignment, 64-byte cache line.
    pub const GENERIC: Self = Self { buffer_align: 256, cache_line: 64 };

    /// Round `size` up to the next multiple of `self.buffer_align`.
    #[inline]
    pub fn align_up(&self, size: usize) -> usize {
        let mask = self.buffer_align - 1;
        (size + mask) & !mask
    }

    /// Round `size` up to the next cache-line boundary.
    #[inline]
    pub fn cache_align(&self, size: usize) -> usize {
        let mask = self.cache_line - 1;
        (size + mask) & !mask
    }

    /// Validate that alignment values are powers of two.
    pub fn validate(&self) -> Result<(), String> {
        if !self.buffer_align.is_power_of_two() {
            return Err(format!("buffer_align ({}) must be a power of two", self.buffer_align));
        }
        if !self.cache_line.is_power_of_two() {
            return Err(format!("cache_line ({}) must be a power of two", self.cache_line));
        }
        Ok(())
    }
}

impl Default for AlignmentRequirements {
    fn default() -> Self {
        Self::A770
    }
}

impl fmt::Display for AlignmentRequirements {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "align={}B, CL={}B", self.buffer_align, self.cache_line)
    }
}

// ---------------------------------------------------------------------------
// LayoutConverter — CPU reference
// ---------------------------------------------------------------------------

/// Converts f32 tensor data between any two [`TensorLayout`] variants.
///
/// All conversions go through row-major as the canonical intermediate form.
#[derive(Debug, Clone, Copy)]
pub struct LayoutConverter;

impl LayoutConverter {
    // -- public API --------------------------------------------------------

    /// Convert `src` from `src_layout` to `dst_layout`.
    ///
    /// `rows` × `cols` describes the *logical* tensor shape.
    pub fn convert(
        src: &[f32],
        rows: usize,
        cols: usize,
        src_layout: TensorLayout,
        dst_layout: TensorLayout,
    ) -> Vec<f32> {
        assert_eq!(
            src.len(),
            Self::required_len(rows, cols, src_layout),
            "source length mismatch for {rows}×{cols} {src_layout}"
        );
        if src_layout == dst_layout {
            return src.to_vec();
        }
        // Canonical path: src → row-major → dst
        let row_major = Self::to_row_major(src, rows, cols, src_layout);
        Self::from_row_major(&row_major, rows, cols, dst_layout)
    }

    /// Minimum element count required for a given layout.
    pub fn required_len(rows: usize, cols: usize, layout: TensorLayout) -> usize {
        match layout {
            TensorLayout::RowMajor | TensorLayout::ColumnMajor | TensorLayout::Interleaved => {
                rows * cols
            }
            TensorLayout::BlockedRC(br, bc) => {
                let padded_r = rows.div_ceil(br) * br;
                let padded_c = cols.div_ceil(bc) * bc;
                padded_r * padded_c
            }
        }
    }

    // -- row-major ↔ other -------------------------------------------------

    fn to_row_major(src: &[f32], rows: usize, cols: usize, layout: TensorLayout) -> Vec<f32> {
        match layout {
            TensorLayout::RowMajor => src.to_vec(),
            TensorLayout::ColumnMajor => {
                let mut out = vec![0.0f32; rows * cols];
                for r in 0..rows {
                    for c in 0..cols {
                        out[r * cols + c] = src[c * rows + r];
                    }
                }
                out
            }
            TensorLayout::BlockedRC(br, bc) => {
                let padded_r = rows.div_ceil(br) * br;
                let padded_c = cols.div_ceil(bc) * bc;
                let blocks_per_row = padded_c / bc;
                let mut out = vec![0.0f32; rows * cols];
                for r in 0..rows {
                    for c in 0..cols {
                        let block_row = r / br;
                        let block_col = c / bc;
                        let in_block_r = r % br;
                        let in_block_c = c % bc;
                        let block_idx = block_row * blocks_per_row + block_col;
                        let elem_idx = block_idx * (br * bc) + in_block_r * bc + in_block_c;
                        out[r * cols + c] = src[elem_idx];
                    }
                }
                let _ = padded_r; // suppress unused warning
                out
            }
            TensorLayout::Interleaved => {
                // Interleaved: for each column, rows are packed sequentially.
                // index(r, c) = c * rows + r  (same indexing as col-major)
                let mut out = vec![0.0f32; rows * cols];
                for r in 0..rows {
                    for c in 0..cols {
                        out[r * cols + c] = src[c * rows + r];
                    }
                }
                out
            }
        }
    }

    fn from_row_major(src: &[f32], rows: usize, cols: usize, layout: TensorLayout) -> Vec<f32> {
        match layout {
            TensorLayout::RowMajor => src.to_vec(),
            TensorLayout::ColumnMajor => {
                let mut out = vec![0.0f32; rows * cols];
                for r in 0..rows {
                    for c in 0..cols {
                        out[c * rows + r] = src[r * cols + c];
                    }
                }
                out
            }
            TensorLayout::BlockedRC(br, bc) => {
                let padded_r = rows.div_ceil(br) * br;
                let padded_c = cols.div_ceil(bc) * bc;
                let blocks_per_row = padded_c / bc;
                let total = padded_r * padded_c;
                let mut out = vec![0.0f32; total];
                for r in 0..padded_r {
                    for c in 0..padded_c {
                        let block_row = r / br;
                        let block_col = c / bc;
                        let in_block_r = r % br;
                        let in_block_c = c % bc;
                        let block_idx = block_row * blocks_per_row + block_col;
                        let elem_idx = block_idx * (br * bc) + in_block_r * bc + in_block_c;
                        let val = if r < rows && c < cols {
                            src[r * cols + c]
                        } else {
                            0.0 // zero-pad
                        };
                        out[elem_idx] = val;
                    }
                }
                out
            }
            TensorLayout::Interleaved => {
                let mut out = vec![0.0f32; rows * cols];
                for r in 0..rows {
                    for c in 0..cols {
                        out[c * rows + r] = src[r * cols + c];
                    }
                }
                out
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TileFormat
// ---------------------------------------------------------------------------

/// Tile configuration for GPU-optimal 2-D access patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileFormat {
    /// Tile height (rows per tile).
    pub tile_rows: usize,
    /// Tile width (columns per tile).
    pub tile_cols: usize,
}

impl TileFormat {
    pub const TILE_16X16: Self = Self { tile_rows: 16, tile_cols: 16 };
    pub const TILE_32X32: Self = Self { tile_rows: 32, tile_cols: 32 };
    pub const TILE_8X8: Self = Self { tile_rows: 8, tile_cols: 8 };

    /// Create a custom tile format.
    pub fn new(tile_rows: usize, tile_cols: usize) -> Self {
        assert!(tile_rows > 0 && tile_cols > 0, "tile dims must be > 0");
        Self { tile_rows, tile_cols }
    }

    /// Number of tiles needed to cover `rows` × `cols`.
    pub fn tile_count(&self, rows: usize, cols: usize) -> (usize, usize) {
        (rows.div_ceil(self.tile_rows), cols.div_ceil(self.tile_cols))
    }

    /// Padded dimensions (multiple of tile size).
    pub fn padded_dims(&self, rows: usize, cols: usize) -> (usize, usize) {
        let (tr, tc) = self.tile_count(rows, cols);
        (tr * self.tile_rows, tc * self.tile_cols)
    }

    /// Total element count in the tiled buffer (includes padding).
    pub fn tiled_len(&self, rows: usize, cols: usize) -> usize {
        let (pr, pc) = self.padded_dims(rows, cols);
        pr * pc
    }

    /// Convert a row-major buffer into tiled layout with zero-padding.
    pub fn tile(&self, src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        assert!(src.len() >= rows * cols, "source too short: {} < {}", src.len(), rows * cols);
        let (tile_rows_count, tile_cols_count) = self.tile_count(rows, cols);
        let total = self.tiled_len(rows, cols);
        let mut out = vec![0.0f32; total];
        let tile_area = self.tile_rows * self.tile_cols;

        for tr in 0..tile_rows_count {
            for tc in 0..tile_cols_count {
                let tile_idx = tr * tile_cols_count + tc;
                let base = tile_idx * tile_area;
                for lr in 0..self.tile_rows {
                    for lc in 0..self.tile_cols {
                        let gr = tr * self.tile_rows + lr;
                        let gc = tc * self.tile_cols + lc;
                        let val = if gr < rows && gc < cols { src[gr * cols + gc] } else { 0.0 };
                        out[base + lr * self.tile_cols + lc] = val;
                    }
                }
            }
        }
        out
    }

    /// Convert a tiled buffer back to row-major, stripping padding.
    pub fn untile(&self, tiled: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let (tile_rows_count, tile_cols_count) = self.tile_count(rows, cols);
        let tile_area = self.tile_rows * self.tile_cols;
        let expected = tile_rows_count * tile_cols_count * tile_area;
        assert!(tiled.len() >= expected, "tiled buffer too short: {} < {expected}", tiled.len());

        let mut out = vec![0.0f32; rows * cols];
        for tr in 0..tile_rows_count {
            for tc in 0..tile_cols_count {
                let tile_idx = tr * tile_cols_count + tc;
                let base = tile_idx * tile_area;
                for lr in 0..self.tile_rows {
                    for lc in 0..self.tile_cols {
                        let gr = tr * self.tile_rows + lr;
                        let gc = tc * self.tile_cols + lc;
                        if gr < rows && gc < cols {
                            out[gr * cols + gc] = tiled[base + lr * self.tile_cols + lc];
                        }
                    }
                }
            }
        }
        out
    }
}

impl fmt::Display for TileFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tile({}×{})", self.tile_rows, self.tile_cols)
    }
}

// ---------------------------------------------------------------------------
// PackedFormat — multi-tensor packing with alignment
// ---------------------------------------------------------------------------

/// Descriptor for one tensor inside a packed buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedEntry {
    /// Tensor name / identifier.
    pub name: String,
    /// Byte offset into the packed buffer.
    pub offset: usize,
    /// Size in bytes (before alignment padding).
    pub size: usize,
    /// Aligned size (including trailing padding).
    pub aligned_size: usize,
}

/// Packs multiple tensor byte slices into a single aligned buffer.
#[derive(Debug, Clone)]
pub struct PackedFormat {
    /// Alignment requirements used for packing.
    pub alignment: AlignmentRequirements,
    /// Metadata for each packed tensor.
    pub entries: Vec<PackedEntry>,
    /// Total packed buffer size.
    pub total_size: usize,
}

impl PackedFormat {
    /// Plan the packing layout for a list of `(name, byte_len)` entries.
    pub fn plan(tensors: &[(&str, usize)], alignment: AlignmentRequirements) -> Self {
        let mut entries = Vec::with_capacity(tensors.len());
        let mut offset = 0usize;
        for &(name, size) in tensors {
            let aligned_size = alignment.align_up(size);
            entries.push(PackedEntry { name: name.to_string(), offset, size, aligned_size });
            offset += aligned_size;
        }
        Self { alignment, entries, total_size: offset }
    }

    /// Pack raw byte slices into a single buffer according to the plan.
    pub fn pack(&self, data: &[&[u8]]) -> Vec<u8> {
        assert_eq!(data.len(), self.entries.len(), "data slice count must match entry count");
        let mut buf = vec![0u8; self.total_size];
        for (entry, &src) in self.entries.iter().zip(data.iter()) {
            assert!(
                src.len() <= entry.size,
                "tensor '{}' data ({}) exceeds planned size ({})",
                entry.name,
                src.len(),
                entry.size,
            );
            buf[entry.offset..entry.offset + src.len()].copy_from_slice(src);
        }
        buf
    }

    /// Unpack a single tensor from the packed buffer by index.
    pub fn unpack<'a>(&self, buf: &'a [u8], index: usize) -> &'a [u8] {
        let entry = &self.entries[index];
        &buf[entry.offset..entry.offset + entry.size]
    }

    /// Look up an entry by name, returning its index.
    pub fn find(&self, name: &str) -> Option<usize> {
        self.entries.iter().position(|e| e.name == name)
    }

    /// Wasted bytes due to alignment padding.
    pub fn padding_overhead(&self) -> usize {
        self.entries.iter().map(|e| e.aligned_size - e.size).sum()
    }
}

impl fmt::Display for PackedFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Packed({} tensors, {}B, {}B padding)",
            self.entries.len(),
            self.total_size,
            self.padding_overhead(),
        )
    }
}

// ---------------------------------------------------------------------------
// QuantFormatAdapter — I2_S / QK256 → GPU layout
// ---------------------------------------------------------------------------

/// Supported source quantization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    /// BitNet I2_S: 4 ternary values per byte (2 bits each).
    I2S,
    /// QK256: 256-element blocks with per-block scale.
    Qk256,
}

impl fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I2S => write!(f, "I2_S"),
            Self::Qk256 => write!(f, "QK256"),
        }
    }
}

/// Result of adapting quantized data for GPU layout.
#[derive(Debug, Clone)]
pub struct AdaptedQuant {
    /// Dequantized f32 values (or repacked bytes for GPU-native path).
    pub data: Vec<f32>,
    /// Source format that was adapted.
    pub source_format: QuantFormat,
    /// Number of logical elements.
    pub element_count: usize,
    /// Number of source bytes consumed.
    pub source_bytes: usize,
}

/// Adapts quantized tensor blocks to GPU-friendly f32 layout.
///
/// CPU reference only — dequantizes to f32 for layout conversion.
#[derive(Debug, Clone, Copy)]
pub struct QuantFormatAdapter;

impl QuantFormatAdapter {
    /// Dequantize I2_S packed bytes to f32.
    ///
    /// Encoding: 2-bit pairs `0b00 → −1`, `0b01 → 0`, `0b10 → +1`.
    pub fn dequant_i2s(packed: &[u8], count: usize) -> AdaptedQuant {
        let mut data = Vec::with_capacity(count);
        for i in 0..count {
            let byte_idx = i / 4;
            let bit_pos = (i % 4) * 2;
            let bits = if byte_idx < packed.len() {
                (packed[byte_idx] >> bit_pos) & 0x03
            } else {
                1 // default to 0
            };
            data.push((bits as f32) - 1.0);
        }
        AdaptedQuant {
            data,
            source_format: QuantFormat::I2S,
            element_count: count,
            source_bytes: packed.len(),
        }
    }

    /// Dequantize QK256 blocks to f32.
    ///
    /// Each 256-element block is stored as: `[f32 scale][64 packed I2_S bytes]`.
    /// Block size: 4 + 64 = 68 bytes → 256 ternary values.
    pub fn dequant_qk256(packed: &[u8], block_count: usize) -> AdaptedQuant {
        const BLOCK_ELEMS: usize = 256;
        const SCALE_BYTES: usize = 4;
        const DATA_BYTES: usize = BLOCK_ELEMS / 4; // 64
        const BLOCK_BYTES: usize = SCALE_BYTES + DATA_BYTES; // 68

        let element_count = block_count * BLOCK_ELEMS;
        let mut data = Vec::with_capacity(element_count);

        for b in 0..block_count {
            let block_start = b * BLOCK_BYTES;
            let scale = if block_start + SCALE_BYTES <= packed.len() {
                let bytes: [u8; 4] =
                    packed[block_start..block_start + SCALE_BYTES].try_into().unwrap();
                f32::from_le_bytes(bytes)
            } else {
                1.0
            };
            let data_start = block_start + SCALE_BYTES;
            for i in 0..BLOCK_ELEMS {
                let byte_idx = data_start + i / 4;
                let bit_pos = (i % 4) * 2;
                let bits =
                    if byte_idx < packed.len() { (packed[byte_idx] >> bit_pos) & 0x03 } else { 1 };
                let ternary = (bits as f32) - 1.0;
                data.push(ternary * scale);
            }
        }
        AdaptedQuant {
            data,
            source_format: QuantFormat::Qk256,
            element_count,
            source_bytes: block_count * BLOCK_BYTES,
        }
    }

    /// Adapt any supported quant format to f32.
    pub fn adapt(packed: &[u8], format: QuantFormat, element_count: usize) -> AdaptedQuant {
        match format {
            QuantFormat::I2S => Self::dequant_i2s(packed, element_count),
            QuantFormat::Qk256 => {
                let block_count = element_count.div_ceil(256);
                Self::dequant_qk256(packed, block_count)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ConversionStep / ConversionPlan
// ---------------------------------------------------------------------------

/// A single step in a multi-step conversion pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum ConversionStep {
    /// Dequantize from a packed format to f32.
    Dequantize(QuantFormat),
    /// Change layout (row-major ↔ column-major, blocked, interleaved).
    Reorder(TensorLayout, TensorLayout),
    /// Tile into GPU-friendly blocks.
    Tile(TileFormat),
    /// Pad rows/cols to alignment boundary (in elements).
    Pad(usize, usize),
}

impl fmt::Display for ConversionStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dequantize(qf) => write!(f, "Dequant({qf})"),
            Self::Reorder(from, to) => write!(f, "Reorder({from}→{to})"),
            Self::Tile(tf) => write!(f, "{tf}"),
            Self::Pad(r, c) => write!(f, "Pad({r}×{c})"),
        }
    }
}

/// A planned sequence of conversion steps for a tensor.
#[derive(Debug, Clone)]
pub struct ConversionPlan {
    /// Logical tensor shape (rows, cols).
    pub shape: (usize, usize),
    /// Ordered steps to execute.
    pub steps: Vec<ConversionStep>,
}

impl ConversionPlan {
    /// Build a new plan for the given shape.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { shape: (rows, cols), steps: Vec::new() }
    }

    /// Append a dequantization step.
    pub fn dequantize(mut self, format: QuantFormat) -> Self {
        self.steps.push(ConversionStep::Dequantize(format));
        self
    }

    /// Append a layout reorder step.
    pub fn reorder(mut self, from: TensorLayout, to: TensorLayout) -> Self {
        self.steps.push(ConversionStep::Reorder(from, to));
        self
    }

    /// Append a tiling step.
    pub fn tile(mut self, format: TileFormat) -> Self {
        self.steps.push(ConversionStep::Tile(format));
        self
    }

    /// Append a padding step (rows, cols to pad to).
    pub fn pad(mut self, target_rows: usize, target_cols: usize) -> Self {
        self.steps.push(ConversionStep::Pad(target_rows, target_cols));
        self
    }

    /// Execute the plan on quantized input data, returning f32 output.
    pub fn execute(&self, quant_data: Option<&[u8]>, f32_data: Option<&[f32]>) -> ConversionResult {
        let start = Instant::now();
        let (rows, cols) = self.shape;
        let mut current: Vec<f32>;
        let mut current_rows = rows;
        let mut current_cols = cols;
        let mut source_bytes = 0usize;

        // Determine initial data
        if let Some(f) = f32_data {
            current = f.to_vec();
        } else {
            current = Vec::new();
        }

        for step in &self.steps {
            match step {
                ConversionStep::Dequantize(format) => {
                    let packed = quant_data.expect("dequant step requires quant_data");
                    source_bytes = packed.len();
                    let adapted = QuantFormatAdapter::adapt(packed, *format, rows * cols);
                    current = adapted.data;
                    current_rows = rows;
                    current_cols = cols;
                }
                ConversionStep::Reorder(from, to) => {
                    current =
                        LayoutConverter::convert(&current, current_rows, current_cols, *from, *to);
                    // BlockedRC may expand buffer size
                    let new_len = LayoutConverter::required_len(current_rows, current_cols, *to);
                    let _ = new_len;
                }
                ConversionStep::Tile(tf) => {
                    current = tf.tile(&current, current_rows, current_cols);
                    let (pr, pc) = tf.padded_dims(current_rows, current_cols);
                    current_rows = pr;
                    current_cols = pc;
                }
                ConversionStep::Pad(target_r, target_c) => {
                    current =
                        Self::pad_data(&current, current_rows, current_cols, *target_r, *target_c);
                    current_rows = *target_r;
                    current_cols = *target_c;
                }
            }
        }

        let elapsed = start.elapsed();
        let output_bytes = current.len() * std::mem::size_of::<f32>();
        ConversionResult {
            data: current,
            rows: current_rows,
            cols: current_cols,
            stats: ConversionStats {
                source_bytes: if source_bytes > 0 {
                    source_bytes
                } else if let Some(f) = f32_data {
                    f.len() * 4
                } else {
                    0
                },
                output_bytes,
                elapsed_us: elapsed.as_micros() as u64,
                steps_executed: self.steps.len(),
            },
        }
    }

    fn pad_data(
        src: &[f32],
        rows: usize,
        cols: usize,
        target_r: usize,
        target_c: usize,
    ) -> Vec<f32> {
        let tr = target_r.max(rows);
        let tc = target_c.max(cols);
        let mut out = vec![0.0f32; tr * tc];
        for r in 0..rows {
            for c in 0..cols {
                out[r * tc + c] = src[r * cols + c];
            }
        }
        out
    }

    /// Number of steps in the plan.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the plan is empty (no steps).
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

impl fmt::Display for ConversionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (r, c) = self.shape;
        write!(f, "Plan({r}×{c}: ")?;
        for (i, step) in self.steps.iter().enumerate() {
            if i > 0 {
                write!(f, " → ")?;
            }
            write!(f, "{step}")?;
        }
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// ConversionResult / ConversionStats
// ---------------------------------------------------------------------------

/// Statistics from a conversion execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConversionStats {
    /// Bytes consumed from the source.
    pub source_bytes: usize,
    /// Bytes produced in the output.
    pub output_bytes: usize,
    /// Wall-clock time in microseconds.
    pub elapsed_us: u64,
    /// Number of pipeline steps executed.
    pub steps_executed: usize,
}

impl ConversionStats {
    /// Compression ratio: source_bytes / output_bytes.
    /// Returns 0.0 if output_bytes is zero.
    pub fn compression_ratio(&self) -> f64 {
        if self.output_bytes == 0 {
            return 0.0;
        }
        self.source_bytes as f64 / self.output_bytes as f64
    }

    /// Throughput in MB/s (based on output bytes).
    pub fn throughput_mbps(&self) -> f64 {
        if self.elapsed_us == 0 {
            return 0.0;
        }
        (self.output_bytes as f64) / (self.elapsed_us as f64) // bytes/µs == MB/s
    }
}

impl fmt::Display for ConversionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}B→{}B ({} steps, {:.1}µs, ratio={:.3})",
            self.source_bytes,
            self.output_bytes,
            self.steps_executed,
            self.elapsed_us,
            self.compression_ratio(),
        )
    }
}

/// Output of executing a [`ConversionPlan`].
#[derive(Debug, Clone)]
pub struct ConversionResult {
    /// Converted f32 data.
    pub data: Vec<f32>,
    /// Output rows (may differ from input due to padding/tiling).
    pub rows: usize,
    /// Output columns (may differ from input due to padding/tiling).
    pub cols: usize,
    /// Conversion statistics.
    pub stats: ConversionStats,
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // TensorLayout Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_layout_display() {
        assert_eq!(TensorLayout::RowMajor.to_string(), "RowMajor");
        assert_eq!(TensorLayout::ColumnMajor.to_string(), "ColumnMajor");
        assert_eq!(TensorLayout::BlockedRC(4, 4).to_string(), "Blocked(4×4)");
        assert_eq!(TensorLayout::Interleaved.to_string(), "Interleaved");
    }

    // -----------------------------------------------------------------------
    // AlignmentRequirements
    // -----------------------------------------------------------------------

    #[test]
    fn test_alignment_a770_defaults() {
        let a = AlignmentRequirements::A770;
        assert_eq!(a.buffer_align, 64);
        assert_eq!(a.cache_line, 128);
    }

    #[test]
    fn test_alignment_generic() {
        let a = AlignmentRequirements::GENERIC;
        assert_eq!(a.buffer_align, 256);
        assert_eq!(a.cache_line, 64);
    }

    #[test]
    fn test_align_up_exact() {
        let a = AlignmentRequirements::A770;
        assert_eq!(a.align_up(64), 64);
        assert_eq!(a.align_up(128), 128);
    }

    #[test]
    fn test_align_up_rounds() {
        let a = AlignmentRequirements::A770;
        assert_eq!(a.align_up(1), 64);
        assert_eq!(a.align_up(65), 128);
        assert_eq!(a.align_up(100), 128);
    }

    #[test]
    fn test_align_up_zero() {
        let a = AlignmentRequirements::A770;
        assert_eq!(a.align_up(0), 0);
    }

    #[test]
    fn test_cache_align() {
        let a = AlignmentRequirements::A770;
        assert_eq!(a.cache_align(1), 128);
        assert_eq!(a.cache_align(128), 128);
        assert_eq!(a.cache_align(129), 256);
    }

    #[test]
    fn test_alignment_validate_ok() {
        assert!(AlignmentRequirements::A770.validate().is_ok());
        assert!(AlignmentRequirements::GENERIC.validate().is_ok());
    }

    #[test]
    fn test_alignment_validate_err() {
        let bad = AlignmentRequirements { buffer_align: 3, cache_line: 128 };
        assert!(bad.validate().is_err());
        let bad2 = AlignmentRequirements { buffer_align: 64, cache_line: 7 };
        assert!(bad2.validate().is_err());
    }

    #[test]
    fn test_alignment_display() {
        let a = AlignmentRequirements::A770;
        assert_eq!(a.to_string(), "align=64B, CL=128B");
    }

    // -----------------------------------------------------------------------
    // LayoutConverter — row ↔ column major
    // -----------------------------------------------------------------------

    #[test]
    fn test_row_to_col_major_2x3() {
        // Row-major: [[1,2,3],[4,5,6]]
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let col =
            LayoutConverter::convert(&src, 2, 3, TensorLayout::RowMajor, TensorLayout::ColumnMajor);
        // Column-major: [1,4, 2,5, 3,6]
        assert_eq!(col, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_col_to_row_major_2x3() {
        let col = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let row =
            LayoutConverter::convert(&col, 2, 3, TensorLayout::ColumnMajor, TensorLayout::RowMajor);
        assert_eq!(row, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_row_col_roundtrip() {
        let src: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let col =
            LayoutConverter::convert(&src, 4, 5, TensorLayout::RowMajor, TensorLayout::ColumnMajor);
        let back =
            LayoutConverter::convert(&col, 4, 5, TensorLayout::ColumnMajor, TensorLayout::RowMajor);
        assert_eq!(src, back);
    }

    #[test]
    fn test_row_col_roundtrip_1x1() {
        let src = [42.0];
        let col =
            LayoutConverter::convert(&src, 1, 1, TensorLayout::RowMajor, TensorLayout::ColumnMajor);
        assert_eq!(col, vec![42.0]);
        let back =
            LayoutConverter::convert(&col, 1, 1, TensorLayout::ColumnMajor, TensorLayout::RowMajor);
        assert_eq!(back, vec![42.0]);
    }

    #[test]
    fn test_same_layout_noop() {
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out =
            LayoutConverter::convert(&src, 3, 4, TensorLayout::RowMajor, TensorLayout::RowMajor);
        assert_eq!(src, out);
    }

    #[test]
    fn test_row_col_single_row() {
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let col =
            LayoutConverter::convert(&src, 1, 4, TensorLayout::RowMajor, TensorLayout::ColumnMajor);
        // 1-row matrix: column-major is identical
        assert_eq!(col, src);
    }

    #[test]
    fn test_row_col_single_col() {
        let src = vec![1.0, 2.0, 3.0];
        let col =
            LayoutConverter::convert(&src, 3, 1, TensorLayout::RowMajor, TensorLayout::ColumnMajor);
        // 1-col matrix: column-major is identical
        assert_eq!(col, src);
    }

    // -----------------------------------------------------------------------
    // LayoutConverter — blocked layout
    // -----------------------------------------------------------------------

    #[test]
    fn test_blocked_exact_fit() {
        // 4×4 matrix with 2×2 blocks → exact fit
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let blocked = LayoutConverter::convert(
            &src,
            4,
            4,
            TensorLayout::RowMajor,
            TensorLayout::BlockedRC(2, 2),
        );
        // Blocks: [0,1,4,5], [2,3,6,7], [8,9,12,13], [10,11,14,15]
        assert_eq!(
            blocked,
            vec![
                0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 12.0, 13.0, 10.0, 11.0, 14.0,
                15.0,
            ]
        );
    }

    #[test]
    fn test_blocked_roundtrip_exact() {
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let blocked = LayoutConverter::convert(
            &src,
            4,
            4,
            TensorLayout::RowMajor,
            TensorLayout::BlockedRC(2, 2),
        );
        let back = LayoutConverter::convert(
            &blocked,
            4,
            4,
            TensorLayout::BlockedRC(2, 2),
            TensorLayout::RowMajor,
        );
        assert_eq!(src, back);
    }

    #[test]
    fn test_blocked_non_divisible() {
        // 3×3 matrix with 2×2 blocks → padded to 4×4
        let src: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let blocked = LayoutConverter::convert(
            &src,
            3,
            3,
            TensorLayout::RowMajor,
            TensorLayout::BlockedRC(2, 2),
        );
        // Padded to 4×4 with zeros, then blocked
        assert_eq!(blocked.len(), 16); // 4×4
        // Roundtrip
        let back = LayoutConverter::convert(
            &blocked,
            3,
            3,
            TensorLayout::BlockedRC(2, 2),
            TensorLayout::RowMajor,
        );
        assert_eq!(back, src);
    }

    #[test]
    fn test_blocked_1x1_block() {
        // BlockedRC(1,1) = row-major
        let src: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let blocked = LayoutConverter::convert(
            &src,
            2,
            3,
            TensorLayout::RowMajor,
            TensorLayout::BlockedRC(1, 1),
        );
        assert_eq!(blocked, src);
    }

    #[test]
    fn test_blocked_required_len() {
        assert_eq!(LayoutConverter::required_len(3, 3, TensorLayout::BlockedRC(2, 2)), 16);
        assert_eq!(LayoutConverter::required_len(4, 4, TensorLayout::BlockedRC(2, 2)), 16);
        assert_eq!(LayoutConverter::required_len(5, 5, TensorLayout::BlockedRC(4, 4)), 64);
    }

    #[test]
    fn test_blocked_large_roundtrip() {
        let src: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let blocked = LayoutConverter::convert(
            &src,
            10,
            10,
            TensorLayout::RowMajor,
            TensorLayout::BlockedRC(4, 4),
        );
        let back = LayoutConverter::convert(
            &blocked,
            10,
            10,
            TensorLayout::BlockedRC(4, 4),
            TensorLayout::RowMajor,
        );
        assert_eq!(src, back);
    }

    // -----------------------------------------------------------------------
    // LayoutConverter — interleaved
    // -----------------------------------------------------------------------

    #[test]
    fn test_interleaved_2x3() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3 row-major
        let inter =
            LayoutConverter::convert(&src, 2, 3, TensorLayout::RowMajor, TensorLayout::Interleaved);
        // Interleaved: col0=[1,4], col1=[2,5], col2=[3,6]
        assert_eq!(inter, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_interleaved_roundtrip() {
        let src: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let inter =
            LayoutConverter::convert(&src, 4, 6, TensorLayout::RowMajor, TensorLayout::Interleaved);
        let back = LayoutConverter::convert(
            &inter,
            4,
            6,
            TensorLayout::Interleaved,
            TensorLayout::RowMajor,
        );
        assert_eq!(src, back);
    }

    // -----------------------------------------------------------------------
    // LayoutConverter — cross-layout conversions
    // -----------------------------------------------------------------------

    #[test]
    fn test_col_to_blocked_roundtrip() {
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let col =
            LayoutConverter::convert(&src, 3, 4, TensorLayout::RowMajor, TensorLayout::ColumnMajor);
        let blocked = LayoutConverter::convert(
            &col,
            3,
            4,
            TensorLayout::ColumnMajor,
            TensorLayout::BlockedRC(2, 2),
        );
        let back = LayoutConverter::convert(
            &blocked,
            3,
            4,
            TensorLayout::BlockedRC(2, 2),
            TensorLayout::RowMajor,
        );
        assert_eq!(src, back);
    }

    #[test]
    fn test_interleaved_to_col_roundtrip() {
        let src: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let inter =
            LayoutConverter::convert(&src, 4, 5, TensorLayout::RowMajor, TensorLayout::Interleaved);
        let col = LayoutConverter::convert(
            &inter,
            4,
            5,
            TensorLayout::Interleaved,
            TensorLayout::ColumnMajor,
        );
        let back =
            LayoutConverter::convert(&col, 4, 5, TensorLayout::ColumnMajor, TensorLayout::RowMajor);
        assert_eq!(src, back);
    }

    // -----------------------------------------------------------------------
    // TileFormat
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_16x16_exact() {
        let tf = TileFormat::TILE_16X16;
        let src: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let tiled = tf.tile(&src, 16, 16);
        assert_eq!(tiled.len(), 256);
        // Single tile: should be identical to row-major
        assert_eq!(tiled, src);
    }

    #[test]
    fn test_tile_untile_roundtrip_exact() {
        let tf = TileFormat::TILE_16X16;
        let rows = 32;
        let cols = 32;
        let src: Vec<f32> = (0..(rows * cols) as u32).map(|i| i as f32).collect();
        let tiled = tf.tile(&src, rows, cols);
        let back = tf.untile(&tiled, rows, cols);
        assert_eq!(src, back);
    }

    #[test]
    fn test_tile_untile_roundtrip_non_divisible() {
        let tf = TileFormat::TILE_16X16;
        let rows = 20;
        let cols = 25;
        let src: Vec<f32> = (0..(rows * cols) as u32).map(|i| i as f32).collect();
        let tiled = tf.tile(&src, rows, cols);
        let (pr, pc) = tf.padded_dims(rows, cols);
        assert_eq!(pr, 32);
        assert_eq!(pc, 32);
        assert_eq!(tiled.len(), 32 * 32);
        let back = tf.untile(&tiled, rows, cols);
        assert_eq!(src, back);
    }

    #[test]
    fn test_tile_32x32_roundtrip() {
        let tf = TileFormat::TILE_32X32;
        let rows = 50;
        let cols = 70;
        let src: Vec<f32> = (0..(rows * cols) as u32).map(|i| i as f32).collect();
        let tiled = tf.tile(&src, rows, cols);
        let back = tf.untile(&tiled, rows, cols);
        assert_eq!(src, back);
    }

    #[test]
    fn test_tile_8x8_small() {
        let tf = TileFormat::TILE_8X8;
        let src: Vec<f32> = (0..3).map(|i| i as f32).collect();
        // 1×3 matrix tiled with 8×8 tiles
        let tiled = tf.tile(&src, 1, 3);
        assert_eq!(tiled.len(), 64); // 8×8
        let back = tf.untile(&tiled, 1, 3);
        assert_eq!(back, src);
    }

    #[test]
    fn test_tile_single_element() {
        let tf = TileFormat::TILE_16X16;
        let src = vec![99.0f32];
        let tiled = tf.tile(&src, 1, 1);
        assert_eq!(tiled.len(), 256); // 16×16
        assert_eq!(tiled[0], 99.0);
        // All padding should be zero
        assert!(tiled[1..].iter().all(|&v| v == 0.0));
        let back = tf.untile(&tiled, 1, 1);
        assert_eq!(back, vec![99.0]);
    }

    #[test]
    fn test_tile_count() {
        let tf = TileFormat::TILE_16X16;
        assert_eq!(tf.tile_count(16, 16), (1, 1));
        assert_eq!(tf.tile_count(17, 16), (2, 1));
        assert_eq!(tf.tile_count(1, 1), (1, 1));
        assert_eq!(tf.tile_count(32, 48), (2, 3));
    }

    #[test]
    fn test_tile_padded_dims() {
        let tf = TileFormat::TILE_16X16;
        assert_eq!(tf.padded_dims(16, 16), (16, 16));
        assert_eq!(tf.padded_dims(17, 1), (32, 16));
    }

    #[test]
    fn test_tile_custom_size() {
        let tf = TileFormat::new(4, 8);
        let src: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let tiled = tf.tile(&src, 4, 8);
        assert_eq!(tiled.len(), 32);
        let back = tf.untile(&tiled, 4, 8);
        assert_eq!(back, src);
    }

    #[test]
    fn test_tile_display() {
        assert_eq!(TileFormat::TILE_16X16.to_string(), "Tile(16×16)");
        assert_eq!(TileFormat::TILE_32X32.to_string(), "Tile(32×32)");
    }

    #[test]
    fn test_tile_tiled_len() {
        let tf = TileFormat::TILE_16X16;
        assert_eq!(tf.tiled_len(16, 16), 256);
        assert_eq!(tf.tiled_len(1, 1), 256);
        assert_eq!(tf.tiled_len(17, 17), 1024); // 32×32
    }

    // -----------------------------------------------------------------------
    // PackedFormat
    // -----------------------------------------------------------------------

    #[test]
    fn test_packed_plan_single() {
        let pf = PackedFormat::plan(&[("w", 100)], AlignmentRequirements::A770);
        assert_eq!(pf.entries.len(), 1);
        assert_eq!(pf.entries[0].offset, 0);
        assert_eq!(pf.entries[0].size, 100);
        assert_eq!(pf.entries[0].aligned_size, 128); // 64-byte aligned
        assert_eq!(pf.total_size, 128);
    }

    #[test]
    fn test_packed_plan_multiple() {
        let pf =
            PackedFormat::plan(&[("a", 50), ("b", 70), ("c", 10)], AlignmentRequirements::A770);
        assert_eq!(pf.entries.len(), 3);
        assert_eq!(pf.entries[0].offset, 0);
        assert_eq!(pf.entries[0].aligned_size, 64);
        assert_eq!(pf.entries[1].offset, 64);
        assert_eq!(pf.entries[1].aligned_size, 128);
        assert_eq!(pf.entries[2].offset, 192);
        assert_eq!(pf.entries[2].aligned_size, 64);
        assert_eq!(pf.total_size, 256);
    }

    #[test]
    fn test_packed_alignment_offsets() {
        let pf = PackedFormat::plan(&[("x", 1), ("y", 1)], AlignmentRequirements::A770);
        // Each tensor aligned to 64 bytes
        assert_eq!(pf.entries[0].offset, 0);
        assert_eq!(pf.entries[1].offset, 64);
        assert!(pf.entries[1].offset % 64 == 0);
    }

    #[test]
    fn test_packed_generic_alignment() {
        let pf = PackedFormat::plan(&[("w", 100)], AlignmentRequirements::GENERIC);
        assert_eq!(pf.entries[0].aligned_size, 256);
        assert_eq!(pf.total_size, 256);
    }

    #[test]
    fn test_packed_pack_unpack() {
        let pf = PackedFormat::plan(&[("a", 4), ("b", 8)], AlignmentRequirements::A770);
        let a_data = [1u8, 2, 3, 4];
        let b_data = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let buf = pf.pack(&[&a_data, &b_data]);
        assert_eq!(pf.unpack(&buf, 0), &a_data);
        assert_eq!(pf.unpack(&buf, 1), &b_data);
    }

    #[test]
    fn test_packed_find_by_name() {
        let pf = PackedFormat::plan(&[("weight", 100), ("bias", 20)], AlignmentRequirements::A770);
        assert_eq!(pf.find("weight"), Some(0));
        assert_eq!(pf.find("bias"), Some(1));
        assert_eq!(pf.find("missing"), None);
    }

    #[test]
    fn test_packed_padding_overhead() {
        let pf = PackedFormat::plan(&[("a", 60), ("b", 60)], AlignmentRequirements::A770);
        // a: 60→64 (4 wasted), b: 60→64 (4 wasted)
        assert_eq!(pf.padding_overhead(), 8);
    }

    #[test]
    fn test_packed_display() {
        let pf = PackedFormat::plan(&[("w", 100)], AlignmentRequirements::A770);
        let s = pf.to_string();
        assert!(s.contains("1 tensors"));
        assert!(s.contains("128B"));
    }

    #[test]
    fn test_packed_empty() {
        let pf = PackedFormat::plan(&[], AlignmentRequirements::A770);
        assert_eq!(pf.entries.len(), 0);
        assert_eq!(pf.total_size, 0);
        assert_eq!(pf.padding_overhead(), 0);
    }

    #[test]
    fn test_packed_exact_alignment() {
        // Size that's already aligned
        let pf = PackedFormat::plan(&[("exact", 64)], AlignmentRequirements::A770);
        assert_eq!(pf.entries[0].aligned_size, 64);
        assert_eq!(pf.padding_overhead(), 0);
    }

    // -----------------------------------------------------------------------
    // QuantFormatAdapter — I2_S
    // -----------------------------------------------------------------------

    #[test]
    fn test_dequant_i2s_basic() {
        // Pack [-1, 0, +1, -1] → byte 0b10_01_00_00 = but let's verify:
        // val -1 → 0b00, val 0 → 0b01, val +1 → 0b10, val -1 → 0b00
        // bits: pos0=00, pos1=01, pos2=10, pos3=00 → 0b00_10_01_00 = 0x24
        let packed = vec![0b00_10_01_00u8];
        let result = QuantFormatAdapter::dequant_i2s(&packed, 4);
        assert_eq!(result.data, vec![-1.0, 0.0, 1.0, -1.0]);
        assert_eq!(result.element_count, 4);
        assert_eq!(result.source_format, QuantFormat::I2S);
    }

    #[test]
    fn test_dequant_i2s_all_zero() {
        // All zeros → 0b01_01_01_01 = 0x55
        let packed = vec![0x55u8];
        let result = QuantFormatAdapter::dequant_i2s(&packed, 4);
        assert_eq!(result.data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dequant_i2s_all_plus_one() {
        // All +1 → 0b10_10_10_10 = 0xAA
        let packed = vec![0xAAu8];
        let result = QuantFormatAdapter::dequant_i2s(&packed, 4);
        assert_eq!(result.data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_dequant_i2s_all_minus_one() {
        // All -1 → 0b00_00_00_00 = 0x00
        let packed = vec![0x00u8];
        let result = QuantFormatAdapter::dequant_i2s(&packed, 4);
        assert_eq!(result.data, vec![-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_dequant_i2s_partial_byte() {
        // Only read 2 values from a full byte
        let packed = vec![0b00_10_01_00u8]; // [-1, 0, +1, -1]
        let result = QuantFormatAdapter::dequant_i2s(&packed, 2);
        assert_eq!(result.data, vec![-1.0, 0.0]);
        assert_eq!(result.element_count, 2);
    }

    // -----------------------------------------------------------------------
    // QuantFormatAdapter — QK256
    // -----------------------------------------------------------------------

    #[test]
    fn test_dequant_qk256_single_block() {
        const BLOCK_ELEMS: usize = 256;
        const DATA_BYTES: usize = BLOCK_ELEMS / 4;
        let scale: f32 = 2.0;
        let mut packed = Vec::with_capacity(4 + DATA_BYTES);
        packed.extend_from_slice(&scale.to_le_bytes());
        // All values = 0b01 → ternary 0 → 0.0 * scale
        packed.resize(4 + DATA_BYTES, 0x55);
        let result = QuantFormatAdapter::dequant_qk256(&packed, 1);
        assert_eq!(result.element_count, 256);
        assert!(result.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_qk256_scale_applied() {
        const BLOCK_ELEMS: usize = 256;
        const DATA_BYTES: usize = BLOCK_ELEMS / 4;
        let scale: f32 = 3.0;
        let mut packed = Vec::with_capacity(4 + DATA_BYTES);
        packed.extend_from_slice(&scale.to_le_bytes());
        // All values = 0b10 → ternary +1 → 1.0 * 3.0 = 3.0
        packed.resize(4 + DATA_BYTES, 0xAA);
        let result = QuantFormatAdapter::dequant_qk256(&packed, 1);
        assert!(result.data.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_dequant_qk256_negative_scale() {
        const BLOCK_ELEMS: usize = 256;
        const DATA_BYTES: usize = BLOCK_ELEMS / 4;
        let scale: f32 = -1.5;
        let mut packed = Vec::with_capacity(4 + DATA_BYTES);
        packed.extend_from_slice(&scale.to_le_bytes());
        // All +1 ternary → +1 * -1.5 = -1.5
        packed.resize(4 + DATA_BYTES, 0xAA);
        let result = QuantFormatAdapter::dequant_qk256(&packed, 1);
        assert!(result.data.iter().all(|&v| (v - (-1.5)).abs() < 1e-6));
    }

    #[test]
    fn test_adapt_dispatches_i2s() {
        let packed = vec![0x55u8]; // all zeros
        let result = QuantFormatAdapter::adapt(&packed, QuantFormat::I2S, 4);
        assert_eq!(result.source_format, QuantFormat::I2S);
        assert_eq!(result.data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_adapt_dispatches_qk256() {
        let scale: f32 = 1.0;
        let mut packed = Vec::with_capacity(68);
        packed.extend_from_slice(&scale.to_le_bytes());
        packed.resize(68, 0x55); // all ternary 0
        let result = QuantFormatAdapter::adapt(&packed, QuantFormat::Qk256, 256);
        assert_eq!(result.source_format, QuantFormat::Qk256);
        assert_eq!(result.element_count, 256);
    }

    #[test]
    fn test_quant_format_display() {
        assert_eq!(QuantFormat::I2S.to_string(), "I2_S");
        assert_eq!(QuantFormat::Qk256.to_string(), "QK256");
    }

    // -----------------------------------------------------------------------
    // ConversionPlan
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_empty() {
        let plan = ConversionPlan::new(4, 4);
        assert!(plan.is_empty());
        assert_eq!(plan.len(), 0);
    }

    #[test]
    fn test_plan_builder_chain() {
        let plan = ConversionPlan::new(16, 16)
            .dequantize(QuantFormat::I2S)
            .reorder(TensorLayout::RowMajor, TensorLayout::ColumnMajor)
            .tile(TileFormat::TILE_16X16);
        assert_eq!(plan.len(), 3);
    }

    #[test]
    fn test_plan_display() {
        let plan = ConversionPlan::new(8, 8)
            .dequantize(QuantFormat::I2S)
            .reorder(TensorLayout::RowMajor, TensorLayout::ColumnMajor);
        let s = plan.to_string();
        assert!(s.contains("8×8"));
        assert!(s.contains("Dequant"));
        assert!(s.contains("Reorder"));
    }

    #[test]
    fn test_plan_execute_reorder_only() {
        let src: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let plan =
            ConversionPlan::new(2, 3).reorder(TensorLayout::RowMajor, TensorLayout::ColumnMajor);
        let result = plan.execute(None, Some(&src));
        assert_eq!(result.data, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 3);
        assert_eq!(result.stats.steps_executed, 1);
    }

    #[test]
    fn test_plan_execute_tile_only() {
        let src: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let plan = ConversionPlan::new(2, 2).tile(TileFormat::TILE_8X8);
        let result = plan.execute(None, Some(&src));
        assert_eq!(result.data.len(), 64); // 8×8 padded
        assert_eq!(result.rows, 8);
        assert_eq!(result.cols, 8);
    }

    #[test]
    fn test_plan_execute_dequant_i2s() {
        let packed = vec![0b00_10_01_00u8]; // [-1, 0, +1, -1]
        let plan = ConversionPlan::new(2, 2).dequantize(QuantFormat::I2S);
        let result = plan.execute(Some(&packed), None);
        assert_eq!(result.data, vec![-1.0, 0.0, 1.0, -1.0]);
    }

    #[test]
    fn test_plan_execute_multi_step() {
        // Dequant → reorder → tile
        let packed = [0x55u8; 2]; // 8 zeros
        let plan = ConversionPlan::new(2, 4)
            .dequantize(QuantFormat::I2S)
            .reorder(TensorLayout::RowMajor, TensorLayout::ColumnMajor)
            .tile(TileFormat::TILE_8X8);
        let result = plan.execute(Some(&packed), None);
        assert_eq!(result.stats.steps_executed, 3);
        assert_eq!(result.data.len(), 64); // 8×8
        // All values should be 0.0 (ternary 0)
        assert!(result.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_plan_execute_pad() {
        let src = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
        let plan = ConversionPlan::new(2, 2).pad(4, 4);
        let result = plan.execute(None, Some(&src));
        assert_eq!(result.rows, 4);
        assert_eq!(result.cols, 4);
        assert_eq!(result.data.len(), 16);
        // Original values preserved
        assert_eq!(result.data[0], 1.0); // (0,0)
        assert_eq!(result.data[1], 2.0); // (0,1)
        assert_eq!(result.data[4], 3.0); // (1,0)
        assert_eq!(result.data[5], 4.0); // (1,1)
        // Padding is zero
        assert_eq!(result.data[2], 0.0);
        assert_eq!(result.data[3], 0.0);
    }

    // -----------------------------------------------------------------------
    // ConversionStats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_compression_ratio() {
        let stats = ConversionStats {
            source_bytes: 100,
            output_bytes: 400,
            elapsed_us: 10,
            steps_executed: 1,
        };
        assert!((stats.compression_ratio() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_stats_compression_ratio_zero_output() {
        let stats = ConversionStats {
            source_bytes: 100,
            output_bytes: 0,
            elapsed_us: 0,
            steps_executed: 0,
        };
        assert_eq!(stats.compression_ratio(), 0.0);
    }

    #[test]
    fn test_stats_throughput() {
        let stats = ConversionStats {
            source_bytes: 0,
            output_bytes: 1_000_000,
            elapsed_us: 1000, // 1ms
            steps_executed: 1,
        };
        // 1MB / 1ms = 1000 MB/s
        assert!((stats.throughput_mbps() - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn test_stats_throughput_zero_time() {
        let stats = ConversionStats {
            source_bytes: 0,
            output_bytes: 100,
            elapsed_us: 0,
            steps_executed: 0,
        };
        assert_eq!(stats.throughput_mbps(), 0.0);
    }

    #[test]
    fn test_stats_display() {
        let stats = ConversionStats {
            source_bytes: 68,
            output_bytes: 1024,
            elapsed_us: 5,
            steps_executed: 2,
        };
        let s = stats.to_string();
        assert!(s.contains("68B"));
        assert!(s.contains("1024B"));
        assert!(s.contains("2 steps"));
    }

    // -----------------------------------------------------------------------
    // ConversionStep display
    // -----------------------------------------------------------------------

    #[test]
    fn test_conversion_step_display() {
        assert!(ConversionStep::Dequantize(QuantFormat::I2S).to_string().contains("I2_S"));
        assert!(
            ConversionStep::Reorder(TensorLayout::RowMajor, TensorLayout::ColumnMajor)
                .to_string()
                .contains("Reorder")
        );
        assert!(ConversionStep::Tile(TileFormat::TILE_16X16).to_string().contains("16"));
        assert!(ConversionStep::Pad(32, 32).to_string().contains("32"));
    }

    // -----------------------------------------------------------------------
    // Property-style roundtrip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_layout_roundtrips() {
        let layouts = [
            TensorLayout::RowMajor,
            TensorLayout::ColumnMajor,
            TensorLayout::BlockedRC(4, 4),
            TensorLayout::Interleaved,
        ];
        let src: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let (rows, cols) = (8, 6);
        for &layout in &layouts {
            let converted =
                LayoutConverter::convert(&src, rows, cols, TensorLayout::RowMajor, layout);
            let back =
                LayoutConverter::convert(&converted, rows, cols, layout, TensorLayout::RowMajor);
            assert_eq!(src, back, "roundtrip failed for {layout}");
        }
    }

    #[test]
    fn test_tile_roundtrip_various_sizes() {
        let tiles = [TileFormat::TILE_8X8, TileFormat::TILE_16X16, TileFormat::TILE_32X32];
        let shapes = [(1, 1), (3, 5), (7, 13), (16, 16), (33, 17)];
        for tf in &tiles {
            for &(rows, cols) in &shapes {
                let src: Vec<f32> = (0..(rows * cols) as u32).map(|i| i as f32).collect();
                let tiled = tf.tile(&src, rows, cols);
                let back = tf.untile(&tiled, rows, cols);
                assert_eq!(src, back, "tile roundtrip failed for {tf} on {rows}×{cols}");
            }
        }
    }

    #[test]
    fn test_blocked_roundtrip_various_blocks() {
        let block_sizes = [(2, 2), (4, 4), (8, 8), (3, 5)];
        let (rows, cols) = (12, 15);
        let src: Vec<f32> = (0..(rows * cols) as u32).map(|i| i as f32).collect();
        for &(br, bc) in &block_sizes {
            let layout = TensorLayout::BlockedRC(br, bc);
            let blocked =
                LayoutConverter::convert(&src, rows, cols, TensorLayout::RowMajor, layout);
            let back =
                LayoutConverter::convert(&blocked, rows, cols, layout, TensorLayout::RowMajor);
            assert_eq!(src, back, "blocked roundtrip failed for {br}×{bc}");
        }
    }

    #[test]
    fn test_layout_conversion_preserves_element_count() {
        let layouts =
            [TensorLayout::RowMajor, TensorLayout::ColumnMajor, TensorLayout::Interleaved];
        let src: Vec<f32> = (0..30).map(|i| i as f32).collect();
        for &layout in &layouts {
            let out = LayoutConverter::convert(&src, 5, 6, TensorLayout::RowMajor, layout);
            assert_eq!(out.len(), 30, "element count changed for {layout}");
        }
    }

    #[test]
    fn test_plan_execute_dequant_then_reorder_then_tile() {
        // Full pipeline: I2_S → row-major → column-major → tiled
        // 4×4 = 16 elements, needs 4 bytes I2_S
        let values: Vec<i8> = (0..16).map(|i| (i % 3) as i8 - 1).collect();
        let packed = pack_i2s(&values);

        let plan = ConversionPlan::new(4, 4)
            .dequantize(QuantFormat::I2S)
            .reorder(TensorLayout::RowMajor, TensorLayout::ColumnMajor)
            .tile(TileFormat::TILE_8X8);

        let result = plan.execute(Some(&packed), None);
        assert_eq!(result.stats.steps_executed, 3);
        assert_eq!(result.data.len(), 64); // 8×8 tile
        assert_eq!(result.rows, 8);
        assert_eq!(result.cols, 8);
    }

    #[test]
    fn test_packed_large_tensor_alignment() {
        let tensors: Vec<(&str, usize)> = (0..10)
            .map(|i| {
                // Leak a string to get &'static str for the vec
                // In tests this is fine
                let name: &str = match i {
                    0 => "t0",
                    1 => "t1",
                    2 => "t2",
                    3 => "t3",
                    4 => "t4",
                    5 => "t5",
                    6 => "t6",
                    7 => "t7",
                    8 => "t8",
                    _ => "t9",
                };
                (name, (i + 1) * 100)
            })
            .collect();
        let pf = PackedFormat::plan(&tensors, AlignmentRequirements::A770);
        for entry in &pf.entries {
            assert!(
                entry.offset % 64 == 0,
                "tensor '{}' offset {} not 64-byte aligned",
                entry.name,
                entry.offset,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Helper used by tests
    // -----------------------------------------------------------------------

    fn pack_i2s(values: &[i8]) -> Vec<u8> {
        let len = values.len().div_ceil(4);
        let mut packed = vec![0u8; len];
        for (i, &v) in values.iter().enumerate() {
            let encoded: u8 = match v {
                -1 => 0,
                0 => 1,
                1 => 2,
                _ => panic!("invalid ternary value: {v}"),
            };
            packed[i / 4] |= encoded << ((i % 4) * 2);
        }
        packed
    }
}
