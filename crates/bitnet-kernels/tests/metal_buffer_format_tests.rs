#![cfg(target_os = "macos")]
#![allow(
    clippy::useless_vec,
    clippy::approx_constant,
    clippy::excessive_precision,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop
)]
//! Metal buffer format validation tests for Apple Silicon.
//!
//! Tests Metal buffer alignment, format packing/unpacking, threadgroup
//! constraints, dispatch grid calculation, endianness, shared memory
//! limits, and texture format compatibility.
//!
//! All tests exercise pure Rust logic — no Metal API or GPU required.

// ── Constants ───────────────────────────────────────────────────────

/// Metal buffer alignment requirement (bytes).
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Maximum total threads per threadgroup on Apple Silicon.
const MAX_THREADGROUP_TOTAL: u32 = 1024;

/// Maximum threads per dimension in a threadgroup.
const MAX_THREADGROUP_PER_DIM: u32 = 1024;

/// SIMD group width on Apple Silicon GPUs (M1–M4).
const SIMD_GROUP_WIDTH: u32 = 32;

/// Default threadgroup shared memory limit (bytes).
const SHARED_MEMORY_LIMIT: usize = 32 * 1024;

// ── Helper structs ──────────────────────────────────────────────────

/// Simulated threadgroup dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ThreadgroupDims {
    x: u32,
    y: u32,
    z: u32,
}

impl ThreadgroupDims {
    fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    fn total(&self) -> u32 {
        self.x * self.y * self.z
    }

    fn is_valid(&self) -> bool {
        self.x >= 1
            && self.y >= 1
            && self.z >= 1
            && self.x <= MAX_THREADGROUP_PER_DIM
            && self.y <= MAX_THREADGROUP_PER_DIM
            && self.z <= MAX_THREADGROUP_PER_DIM
            && self.total() <= MAX_THREADGROUP_TOTAL
    }
}

/// Dispatch grid dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DispatchGrid {
    groups_x: u32,
    groups_y: u32,
    groups_z: u32,
}

/// Metal-compatible pixel/buffer format identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(dead_code)]
enum MetalFormat {
    Float32 = 0,
    Float16 = 1,
    Int8 = 2,
    Uint8 = 3,
    Int32 = 4,
    Uint32 = 5,
}

impl MetalFormat {
    fn bytes_per_element(self) -> usize {
        match self {
            Self::Float32 | Self::Int32 | Self::Uint32 => 4,
            Self::Float16 => 2,
            Self::Int8 | Self::Uint8 => 1,
        }
    }
}

// ── Helper functions ────────────────────────────────────────────────

/// Round `size` up to the next multiple of `alignment`.
fn align_up(size: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (size + alignment - 1) & !(alignment - 1)
}

/// Check if `size` is aligned to `alignment`.
fn is_aligned(size: usize, alignment: usize) -> bool {
    size % alignment == 0
}

/// Compute dispatch grid dimensions for a tensor shape given threadgroup dims.
fn compute_dispatch_grid(shape: &[u32], threadgroup: &ThreadgroupDims) -> DispatchGrid {
    let work_x = shape.first().copied().unwrap_or(1);
    let work_y = if shape.len() > 1 { shape[1] } else { 1 };
    let work_z = if shape.len() > 2 { shape[2..].iter().copied().product() } else { 1 };

    DispatchGrid {
        groups_x: work_x.div_ceil(threadgroup.x),
        groups_y: work_y.div_ceil(threadgroup.y),
        groups_z: work_z.div_ceil(threadgroup.z),
    }
}

/// Pack a pair of f32 values into f16 bytes (simplified: truncate mantissa).
fn pack_f16_pair(a: f32, b: f32) -> [u8; 4] {
    let ha = f32_to_f16_bits(a);
    let hb = f32_to_f16_bits(b);
    [(ha & 0xFF) as u8, (ha >> 8) as u8, (hb & 0xFF) as u8, (hb >> 8) as u8]
}

/// Minimal f32 → f16 bit conversion (IEEE 754 truncation, no rounding).
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = (bits >> 13) & 0x3FF;

    if exponent <= 0 {
        // Flush subnormals to zero for simplicity.
        sign as u16
    } else if exponent >= 31 {
        // Infinity / overflow.
        (sign | 0x7C00) as u16
    } else {
        (sign | ((exponent as u32) << 10) | mantissa) as u16
    }
}

/// Unpack f16 bits back to f32.
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exponent = ((h >> 10) & 0x1F) as i32;
    let mantissa = (h & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal — not needed for these tests, return 0.
        return f32::from_bits(sign << 31);
    }
    if exponent == 31 {
        let f_bits = (sign << 31) | 0x7F800000 | (mantissa << 13);
        return f32::from_bits(f_bits);
    }

    let f_exp = (exponent - 15 + 127) as u32;
    let f_bits = (sign << 31) | (f_exp << 23) | (mantissa << 13);
    f32::from_bits(f_bits)
}

/// Pack four 2-bit signed values into a single byte (little-endian order).
fn pack_i2s_byte(vals: [i8; 4]) -> u8 {
    let mut byte: u8 = 0;
    for (i, &v) in vals.iter().enumerate() {
        let bits = (v & 0x3) as u8;
        byte |= bits << (i * 2);
    }
    byte
}

/// Unpack a byte into four 2-bit signed values.
fn unpack_i2s_byte(byte: u8) -> [i8; 4] {
    let mut vals = [0i8; 4];
    for i in 0..4 {
        let raw = ((byte >> (i * 2)) & 0x3) as i8;
        // Sign-extend: 2-bit signed: 0b10 = -2, 0b11 = -1, 0b00 = 0, 0b01 = 1
        vals[i] = if raw >= 2 { raw - 4 } else { raw };
    }
    vals
}

/// Mapping of Metal texture format names to byte-per-pixel.
fn texture_format_bpp(name: &str) -> Option<usize> {
    match name {
        "r8unorm" | "r8snorm" | "r8uint" | "r8sint" => Some(1),
        "r16float" | "r16uint" | "r16sint" | "rg8unorm" => Some(2),
        "r32float" | "r32uint" | "r32sint" | "rg16float" | "rgba8unorm" => Some(4),
        "rg32float" | "rgba16float" => Some(8),
        "rgba32float" => Some(16),
        _ => None,
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── (a) Buffer alignment ────────────────────────────────────────

    #[test]
    fn test_metal_buffer_alignment_256() {
        assert_eq!(METAL_BUFFER_ALIGNMENT, 256);
        assert!(METAL_BUFFER_ALIGNMENT.is_power_of_two());

        // Exact multiples are already aligned.
        for mult in [1, 2, 4, 16, 64, 256] {
            let size = METAL_BUFFER_ALIGNMENT * mult;
            assert!(is_aligned(size, METAL_BUFFER_ALIGNMENT), "size={size}");
        }

        // Off-by-one is not aligned.
        assert!(!is_aligned(1, METAL_BUFFER_ALIGNMENT));
        assert!(!is_aligned(255, METAL_BUFFER_ALIGNMENT));
        assert!(!is_aligned(257, METAL_BUFFER_ALIGNMENT));

        // Zero is trivially aligned.
        assert!(is_aligned(0, METAL_BUFFER_ALIGNMENT));
    }

    // ── (b) Buffer size rounding ────────────────────────────────────

    #[test]
    fn test_metal_buffer_size_rounding() {
        // Zero stays zero.
        assert_eq!(align_up(0, METAL_BUFFER_ALIGNMENT), 0);

        // Exact alignment stays unchanged.
        assert_eq!(align_up(256, METAL_BUFFER_ALIGNMENT), 256);
        assert_eq!(align_up(512, METAL_BUFFER_ALIGNMENT), 512);

        // Round up from partial.
        assert_eq!(align_up(1, METAL_BUFFER_ALIGNMENT), 256);
        assert_eq!(align_up(100, METAL_BUFFER_ALIGNMENT), 256);
        assert_eq!(align_up(255, METAL_BUFFER_ALIGNMENT), 256);
        assert_eq!(align_up(257, METAL_BUFFER_ALIGNMENT), 512);

        // Large sizes round correctly.
        assert_eq!(align_up(1024, METAL_BUFFER_ALIGNMENT), 1024);
        assert_eq!(align_up(1025, METAL_BUFFER_ALIGNMENT), 1280);

        // Aligned result is always >= original and divisible by alignment.
        for size in [0, 1, 7, 128, 255, 256, 300, 1000, 65536] {
            let aligned = align_up(size, METAL_BUFFER_ALIGNMENT);
            assert!(aligned >= size, "aligned={aligned} < size={size}");
            assert!(is_aligned(aligned, METAL_BUFFER_ALIGNMENT), "aligned={aligned} not aligned");
        }
    }

    // ── (c) f16 format layout ───────────────────────────────────────

    #[test]
    fn test_metal_format_f16_layout() {
        // f16 occupies 2 bytes per element.
        assert_eq!(MetalFormat::Float16.bytes_per_element(), 2);

        // Round-trip known values through f16.
        let test_values: &[f32] = &[0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 65504.0];
        for &v in test_values {
            let bits = f32_to_f16_bits(v);
            let back = f16_bits_to_f32(bits);
            assert!(
                (back - v).abs() < 1e-3 || (v.abs() > 1000.0 && (back - v).abs() / v.abs() < 1e-2),
                "f16 round-trip failed for {v}: got {back}"
            );
        }

        // Zero has all-zero bits (positive zero).
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);

        // 1.0 in f16 is 0x3C00.
        assert_eq!(f32_to_f16_bits(1.0), 0x3C00);

        // Negative 1.0 in f16 is 0xBC00.
        assert_eq!(f32_to_f16_bits(-1.0), 0xBC00);

        // Pack pair produces little-endian layout.
        let packed = pack_f16_pair(1.0, -1.0);
        assert_eq!(packed[0], 0x00); // low byte of 0x3C00
        assert_eq!(packed[1], 0x3C); // high byte of 0x3C00
        assert_eq!(packed[2], 0x00); // low byte of 0xBC00
        assert_eq!(packed[3], 0xBC); // high byte of 0xBC00
    }

    // ── (d) I2_S format layout ──────────────────────────────────────

    #[test]
    fn test_metal_format_i2s_layout() {
        // Four 2-bit values fit in one byte.
        let byte = pack_i2s_byte([0, 1, -1, -2]);
        let unpacked = unpack_i2s_byte(byte);
        assert_eq!(unpacked, [0, 1, -1, -2]);

        // All zeros.
        assert_eq!(pack_i2s_byte([0, 0, 0, 0]), 0x00);
        assert_eq!(unpack_i2s_byte(0x00), [0, 0, 0, 0]);

        // All ones.
        let byte = pack_i2s_byte([1, 1, 1, 1]);
        assert_eq!(unpack_i2s_byte(byte), [1, 1, 1, 1]);

        // Mixed negative round-trip.
        let vals = [-2, -1, 0, 1];
        assert_eq!(unpack_i2s_byte(pack_i2s_byte(vals)), vals);

        // Buffer size: N elements of i2s require ceil(N/4) bytes.
        for n in [1, 4, 7, 8, 31, 32, 100, 256] {
            let raw_bytes = (n + 3) / 4;
            let aligned = align_up(raw_bytes, METAL_BUFFER_ALIGNMENT);
            assert!(aligned >= raw_bytes);
            assert!(is_aligned(aligned, METAL_BUFFER_ALIGNMENT));
        }
    }

    // ── (e) Threadgroup dimensions ──────────────────────────────────

    #[test]
    fn test_metal_threadgroup_dimensions() {
        // Typical valid configurations.
        let valid = [
            ThreadgroupDims::new(256, 1, 1),
            ThreadgroupDims::new(1, 256, 1),
            ThreadgroupDims::new(32, 32, 1),
            ThreadgroupDims::new(16, 16, 4),
            ThreadgroupDims::new(8, 8, 16),
            ThreadgroupDims::new(1024, 1, 1),
            ThreadgroupDims::new(1, 1, 1),
        ];
        for dims in &valid {
            assert!(dims.is_valid(), "expected valid: {dims:?}");
            assert!(dims.total() <= MAX_THREADGROUP_TOTAL);
        }

        // Invalid: total exceeds 1024.
        let invalid_total = ThreadgroupDims::new(32, 32, 2); // 2048
        assert!(!invalid_total.is_valid());

        // Invalid: per-dimension exceeds 1024.
        let invalid_dim = ThreadgroupDims::new(2048, 1, 1);
        assert!(!invalid_dim.is_valid());

        // Invalid: zero dimension.
        let zero_dim = ThreadgroupDims::new(0, 1, 1);
        assert!(!zero_dim.is_valid());
    }

    // ── (f) SIMD group width ────────────────────────────────────────

    #[test]
    fn test_metal_simd_group_width_32() {
        assert_eq!(SIMD_GROUP_WIDTH, 32);

        // Threadgroup sizes should ideally be a multiple of SIMD group width.
        for tg_size in [32, 64, 128, 256, 512, 1024] {
            assert_eq!(
                tg_size % SIMD_GROUP_WIDTH,
                0,
                "threadgroup size {tg_size} not a multiple of SIMD width"
            );
        }

        // Number of SIMD groups in a threadgroup.
        let tg = ThreadgroupDims::new(256, 1, 1);
        let simd_groups = tg.total() / SIMD_GROUP_WIDTH;
        assert_eq!(simd_groups, 8);
    }

    // ── (g) Dispatch grid calculation ───────────────────────────────

    #[test]
    fn test_metal_dispatch_grid_calculation() {
        let tg = ThreadgroupDims::new(32, 32, 1);

        // Exact fit.
        let grid = compute_dispatch_grid(&[64, 64], &tg);
        assert_eq!(grid.groups_x, 2);
        assert_eq!(grid.groups_y, 2);
        assert_eq!(grid.groups_z, 1);

        // Non-exact: needs rounding up.
        let grid = compute_dispatch_grid(&[33, 33], &tg);
        assert_eq!(grid.groups_x, 2);
        assert_eq!(grid.groups_y, 2);

        // 1-D tensor.
        let tg1d = ThreadgroupDims::new(256, 1, 1);
        let grid = compute_dispatch_grid(&[1000], &tg1d);
        assert_eq!(grid.groups_x, 4); // ceil(1000/256)
        assert_eq!(grid.groups_y, 1);
        assert_eq!(grid.groups_z, 1);

        // 3-D tensor: z dims are collapsed.
        let grid = compute_dispatch_grid(&[128, 64, 4, 2], &tg);
        assert_eq!(grid.groups_x, 4); // ceil(128/32)
        assert_eq!(grid.groups_y, 2); // ceil(64/32)
        assert_eq!(grid.groups_z, 8); // 4*2 = 8

        // Single element.
        let grid = compute_dispatch_grid(&[1], &ThreadgroupDims::new(1, 1, 1));
        assert_eq!(grid.groups_x, 1);
        assert_eq!(grid.groups_y, 1);
        assert_eq!(grid.groups_z, 1);
    }

    // ── (h) Endianness ──────────────────────────────────────────────

    #[test]
    fn test_metal_buffer_contents_endianness() {
        // Metal on Apple Silicon uses little-endian byte order.
        let val: u32 = 0xDEADBEEF;
        let le_bytes = val.to_le_bytes();
        assert_eq!(le_bytes, [0xEF, 0xBE, 0xAD, 0xDE]);

        // f32 in little-endian.
        let fval: f32 = 1.0;
        let le = fval.to_le_bytes();
        assert_eq!(le, [0x00, 0x00, 0x80, 0x3F]);

        // f16 1.0 (0x3C00) in little-endian.
        let h: u16 = 0x3C00;
        let le = h.to_le_bytes();
        assert_eq!(le, [0x00, 0x3C]);

        // Multi-element buffer: elements are contiguous in little-endian.
        let values: [f32; 3] = [1.0, 2.0, 3.0];
        let mut buf = Vec::with_capacity(12);
        for v in &values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(buf.len(), 12);
        // Read back.
        for (i, v) in values.iter().enumerate() {
            let offset = i * 4;
            let bytes: [u8; 4] = buf[offset..offset + 4].try_into().unwrap();
            assert_eq!(f32::from_le_bytes(bytes), *v);
        }
    }

    // ── (i) Shared memory limits ────────────────────────────────────

    #[test]
    fn test_metal_shared_memory_limits() {
        assert_eq!(SHARED_MEMORY_LIMIT, 32 * 1024);

        // A tile of 32×32 f32 values.
        let tile_bytes = 32 * 32 * std::mem::size_of::<f32>();
        assert_eq!(tile_bytes, 4096);
        assert!(tile_bytes <= SHARED_MEMORY_LIMIT);

        // Maximum number of f32 elements that fit.
        let max_f32 = SHARED_MEMORY_LIMIT / std::mem::size_of::<f32>();
        assert_eq!(max_f32, 8192);

        // Maximum number of f16 elements that fit.
        let max_f16 = SHARED_MEMORY_LIMIT / MetalFormat::Float16.bytes_per_element();
        assert_eq!(max_f16, 16384);

        // Exceeding the limit.
        let oversized = 33 * 1024;
        assert!(oversized > SHARED_MEMORY_LIMIT);

        // Typical tiling configs that fit in shared memory.
        let configs: &[(usize, usize, usize)] = &[
            (64, 64, 4),  // 64×64 f32 tile = 16 KB
            (128, 64, 2), // 128×64 f16 tile = 16 KB
            (32, 32, 4),  // 32×32 f32 tile = 4 KB
        ];
        for &(rows, cols, elem_size) in configs {
            let bytes = rows * cols * elem_size;
            assert!(
                bytes <= SHARED_MEMORY_LIMIT,
                "tile {rows}×{cols}×{elem_size} = {bytes} > {SHARED_MEMORY_LIMIT}"
            );
        }
    }

    // ── (j) Texture format compatibility ────────────────────────────

    #[test]
    fn test_metal_texture_format_compatibility() {
        // Known formats resolve correctly.
        assert_eq!(texture_format_bpp("r8unorm"), Some(1));
        assert_eq!(texture_format_bpp("r16float"), Some(2));
        assert_eq!(texture_format_bpp("r32float"), Some(4));
        assert_eq!(texture_format_bpp("rg16float"), Some(4));
        assert_eq!(texture_format_bpp("rg32float"), Some(8));
        assert_eq!(texture_format_bpp("rgba8unorm"), Some(4));
        assert_eq!(texture_format_bpp("rgba16float"), Some(8));
        assert_eq!(texture_format_bpp("rgba32float"), Some(16));

        // Unknown format returns None.
        assert_eq!(texture_format_bpp("rgb10a2unorm"), None);
        assert_eq!(texture_format_bpp(""), None);

        // MetalFormat element sizes.
        assert_eq!(MetalFormat::Float32.bytes_per_element(), 4);
        assert_eq!(MetalFormat::Float16.bytes_per_element(), 2);
        assert_eq!(MetalFormat::Int8.bytes_per_element(), 1);
        assert_eq!(MetalFormat::Uint8.bytes_per_element(), 1);
        assert_eq!(MetalFormat::Int32.bytes_per_element(), 4);
        assert_eq!(MetalFormat::Uint32.bytes_per_element(), 4);

        // Buffer size for a 1024-element f16 texture row, aligned to Metal requirements.
        let row_bytes = 1024 * MetalFormat::Float16.bytes_per_element();
        let aligned = align_up(row_bytes, METAL_BUFFER_ALIGNMENT);
        assert_eq!(row_bytes, 2048);
        assert_eq!(aligned, 2048); // 2048 is already 256-aligned
    }
}
