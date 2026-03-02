//! AVX-512 wide operations with scalar fallbacks.
//!
//! Provides 512-bit fused multiply-add (FMA), masked operations, conflict
//! detection, ternary logic (`vpternlog`), and compress/expand helpers.
//! Every routine probes `is_x86_feature_detected!("avx512f")` at runtime and
//! falls back to a portable scalar implementation on unsupported hardware.

// ── Feature detection ────────────────────────────────────────────────

/// Returns `true` when the current CPU supports AVX-512F.
#[inline]
pub fn avx512f_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ── Fused multiply-add (FMA) ─────────────────────────────────────────

/// Element-wise `a * b + c` on 16-wide f32 lanes.
///
/// Uses AVX-512 `vfmadd132ps` when available, scalar otherwise.
#[inline]
pub fn fma_f32x16(a: &[f32; 16], b: &[f32; 16], c: &[f32; 16]) -> [f32; 16] {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512f_available() {
            // SAFETY: `avx512f_available()` guarantees AVX-512F support.
            // All input slices are exactly 16 elements, matching the 512-bit
            // register width.  Loads and stores are to properly aligned stack
            // arrays through `_mm512_loadu_ps`/`_mm512_storeu_ps` which handle
            // unaligned pointers.
            unsafe { return fma_f32x16_avx512(a, b, c) }
        }
    }
    fma_f32x16_scalar(a, b, c)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fma_f32x16_avx512(a: &[f32; 16], b: &[f32; 16], c: &[f32; 16]) -> [f32; 16] {
    use std::arch::x86_64::*;
    // SAFETY: target_feature guarantees the required ISA extensions;
    // all pointers come from valid fixed-size array references.
    unsafe {
        let va = _mm512_loadu_ps(a.as_ptr());
        let vb = _mm512_loadu_ps(b.as_ptr());
        let vc = _mm512_loadu_ps(c.as_ptr());
        let vr = _mm512_fmadd_ps(va, vb, vc);
        let mut out = [0.0f32; 16];
        _mm512_storeu_ps(out.as_mut_ptr(), vr);
        out
    }
}

/// Scalar reference: `a[i] * b[i] + c[i]`.
#[inline]
pub fn fma_f32x16_scalar(a: &[f32; 16], b: &[f32; 16], c: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = a[i].mul_add(b[i], c[i]);
    }
    out
}

// ── Masked add ───────────────────────────────────────────────────────

/// Masked element-wise add: `if mask[i] { a[i] + b[i] } else { a[i] }`.
///
/// Uses AVX-512 mask registers (`__mmask16`) when available.
#[inline]
pub fn masked_add_f32x16(a: &[f32; 16], b: &[f32; 16], mask: u16) -> [f32; 16] {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512f_available() {
            // SAFETY: `avx512f_available()` guarantees AVX-512F support.
            // `mask` is a plain u16 interpreted as a 16-bit mask register.
            // Input arrays are exactly 16 elements wide.
            unsafe { return masked_add_f32x16_avx512(a, b, mask) }
        }
    }
    masked_add_f32x16_scalar(a, b, mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn masked_add_f32x16_avx512(a: &[f32; 16], b: &[f32; 16], mask: u16) -> [f32; 16] {
    use std::arch::x86_64::*;
    // SAFETY: target_feature guarantees the required ISA extensions;
    // all pointers come from valid fixed-size array references.
    unsafe {
        let va = _mm512_loadu_ps(a.as_ptr());
        let vb = _mm512_loadu_ps(b.as_ptr());
        let vr = _mm512_mask_add_ps(va, mask, va, vb);
        let mut out = [0.0f32; 16];
        _mm512_storeu_ps(out.as_mut_ptr(), vr);
        out
    }
}

/// Scalar masked add.
#[inline]
pub fn masked_add_f32x16_scalar(a: &[f32; 16], b: &[f32; 16], mask: u16) -> [f32; 16] {
    let mut out = *a;
    for i in 0..16 {
        if mask & (1 << i) != 0 {
            out[i] = a[i] + b[i];
        }
    }
    out
}

// ── Masked multiply ──────────────────────────────────────────────────

/// Masked element-wise multiply: `if mask[i] { a[i] * b[i] } else { a[i] }`.
#[inline]
pub fn masked_mul_f32x16(a: &[f32; 16], b: &[f32; 16], mask: u16) -> [f32; 16] {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512f_available() {
            // SAFETY: `avx512f_available()` guarantees AVX-512F support.
            // Input arrays have exactly 16 elements; mask is a plain u16.
            unsafe { return masked_mul_f32x16_avx512(a, b, mask) }
        }
    }
    masked_mul_f32x16_scalar(a, b, mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn masked_mul_f32x16_avx512(a: &[f32; 16], b: &[f32; 16], mask: u16) -> [f32; 16] {
    use std::arch::x86_64::*;
    // SAFETY: target_feature guarantees the required ISA extensions;
    // all pointers come from valid fixed-size array references.
    unsafe {
        let va = _mm512_loadu_ps(a.as_ptr());
        let vb = _mm512_loadu_ps(b.as_ptr());
        let vr = _mm512_mask_mul_ps(va, mask, va, vb);
        let mut out = [0.0f32; 16];
        _mm512_storeu_ps(out.as_mut_ptr(), vr);
        out
    }
}

/// Scalar masked multiply.
#[inline]
pub fn masked_mul_f32x16_scalar(a: &[f32; 16], b: &[f32; 16], mask: u16) -> [f32; 16] {
    let mut out = *a;
    for i in 0..16 {
        if mask & (1 << i) != 0 {
            out[i] = a[i] * b[i];
        }
    }
    out
}

// ── Masked FMA ───────────────────────────────────────────────────────

/// Masked fused multiply-add:
/// `if mask[i] { a[i]*b[i]+c[i] } else { c[i] }`.
#[inline]
pub fn masked_fma_f32x16(a: &[f32; 16], b: &[f32; 16], c: &[f32; 16], mask: u16) -> [f32; 16] {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512f_available() {
            // SAFETY: `avx512f_available()` guarantees AVX-512F support.
            // All three input arrays are exactly 16 elements.
            unsafe { return masked_fma_f32x16_avx512(a, b, c, mask) }
        }
    }
    masked_fma_f32x16_scalar(a, b, c, mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn masked_fma_f32x16_avx512(
    a: &[f32; 16],
    b: &[f32; 16],
    c: &[f32; 16],
    mask: u16,
) -> [f32; 16] {
    use std::arch::x86_64::*;
    // SAFETY: target_feature guarantees AVX-512F; arrays are 16 elements.
    unsafe {
        let va = _mm512_loadu_ps(a.as_ptr());
        let vb = _mm512_loadu_ps(b.as_ptr());
        let vc = _mm512_loadu_ps(c.as_ptr());
        // mask3_fmadd: for each lane, if mask bit set → a*b+c, else → c
        let vr = _mm512_mask3_fmadd_ps(va, vb, vc, mask);
        let mut out = [0.0f32; 16];
        _mm512_storeu_ps(out.as_mut_ptr(), vr);
        out
    }
}

/// Scalar masked FMA.
#[inline]
pub fn masked_fma_f32x16_scalar(
    a: &[f32; 16],
    b: &[f32; 16],
    c: &[f32; 16],
    mask: u16,
) -> [f32; 16] {
    let mut out = *c;
    for i in 0..16 {
        if mask & (1 << i) != 0 {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
    }
    out
}

// ── Conflict detection (i32 lanes) ──────────────────────────────────

/// Per-lane conflict mask for 16 × i32 indices.
///
/// For each lane `i`, the returned `conflicts[i]` is a bitmask of all
/// **earlier** lanes `j < i` that hold the same value.  This mirrors the
/// semantics of `vpconflictd`.
#[inline]
pub fn conflict_detect_i32x16(indices: &[i32; 16]) -> [u16; 16] {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512cd_available() {
            // SAFETY: `avx512cd_available()` guarantees AVX-512CD support.
            // The input array is exactly 16 elements matching the register
            // width. The output is extracted via `_mm512_extracti32x4_epi32`
            // into four 128-bit chunks and reassembled into per-lane u16
            // conflict masks.
            unsafe { return conflict_detect_i32x16_avx512(indices) }
        }
    }
    conflict_detect_i32x16_scalar(indices)
}

/// Returns `true` when AVX-512CD (conflict detection) is supported.
#[inline]
pub fn avx512cd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512cd")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512cd")]
unsafe fn conflict_detect_i32x16_avx512(indices: &[i32; 16]) -> [u16; 16] {
    use std::arch::x86_64::*;
    // SAFETY: target_feature guarantees the required ISA extensions;
    // all pointers come from valid fixed-size array references.
    unsafe {
        let vi = _mm512_loadu_epi32(indices.as_ptr());
        let vc = _mm512_conflict_epi32(vi);
        // Extract each 32-bit lane's low 16 bits as the conflict mask.
        let mut out = [0u16; 16];
        let mut tmp = [0i32; 16];
        _mm512_storeu_epi32(tmp.as_mut_ptr(), vc);
        for i in 0..16 {
            out[i] = tmp[i] as u16;
        }
        out
    }
}

/// Scalar conflict detection: for each lane `i`, set bit `j` if
/// `indices[j] == indices[i]` and `j < i`.
#[inline]
pub fn conflict_detect_i32x16_scalar(indices: &[i32; 16]) -> [u16; 16] {
    let mut out = [0u16; 16];
    for i in 0..16 {
        for j in 0..i {
            if indices[j] == indices[i] {
                out[i] |= 1 << j;
            }
        }
    }
    out
}

// ── Ternary logic (vpternlog) ────────────────────────────────────────

/// Bit-wise ternary logic on three 16-wide i32 vectors.
///
/// For every bit position, the three source bits form a 3-bit index into the
/// 8-bit `imm` truth table.  This is equivalent to `vpternlogd`.
#[inline]
pub fn ternary_logic_i32x16(a: &[i32; 16], b: &[i32; 16], c: &[i32; 16], imm: u8) -> [i32; 16] {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512f_available() {
            // SAFETY: `avx512f_available()` guarantees AVX-512F support.
            // All input arrays are exactly 16 i32 elements.
            // `imm` selects the truth-table row at compile-time via a
            // const-generic dispatch helper that covers common patterns.
            unsafe { return ternary_logic_i32x16_avx512(a, b, c, imm) }
        }
    }
    ternary_logic_i32x16_scalar(a, b, c, imm)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn ternary_logic_i32x16_avx512(
    a: &[i32; 16],
    b: &[i32; 16],
    c: &[i32; 16],
    imm: u8,
) -> [i32; 16] {
    use std::arch::x86_64::*;
    // SAFETY: target_feature guarantees AVX-512F; arrays are 16 i32 elements.
    // `_mm512_ternarylogic_epi32` requires a compile-time immediate so we
    // dispatch the most useful truth-table values; everything else
    // falls through to scalar.
    unsafe {
        let va = _mm512_loadu_epi32(a.as_ptr());
        let vb = _mm512_loadu_epi32(b.as_ptr());
        let vc = _mm512_loadu_epi32(c.as_ptr());
        let vr = match imm {
            // a AND b AND c
            0x80 => _mm512_ternarylogic_epi32::<0x80>(va, vb, vc),
            // a XOR b XOR c
            0x96 => _mm512_ternarylogic_epi32::<0x96>(va, vb, vc),
            // a OR b OR c
            0xFE => _mm512_ternarylogic_epi32::<0xFE>(va, vb, vc),
            // (a AND b) OR c
            0xF8 => _mm512_ternarylogic_epi32::<0xF8>(va, vb, vc),
            // a AND (b OR c)
            0xA8 => _mm512_ternarylogic_epi32::<0xA8>(va, vb, vc),
            // (a XOR b) AND c
            0x60 => _mm512_ternarylogic_epi32::<0x60>(va, vb, vc),
            // NOT a (ignores b, c)
            0x0F => _mm512_ternarylogic_epi32::<0x0F>(va, vb, vc),
            // NOT c (ignores a, b)
            0x55 => _mm512_ternarylogic_epi32::<0x55>(va, vb, vc),
            _ => {
                // Unsupported immediate — fall through to scalar.
                let mut out = [0i32; 16];
                _mm512_storeu_epi32(out.as_mut_ptr(), va);
                return ternary_logic_i32x16_scalar(&out, b, c, imm);
            }
        };
        let mut out = [0i32; 16];
        _mm512_storeu_epi32(out.as_mut_ptr(), vr);
        out
    }
}

/// Scalar ternary logic: for each bit, the three source bits index into `imm`.
#[inline]
pub fn ternary_logic_i32x16_scalar(
    a: &[i32; 16],
    b: &[i32; 16],
    c: &[i32; 16],
    imm: u8,
) -> [i32; 16] {
    let mut out = [0i32; 16];
    for lane in 0..16 {
        let mut r: i32 = 0;
        for bit in 0..32 {
            let ba = (a[lane] >> bit) & 1;
            let bb = (b[lane] >> bit) & 1;
            let bc = (c[lane] >> bit) & 1;
            let idx = (ba << 2) | (bb << 1) | bc;
            let result_bit = (imm >> idx) & 1;
            r |= (result_bit as i32) << bit;
        }
        out[lane] = r;
    }
    out
}

// ── Compress (sparse store) ──────────────────────────────────────────

/// Compress: pack the lanes of `src` whose corresponding `mask` bit is set
/// into the low positions of the result.  The remaining high lanes are zero.
#[inline]
pub fn compress_f32x16(src: &[f32; 16], mask: u16) -> [f32; 16] {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512f_available() {
            // SAFETY: `avx512f_available()` guarantees AVX-512F support.
            // `src` is exactly 16 f32 elements; `mask` is a plain u16
            // interpreted as a 16-bit k-mask.
            unsafe { return compress_f32x16_avx512(src, mask) }
        }
    }
    compress_f32x16_scalar(src, mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn compress_f32x16_avx512(src: &[f32; 16], mask: u16) -> [f32; 16] {
    use std::arch::x86_64::*;
    // SAFETY: target_feature guarantees the required ISA extensions;
    // all pointers come from valid fixed-size array references.
    unsafe {
        let vs = _mm512_loadu_ps(src.as_ptr());
        let zero = _mm512_setzero_ps();
        let vr = _mm512_mask_compress_ps(zero, mask, vs);
        let mut out = [0.0f32; 16];
        _mm512_storeu_ps(out.as_mut_ptr(), vr);
        out
    }
}

/// Scalar compress: selected lanes packed left, rest zero.
#[inline]
pub fn compress_f32x16_scalar(src: &[f32; 16], mask: u16) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    let mut j = 0;
    for (i, &val) in src.iter().enumerate() {
        if mask & (1 << i) != 0 {
            out[j] = val;
            j += 1;
        }
    }
    out
}

// ── Expand (sparse load) ─────────────────────────────────────────────

/// Expand: scatter the low `popcount(mask)` elements of `src` into the lanes
/// whose `mask` bit is set.  Unselected lanes are zero.
#[inline]
pub fn expand_f32x16(src: &[f32; 16], mask: u16) -> [f32; 16] {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512f_available() {
            // SAFETY: `avx512f_available()` guarantees AVX-512F support.
            // `src` is 16 f32 elements; `mask` is a u16 k-mask.
            unsafe { return expand_f32x16_avx512(src, mask) }
        }
    }
    expand_f32x16_scalar(src, mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn expand_f32x16_avx512(src: &[f32; 16], mask: u16) -> [f32; 16] {
    use std::arch::x86_64::*;
    // SAFETY: target_feature guarantees the required ISA extensions;
    // all pointers come from valid fixed-size array references.
    unsafe {
        let vs = _mm512_loadu_ps(src.as_ptr());
        let zero = _mm512_setzero_ps();
        let vr = _mm512_mask_expand_ps(zero, mask, vs);
        let mut out = [0.0f32; 16];
        _mm512_storeu_ps(out.as_mut_ptr(), vr);
        out
    }
}

/// Scalar expand: low elements scatter into mask-set positions.
#[inline]
pub fn expand_f32x16_scalar(src: &[f32; 16], mask: u16) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    let mut j = 0;
    for (i, slot) in out.iter_mut().enumerate() {
        if mask & (1 << i) != 0 {
            *slot = src[j];
            j += 1;
        }
    }
    out
}

// ── Horizontal reduce-add ────────────────────────────────────────────

/// Horizontal sum of 16 × f32 lanes.
#[inline]
pub fn reduce_add_f32x16(src: &[f32; 16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512f_available() {
            // SAFETY: `avx512f_available()` guarantees AVX-512F support.
            // `src` is exactly 16 f32 elements matching the register width.
            unsafe { return reduce_add_f32x16_avx512(src) }
        }
    }
    reduce_add_f32x16_scalar(src)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn reduce_add_f32x16_avx512(src: &[f32; 16]) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: target_feature guarantees the required ISA extensions;
    // all pointers come from valid fixed-size array references.
    unsafe {
        let vs = _mm512_loadu_ps(src.as_ptr());
        _mm512_reduce_add_ps(vs)
    }
}

/// Scalar horizontal sum.
#[inline]
pub fn reduce_add_f32x16_scalar(src: &[f32; 16]) -> f32 {
    src.iter().sum()
}

// ── Masked reduce-add ────────────────────────────────────────────────

/// Horizontal sum of the lanes selected by `mask`.
#[inline]
pub fn masked_reduce_add_f32x16(src: &[f32; 16], mask: u16) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512f_available() {
            // SAFETY: `avx512f_available()` guarantees AVX-512F support.
            // `src` has 16 elements; `mask` is a u16 k-mask.
            unsafe { return masked_reduce_add_f32x16_avx512(src, mask) }
        }
    }
    masked_reduce_add_f32x16_scalar(src, mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn masked_reduce_add_f32x16_avx512(src: &[f32; 16], mask: u16) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: target_feature guarantees the required ISA extensions;
    // all pointers come from valid fixed-size array references.
    unsafe {
        let vs = _mm512_loadu_ps(src.as_ptr());
        _mm512_mask_reduce_add_ps(mask, vs)
    }
}

/// Scalar masked horizontal sum.
#[inline]
pub fn masked_reduce_add_f32x16_scalar(src: &[f32; 16], mask: u16) -> f32 {
    let mut sum = 0.0f32;
    for (i, &val) in src.iter().enumerate() {
        if mask & (1 << i) != 0 {
            sum += val;
        }
    }
    sum
}

// ── Broadcast ────────────────────────────────────────────────────────

/// Broadcast a single f32 into all 16 lanes.
#[inline]
pub fn broadcast_f32x16(val: f32) -> [f32; 16] {
    [val; 16]
}

// ── Blend (select) ───────────────────────────────────────────────────

/// Per-lane select: `if mask[i] { b[i] } else { a[i] }`.
#[inline]
pub fn blend_f32x16(a: &[f32; 16], b: &[f32; 16], mask: u16) -> [f32; 16] {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512f_available() {
            // SAFETY: `avx512f_available()` guarantees AVX-512F support.
            // Both arrays are 16 elements; `mask` is a u16 k-mask.
            unsafe { return blend_f32x16_avx512(a, b, mask) }
        }
    }
    blend_f32x16_scalar(a, b, mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn blend_f32x16_avx512(a: &[f32; 16], b: &[f32; 16], mask: u16) -> [f32; 16] {
    use std::arch::x86_64::*;
    // SAFETY: target_feature guarantees the required ISA extensions;
    // all pointers come from valid fixed-size array references.
    unsafe {
        let va = _mm512_loadu_ps(a.as_ptr());
        let vb = _mm512_loadu_ps(b.as_ptr());
        let vr = _mm512_mask_blend_ps(mask, va, vb);
        let mut out = [0.0f32; 16];
        _mm512_storeu_ps(out.as_mut_ptr(), vr);
        out
    }
}

/// Scalar blend.
#[inline]
pub fn blend_f32x16_scalar(a: &[f32; 16], b: &[f32; 16], mask: u16) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = if mask & (1 << i) != 0 { b[i] } else { a[i] };
    }
    out
}

// ── Absolute value ───────────────────────────────────────────────────

/// Element-wise absolute value of 16 × f32 lanes.
#[inline]
pub fn abs_f32x16(src: &[f32; 16]) -> [f32; 16] {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512f_available() {
            // SAFETY: `avx512f_available()` guarantees AVX-512F support.
            // `src` is exactly 16 elements.
            unsafe { return abs_f32x16_avx512(src) }
        }
    }
    abs_f32x16_scalar(src)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn abs_f32x16_avx512(src: &[f32; 16]) -> [f32; 16] {
    use std::arch::x86_64::*;
    // SAFETY: target_feature guarantees the required ISA extensions;
    // all pointers come from valid fixed-size array references.
    unsafe {
        let vs = _mm512_loadu_ps(src.as_ptr());
        let vr = _mm512_abs_ps(vs);
        let mut out = [0.0f32; 16];
        _mm512_storeu_ps(out.as_mut_ptr(), vr);
        out
    }
}

/// Scalar abs.
#[inline]
pub fn abs_f32x16_scalar(src: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for (o, &s) in out.iter_mut().zip(src.iter()) {
        *o = s.abs();
    }
    out
}

// ── Min / Max ────────────────────────────────────────────────────────

/// Element-wise minimum.
#[inline]
pub fn min_f32x16(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512f_available() {
            // SAFETY: `avx512f_available()` guarantees AVX-512F support.
            unsafe { return min_f32x16_avx512(a, b) }
        }
    }
    min_f32x16_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn min_f32x16_avx512(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    use std::arch::x86_64::*;
    // SAFETY: target_feature guarantees the required ISA extensions;
    // all pointers come from valid fixed-size array references.
    unsafe {
        let va = _mm512_loadu_ps(a.as_ptr());
        let vb = _mm512_loadu_ps(b.as_ptr());
        let vr = _mm512_min_ps(va, vb);
        let mut out = [0.0f32; 16];
        _mm512_storeu_ps(out.as_mut_ptr(), vr);
        out
    }
}

/// Scalar min.
#[inline]
pub fn min_f32x16_scalar(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = a[i].min(b[i]);
    }
    out
}

/// Element-wise maximum.
#[inline]
pub fn max_f32x16(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512f_available() {
            // SAFETY: `avx512f_available()` guarantees AVX-512F support.
            unsafe { return max_f32x16_avx512(a, b) }
        }
    }
    max_f32x16_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn max_f32x16_avx512(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    use std::arch::x86_64::*;
    // SAFETY: target_feature guarantees the required ISA extensions;
    // all pointers come from valid fixed-size array references.
    unsafe {
        let va = _mm512_loadu_ps(a.as_ptr());
        let vb = _mm512_loadu_ps(b.as_ptr());
        let vr = _mm512_max_ps(va, vb);
        let mut out = [0.0f32; 16];
        _mm512_storeu_ps(out.as_mut_ptr(), vr);
        out
    }
}

/// Scalar max.
#[inline]
pub fn max_f32x16_scalar(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = a[i].max(b[i]);
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers
    fn ones() -> [f32; 16] {
        [1.0; 16]
    }
    fn zeros() -> [f32; 16] {
        [0.0; 16]
    }
    fn iota() -> [f32; 16] {
        let mut v = [0.0f32; 16];
        for i in 0..16 {
            v[i] = i as f32;
        }
        v
    }
    fn neg_iota() -> [f32; 16] {
        let mut v = [0.0f32; 16];
        for i in 0..16 {
            v[i] = -(i as f32);
        }
        v
    }

    // ── Feature detection ────────────────────────────────────────────

    #[test]
    fn test_avx512f_available_returns_bool() {
        // Just verify it doesn't panic.
        let _ = avx512f_available();
    }

    #[test]
    fn test_avx512cd_available_returns_bool() {
        let _ = avx512cd_available();
    }

    // ── FMA ──────────────────────────────────────────────────────────

    #[test]
    fn test_fma_identity() {
        let a = iota();
        let b = ones();
        let c = zeros();
        let r = fma_f32x16(&a, &b, &c);
        assert_eq!(r, a, "a*1+0 == a");
    }

    #[test]
    fn test_fma_add_only() {
        let a = zeros();
        let b = zeros();
        let c = iota();
        let r = fma_f32x16(&a, &b, &c);
        assert_eq!(r, c, "0*0+c == c");
    }

    #[test]
    fn test_fma_values() {
        let a = [2.0f32; 16];
        let b = [3.0f32; 16];
        let c = [4.0f32; 16];
        let r = fma_f32x16(&a, &b, &c);
        for v in r {
            assert!((v - 10.0).abs() < 1e-6, "2*3+4 == 10");
        }
    }

    #[test]
    fn test_fma_scalar_matches_dispatch() {
        let a = iota();
        let b = [2.0f32; 16];
        let c = [0.5f32; 16];
        let dispatched = fma_f32x16(&a, &b, &c);
        let scalar = fma_f32x16_scalar(&a, &b, &c);
        for i in 0..16 {
            assert!((dispatched[i] - scalar[i]).abs() < 1e-5, "lane {i} mismatch");
        }
    }

    // ── Masked add ───────────────────────────────────────────────────

    #[test]
    fn test_masked_add_all_set() {
        let a = iota();
        let b = ones();
        let r = masked_add_f32x16(&a, &b, 0xFFFF);
        for i in 0..16 {
            assert!((r[i] - (i as f32 + 1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_masked_add_none_set() {
        let a = iota();
        let b = ones();
        let r = masked_add_f32x16(&a, &b, 0x0000);
        assert_eq!(r, a);
    }

    #[test]
    fn test_masked_add_even_lanes() {
        let a = zeros();
        let b = ones();
        let mask: u16 = 0x5555; // bits 0,2,4,...
        let r = masked_add_f32x16(&a, &b, mask);
        for i in 0..16 {
            let expected = if i % 2 == 0 { 1.0 } else { 0.0 };
            assert!((r[i] - expected).abs() < 1e-6, "lane {i}");
        }
    }

    #[test]
    fn test_masked_add_scalar_matches_dispatch() {
        let a = iota();
        let b = [3.0f32; 16];
        let mask: u16 = 0xABCD;
        let dispatched = masked_add_f32x16(&a, &b, mask);
        let scalar = masked_add_f32x16_scalar(&a, &b, mask);
        assert_eq!(dispatched, scalar);
    }

    // ── Masked multiply ──────────────────────────────────────────────

    #[test]
    fn test_masked_mul_all_set() {
        let a = [2.0f32; 16];
        let b = [3.0f32; 16];
        let r = masked_mul_f32x16(&a, &b, 0xFFFF);
        for v in r {
            assert!((v - 6.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_masked_mul_none_set() {
        let a = [2.0f32; 16];
        let b = [3.0f32; 16];
        let r = masked_mul_f32x16(&a, &b, 0x0000);
        assert_eq!(r, a);
    }

    #[test]
    fn test_masked_mul_scalar_matches_dispatch() {
        let a = iota();
        let b = [5.0f32; 16];
        let mask: u16 = 0x1234;
        let dispatched = masked_mul_f32x16(&a, &b, mask);
        let scalar = masked_mul_f32x16_scalar(&a, &b, mask);
        assert_eq!(dispatched, scalar);
    }

    // ── Masked FMA ───────────────────────────────────────────────────

    #[test]
    fn test_masked_fma_all_set() {
        let a = [2.0f32; 16];
        let b = [3.0f32; 16];
        let c = [1.0f32; 16];
        let r = masked_fma_f32x16(&a, &b, &c, 0xFFFF);
        for v in r {
            assert!((v - 7.0).abs() < 1e-6, "2*3+1 == 7");
        }
    }

    #[test]
    fn test_masked_fma_none_set() {
        let a = [2.0f32; 16];
        let b = [3.0f32; 16];
        let c = [1.0f32; 16];
        let r = masked_fma_f32x16(&a, &b, &c, 0x0000);
        assert_eq!(r, c, "unmasked lanes keep c");
    }

    #[test]
    fn test_masked_fma_scalar_matches_dispatch() {
        let a = iota();
        let b = [2.0f32; 16];
        let c = [10.0f32; 16];
        let mask: u16 = 0xF0F0;
        let dispatched = masked_fma_f32x16(&a, &b, &c, mask);
        let scalar = masked_fma_f32x16_scalar(&a, &b, &c, mask);
        for i in 0..16 {
            assert!((dispatched[i] - scalar[i]).abs() < 1e-5, "lane {i} mismatch");
        }
    }

    // ── Conflict detection ───────────────────────────────────────────

    #[test]
    fn test_conflict_all_unique() {
        let indices: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let c = conflict_detect_i32x16(&indices);
        for mask in c {
            assert_eq!(mask, 0, "unique indices → no conflicts");
        }
    }

    #[test]
    fn test_conflict_all_same() {
        let indices = [42i32; 16];
        let c = conflict_detect_i32x16(&indices);
        assert_eq!(c[0], 0, "first lane has no earlier duplicates");
        for i in 1..16 {
            // Lane i conflicts with all lanes 0..i.
            let expected = (1u16 << i) - 1;
            assert_eq!(c[i], expected, "lane {i}");
        }
    }

    #[test]
    fn test_conflict_pair() {
        let mut indices = [0i32; 16];
        for i in 0..16 {
            indices[i] = i as i32;
        }
        // Make lane 5 duplicate lane 2.
        indices[5] = 2;
        let c = conflict_detect_i32x16(&indices);
        assert_eq!(c[5], 1 << 2, "lane 5 conflicts with lane 2");
        assert_eq!(c[2], 0, "lane 2 has no earlier conflict");
    }

    #[test]
    fn test_conflict_scalar_matches_dispatch() {
        let indices = [1, 2, 1, 3, 2, 1, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6];
        let dispatched = conflict_detect_i32x16(&indices);
        let scalar = conflict_detect_i32x16_scalar(&indices);
        assert_eq!(dispatched, scalar);
    }

    // ── Ternary logic ────────────────────────────────────────────────

    #[test]
    fn test_ternlog_and() {
        let a = [0x0F0F0F0Fi32; 16];
        let b = [0x00FF00FFi32; 16];
        let c = [0x0000FFFFi32; 16];
        let r = ternary_logic_i32x16(&a, &b, &c, 0x80);
        let s = ternary_logic_i32x16_scalar(&a, &b, &c, 0x80);
        assert_eq!(r, s);
        // a AND b AND c
        for i in 0..16 {
            assert_eq!(r[i], a[i] & b[i] & c[i]);
        }
    }

    #[test]
    fn test_ternlog_or() {
        let a = [0x0F0F0F0Fi32; 16];
        let b = [0x00FF00FFi32; 16];
        let c = [0x0000FFFFi32; 16];
        let r = ternary_logic_i32x16(&a, &b, &c, 0xFE);
        for i in 0..16 {
            assert_eq!(r[i], a[i] | b[i] | c[i]);
        }
    }

    #[test]
    fn test_ternlog_xor() {
        let a = [0x0F0F0F0Fi32; 16];
        let b = [0x00FF00FFi32; 16];
        let c = [0x0000FFFFi32; 16];
        let r = ternary_logic_i32x16(&a, &b, &c, 0x96);
        for i in 0..16 {
            assert_eq!(r[i], a[i] ^ b[i] ^ c[i]);
        }
    }

    #[test]
    fn test_ternlog_not_a() {
        let a = [0x0F0F0F0Fi32; 16];
        let b = [0i32; 16];
        let c = [0i32; 16];
        // NOT a: result=1 when a-bit=0 (idx 0..3), imm = 0x0F
        let r = ternary_logic_i32x16(&a, &b, &c, 0x0F);
        for i in 0..16 {
            assert_eq!(r[i], !a[i]);
        }
    }

    #[test]
    fn test_ternlog_scalar_matches_dispatch() {
        let a = [0xDEADBEEFu32 as i32; 16];
        let b = [0xCAFEBABEu32 as i32; 16];
        let c = [0x12345678i32; 16];
        for imm in [0x0F, 0x55, 0x60, 0x80, 0x96, 0xA8, 0xF8, 0xFE] {
            let dispatched = ternary_logic_i32x16(&a, &b, &c, imm);
            let scalar = ternary_logic_i32x16_scalar(&a, &b, &c, imm);
            assert_eq!(dispatched, scalar, "imm=0x{imm:02X}");
        }
    }

    // ── Compress / Expand ────────────────────────────────────────────

    #[test]
    fn test_compress_all_set() {
        let src = iota();
        let r = compress_f32x16(&src, 0xFFFF);
        assert_eq!(r, src, "all set → identity");
    }

    #[test]
    fn test_compress_none_set() {
        let src = iota();
        let r = compress_f32x16(&src, 0x0000);
        assert_eq!(r, zeros(), "none set → all zero");
    }

    #[test]
    fn test_compress_even_lanes() {
        let src = iota();
        let mask: u16 = 0x5555;
        let r = compress_f32x16(&src, mask);
        // Even lanes: 0,2,4,6,8,10,12,14 → packed into first 8 slots
        let expected =
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(r, expected);
    }

    #[test]
    fn test_compress_scalar_matches_dispatch() {
        let src = iota();
        let mask: u16 = 0x9C3A;
        let dispatched = compress_f32x16(&src, mask);
        let scalar = compress_f32x16_scalar(&src, mask);
        assert_eq!(dispatched, scalar);
    }

    #[test]
    fn test_expand_all_set() {
        let src = iota();
        let r = expand_f32x16(&src, 0xFFFF);
        assert_eq!(r, src, "all set → identity");
    }

    #[test]
    fn test_expand_none_set() {
        let src = iota();
        let r = expand_f32x16(&src, 0x0000);
        assert_eq!(r, zeros(), "none set → all zero");
    }

    #[test]
    fn test_expand_even_lanes() {
        let src = iota();
        let mask: u16 = 0x5555;
        let r = expand_f32x16(&src, mask);
        // First 8 source elements scatter into even lanes.
        let expected =
            [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0];
        assert_eq!(r, expected);
    }

    #[test]
    fn test_expand_scalar_matches_dispatch() {
        let src = iota();
        let mask: u16 = 0x9C3A;
        let dispatched = expand_f32x16(&src, mask);
        let scalar = expand_f32x16_scalar(&src, mask);
        assert_eq!(dispatched, scalar);
    }

    #[test]
    fn test_compress_expand_roundtrip() {
        let src = iota();
        let mask: u16 = 0xF00F;
        let compressed = compress_f32x16(&src, mask);
        let expanded = expand_f32x16(&compressed, mask);
        // The selected lanes should survive the round-trip.
        for i in 0..16 {
            if mask & (1 << i) != 0 {
                assert!((expanded[i] - src[i]).abs() < 1e-6, "lane {i}: roundtrip mismatch");
            } else {
                assert!(expanded[i].abs() < 1e-6, "lane {i}: unselected should be zero");
            }
        }
    }

    // ── Reduce-add ───────────────────────────────────────────────────

    #[test]
    fn test_reduce_add_iota() {
        let src = iota();
        let r = reduce_add_f32x16(&src);
        // sum(0..16) = 120
        assert!((r - 120.0).abs() < 1e-4);
    }

    #[test]
    fn test_reduce_add_scalar_matches_dispatch() {
        let src = iota();
        let dispatched = reduce_add_f32x16(&src);
        let scalar = reduce_add_f32x16_scalar(&src);
        assert!((dispatched - scalar).abs() < 1e-4);
    }

    #[test]
    fn test_masked_reduce_add() {
        let src = ones();
        let mask: u16 = 0x00FF; // lower 8 lanes
        let r = masked_reduce_add_f32x16(&src, mask);
        assert!((r - 8.0).abs() < 1e-4);
    }

    #[test]
    fn test_masked_reduce_add_scalar_matches_dispatch() {
        let src = iota();
        let mask: u16 = 0xAAAA;
        let dispatched = masked_reduce_add_f32x16(&src, mask);
        let scalar = masked_reduce_add_f32x16_scalar(&src, mask);
        assert!((dispatched - scalar).abs() < 1e-4);
    }

    // ── Blend ────────────────────────────────────────────────────────

    #[test]
    fn test_blend_all_a() {
        let a = iota();
        let b = ones();
        let r = blend_f32x16(&a, &b, 0x0000);
        assert_eq!(r, a);
    }

    #[test]
    fn test_blend_all_b() {
        let a = iota();
        let b = ones();
        let r = blend_f32x16(&a, &b, 0xFFFF);
        assert_eq!(r, b);
    }

    #[test]
    fn test_blend_scalar_matches_dispatch() {
        let a = iota();
        let b = [99.0f32; 16];
        let mask: u16 = 0xC3C3;
        let dispatched = blend_f32x16(&a, &b, mask);
        let scalar = blend_f32x16_scalar(&a, &b, mask);
        assert_eq!(dispatched, scalar);
    }

    // ── Abs ──────────────────────────────────────────────────────────

    #[test]
    fn test_abs_positive() {
        let src = iota();
        let r = abs_f32x16(&src);
        assert_eq!(r, src);
    }

    #[test]
    fn test_abs_negative() {
        let src = neg_iota();
        let r = abs_f32x16(&src);
        let expected = iota();
        assert_eq!(r, expected);
    }

    #[test]
    fn test_abs_scalar_matches_dispatch() {
        let mut src = neg_iota();
        src[3] = 5.0;
        src[7] = -0.0;
        let dispatched = abs_f32x16(&src);
        let scalar = abs_f32x16_scalar(&src);
        assert_eq!(dispatched, scalar);
    }

    // ── Min / Max ────────────────────────────────────────────────────

    #[test]
    fn test_min_same() {
        let a = iota();
        let r = min_f32x16(&a, &a);
        assert_eq!(r, a);
    }

    #[test]
    fn test_min_values() {
        let a = [1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0, 0.0, 9.0, 1.0, 5.0, 3.0, 7.0, 2.0, 6.0];
        let b = [2.0, 4.0, 4.0, 6.0, 3.0, 5.0, 5.0, 7.0, 1.0, 8.0, 2.0, 4.0, 4.0, 6.0, 3.0, 5.0];
        let r = min_f32x16(&a, &b);
        for i in 0..16 {
            assert_eq!(r[i], a[i].min(b[i]), "lane {i}");
        }
    }

    #[test]
    fn test_max_same() {
        let a = iota();
        let r = max_f32x16(&a, &a);
        assert_eq!(r, a);
    }

    #[test]
    fn test_max_values() {
        let a = [1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0, 0.0, 9.0, 1.0, 5.0, 3.0, 7.0, 2.0, 6.0];
        let b = [2.0, 4.0, 4.0, 6.0, 3.0, 5.0, 5.0, 7.0, 1.0, 8.0, 2.0, 4.0, 4.0, 6.0, 3.0, 5.0];
        let r = max_f32x16(&a, &b);
        for i in 0..16 {
            assert_eq!(r[i], a[i].max(b[i]), "lane {i}");
        }
    }

    #[test]
    fn test_min_max_scalar_match() {
        let a = iota();
        let b: [f32; 16] =
            [15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        assert_eq!(min_f32x16(&a, &b), min_f32x16_scalar(&a, &b));
        assert_eq!(max_f32x16(&a, &b), max_f32x16_scalar(&a, &b));
    }

    // ── Broadcast ────────────────────────────────────────────────────

    #[test]
    fn test_broadcast() {
        let r = broadcast_f32x16(3.14);
        for v in r {
            assert!((v - 3.14).abs() < 1e-6);
        }
    }
}
