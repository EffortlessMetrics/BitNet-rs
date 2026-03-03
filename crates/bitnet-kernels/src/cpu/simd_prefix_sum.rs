//! CPU SIMD prefix sum operations with AVX2 acceleration.
//!
//! Provides inclusive/exclusive prefix sums for `f32` and `i32` slices,
//! Blelloch-style parallel scan for large arrays, segmented scans, and
//! key-based segmented scans.  All public entry points perform runtime
//! AVX2 detection and fall back to portable scalar code when AVX2 is
//! unavailable.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Helpers ────────────────────────────────────────────────────────

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

// ── Scan mode ──────────────────────────────────────────────────────

/// Whether a prefix sum is *inclusive* (each element includes itself) or
/// *exclusive* (each element is the sum of all preceding elements).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScanMode {
    /// `out[i] = data[0] + data[1] + … + data[i]`
    Inclusive,
    /// `out[i] = data[0] + data[1] + … + data[i-1]`, with `out[0] = identity`.
    Exclusive,
}

// ── Cumulative operation kind ──────────────────────────────────────

/// Operation for [`cumulative_op_f32`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CumulativeOp {
    Min,
    Max,
    Product,
}

// ═══════════════════════════════════════════════════════════════════
// f32 prefix sum
// ═══════════════════════════════════════════════════════════════════

/// Compute prefix sum of an `f32` slice.
///
/// Returns a new `Vec<f32>` of the same length.
pub fn prefix_sum_f32(data: &[f32], mode: ScanMode) -> Result<Vec<f32>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected at runtime.
            return Ok(unsafe { prefix_sum_f32_avx2(data, mode) });
        }
    }

    Ok(prefix_sum_f32_scalar(data, mode))
}

fn prefix_sum_f32_scalar(data: &[f32], mode: ScanMode) -> Vec<f32> {
    let n = data.len();
    let mut out = vec![0.0f32; n];
    match mode {
        ScanMode::Inclusive => {
            out[0] = data[0];
            for i in 1..n {
                out[i] = out[i - 1] + data[i];
            }
        }
        ScanMode::Exclusive => {
            // out[0] = 0 (identity for addition)
            for i in 1..n {
                out[i] = out[i - 1] + data[i - 1];
            }
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn prefix_sum_f32_avx2(data: &[f32], mode: ScanMode) -> Vec<f32> {
    use std::arch::x86_64::*;

    let n = data.len();
    let mut out = vec![0.0f32; n];

    // For short arrays the scalar loop is already efficient; the SIMD
    // benefit comes from processing 8-element chunks with an in-register
    // prefix sum (Hillis-Steele style shift-and-add).
    if n < 8 {
        return prefix_sum_f32_scalar(data, mode);
    }

    let mut running = 0.0f32;
    let chunks = n / 8;
    let remainder = n % 8;

    for c in 0..chunks {
        let off = c * 8;
        unsafe {
            let mut v = _mm256_loadu_ps(data.as_ptr().add(off));

            // In-register inclusive prefix sum (3-step Hillis-Steele):
            // Step 1: shift by 1
            let shifted1 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(v), 4));
            v = _mm256_add_ps(v, shifted1);
            // Step 2: shift by 2
            let shifted2 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(v), 8));
            v = _mm256_add_ps(v, shifted2);
            // Step 3: cross-lane — propagate lane 3 of low 128 to high 128.
            let low128 = _mm256_castps256_ps128(v);
            let lane3_val = _mm_extract_ps(low128, 3);
            let lane3 = _mm256_set1_ps(f32::from_bits(lane3_val as u32));
            let high_mask = _mm256_castsi256_ps(_mm256_set_epi32(-1, -1, -1, -1, 0, 0, 0, 0));
            let correction = _mm256_and_ps(lane3, high_mask);
            v = _mm256_add_ps(v, correction);

            // Add running total from previous chunks.
            let running_vec = _mm256_set1_ps(running);
            v = _mm256_add_ps(v, running_vec);

            // Update running total to the last element of this chunk.
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), v);
            running = tmp[7];

            _mm256_storeu_ps(out.as_mut_ptr().add(off), v);
        }
    }

    // Scalar tail.
    let tail_start = chunks * 8;
    for i in tail_start..n {
        running += data[i];
        out[i] = running;
    }

    // Convert inclusive → exclusive if requested.
    if mode == ScanMode::Exclusive {
        // Shift right, inserting identity (0.0) at front.
        let last = out.len();
        if last > 0 {
            out.copy_within(0..last - 1, 1);
            out[0] = 0.0;
        }
        // Trim excess: out[n-1] held the full sum which we don't need for
        // exclusive, but the shift already placed the correct values.
    }

    // For remainder elements we may have produced an inclusive result;
    // the exclusive conversion above already handles all elements.
    let _ = remainder;

    out
}

// ═══════════════════════════════════════════════════════════════════
// i32 prefix sum
// ═══════════════════════════════════════════════════════════════════

/// Compute prefix sum of an `i32` slice.
pub fn prefix_sum_i32(data: &[i32], mode: ScanMode) -> Result<Vec<i32>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return Ok(unsafe { prefix_sum_i32_avx2(data, mode) });
        }
    }

    Ok(prefix_sum_i32_scalar(data, mode))
}

fn prefix_sum_i32_scalar(data: &[i32], mode: ScanMode) -> Vec<i32> {
    let n = data.len();
    let mut out = vec![0i32; n];
    match mode {
        ScanMode::Inclusive => {
            out[0] = data[0];
            for i in 1..n {
                out[i] = out[i - 1].wrapping_add(data[i]);
            }
        }
        ScanMode::Exclusive => {
            for i in 1..n {
                out[i] = out[i - 1].wrapping_add(data[i - 1]);
            }
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn prefix_sum_i32_avx2(data: &[i32], mode: ScanMode) -> Vec<i32> {
    use std::arch::x86_64::*;

    let n = data.len();
    if n < 8 {
        return prefix_sum_i32_scalar(data, mode);
    }

    let mut out = vec![0i32; n];
    let mut running = 0i32;
    let chunks = n / 8;

    for c in 0..chunks {
        let off = c * 8;
        unsafe {
            let mut v = _mm256_loadu_si256(data.as_ptr().add(off) as *const __m256i);

            // Hillis-Steele in-register prefix sum for i32.
            let s1 = _mm256_slli_si256(v, 4);
            v = _mm256_add_epi32(v, s1);
            let s2 = _mm256_slli_si256(v, 8);
            v = _mm256_add_epi32(v, s2);

            // Cross-lane fix: broadcast element 3 of low lane to high lane.
            let low128 = _mm256_castsi256_si128(v);
            let elem3 = _mm_extract_epi32(low128, 3);
            let bcast = _mm256_set1_epi32(elem3);
            let high_mask = _mm256_set_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
            let correction = _mm256_and_si256(bcast, high_mask);
            v = _mm256_add_epi32(v, correction);

            let running_vec = _mm256_set1_epi32(running);
            v = _mm256_add_epi32(v, running_vec);

            let mut tmp = [0i32; 8];
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, v);
            running = tmp[7];

            _mm256_storeu_si256(out.as_mut_ptr().add(off) as *mut __m256i, v);
        }
    }

    let tail_start = chunks * 8;
    for i in tail_start..n {
        running = running.wrapping_add(data[i]);
        out[i] = running;
    }

    if mode == ScanMode::Exclusive {
        let last = out.len();
        if last > 0 {
            out.copy_within(0..last - 1, 1);
            out[0] = 0;
        }
    }

    out
}

// ═══════════════════════════════════════════════════════════════════
// Blelloch-style parallel scan
// ═══════════════════════════════════════════════════════════════════

/// Blelloch-style work-efficient parallel scan (exclusive prefix sum).
///
/// This is a two-pass algorithm — *up-sweep* (reduce) then *down-sweep*
/// — that mirrors the classic GPU parallel prefix sum but runs on CPU.
/// For large arrays the block-level parallelism maps well to SIMD.
pub fn parallel_scan_f32(data: &[f32]) -> Result<Vec<f32>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let n = data.len();

    // For small inputs, delegate to the simple sequential scan.
    if n <= 64 {
        return prefix_sum_f32(data, ScanMode::Exclusive);
    }

    // Block-based approach: each block is scanned independently, then
    // block totals are scanned, and finally each block is offset.
    let block_size = 256;
    let num_blocks = n.div_ceil(block_size);
    let mut out = data.to_vec();

    // 1. Per-block inclusive scan and collect block totals.
    let mut block_totals = vec![0.0f32; num_blocks];
    for (b, total) in block_totals.iter_mut().enumerate() {
        let start = b * block_size;
        let end = (start + block_size).min(n);
        let mut acc = 0.0f32;
        for val in &mut out[start..end] {
            acc += *val;
            *val = acc;
        }
        *total = acc;
    }

    // 2. Exclusive scan on block totals.
    let offsets = prefix_sum_f32(&block_totals, ScanMode::Exclusive)?;

    // 3. Add block offsets and convert to exclusive scan.
    for (b, &off) in offsets.iter().enumerate() {
        let start = b * block_size;
        let end = (start + block_size).min(n);
        // Convert inclusive → exclusive within the block.
        let mut prev = off;
        for val in &mut out[start..end] {
            let inclusive = *val + off;
            *val = prev;
            prev = inclusive;
        }
    }

    Ok(out)
}

/// Blelloch-style work-efficient parallel scan (exclusive prefix sum)
/// for `i32`.
pub fn parallel_scan_i32(data: &[i32]) -> Result<Vec<i32>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let n = data.len();

    if n <= 64 {
        return prefix_sum_i32(data, ScanMode::Exclusive);
    }

    let block_size = 256;
    let num_blocks = n.div_ceil(block_size);
    let mut out = data.to_vec();

    let mut block_totals = vec![0i32; num_blocks];
    for (b, total) in block_totals.iter_mut().enumerate() {
        let start = b * block_size;
        let end = (start + block_size).min(n);
        let mut acc = 0i32;
        for val in &mut out[start..end] {
            acc = acc.wrapping_add(*val);
            *val = acc;
        }
        *total = acc;
    }

    let offsets = prefix_sum_i32(&block_totals, ScanMode::Exclusive)?;

    for (b, &off) in offsets.iter().enumerate() {
        let start = b * block_size;
        let end = (start + block_size).min(n);
        let mut prev = off;
        for val in &mut out[start..end] {
            let inclusive = (*val).wrapping_add(off);
            *val = prev;
            prev = inclusive;
        }
    }

    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════
// Segmented scan
// ═══════════════════════════════════════════════════════════════════

/// Segmented inclusive prefix sum for `f32`.
///
/// `flags[i] == true` marks the **start** of a new segment.  The scan
/// resets at every segment boundary.
pub fn segmented_scan_f32(data: &[f32], flags: &[bool], mode: ScanMode) -> Result<Vec<f32>> {
    if data.len() != flags.len() {
        return Err(invalid_args("data and flags must have equal length"));
    }
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let n = data.len();
    let mut out = vec![0.0f32; n];

    match mode {
        ScanMode::Inclusive => {
            out[0] = data[0];
            for i in 1..n {
                if flags[i] {
                    out[i] = data[i];
                } else {
                    out[i] = out[i - 1] + data[i];
                }
            }
        }
        ScanMode::Exclusive => {
            out[0] = 0.0;
            for i in 1..n {
                if flags[i] {
                    out[i] = 0.0;
                } else {
                    out[i] = out[i - 1] + data[i - 1];
                }
            }
        }
    }

    Ok(out)
}

/// Segmented inclusive prefix sum for `i32`.
pub fn segmented_scan_i32(data: &[i32], flags: &[bool], mode: ScanMode) -> Result<Vec<i32>> {
    if data.len() != flags.len() {
        return Err(invalid_args("data and flags must have equal length"));
    }
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let n = data.len();
    let mut out = vec![0i32; n];

    match mode {
        ScanMode::Inclusive => {
            out[0] = data[0];
            for i in 1..n {
                if flags[i] {
                    out[i] = data[i];
                } else {
                    out[i] = out[i - 1].wrapping_add(data[i]);
                }
            }
        }
        ScanMode::Exclusive => {
            out[0] = 0;
            for i in 1..n {
                if flags[i] {
                    out[i] = 0;
                } else {
                    out[i] = out[i - 1].wrapping_add(data[i - 1]);
                }
            }
        }
    }

    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════
// Key-based segmented scan
// ═══════════════════════════════════════════════════════════════════

/// Key-based segmented inclusive prefix sum for `f32`.
///
/// A new segment starts whenever `keys[i] != keys[i-1]`.
pub fn scan_by_key_f32(keys: &[u32], data: &[f32], mode: ScanMode) -> Result<Vec<f32>> {
    if keys.len() != data.len() {
        return Err(invalid_args("keys and data must have equal length"));
    }
    if data.is_empty() {
        return Ok(Vec::new());
    }

    // Convert keys to segment-start flags, then delegate.
    let flags = keys_to_flags(keys);
    segmented_scan_f32(data, &flags, mode)
}

/// Key-based segmented inclusive prefix sum for `i32`.
pub fn scan_by_key_i32(keys: &[u32], data: &[i32], mode: ScanMode) -> Result<Vec<i32>> {
    if keys.len() != data.len() {
        return Err(invalid_args("keys and data must have equal length"));
    }
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let flags = keys_to_flags(keys);
    segmented_scan_i32(data, &flags, mode)
}

fn keys_to_flags(keys: &[u32]) -> Vec<bool> {
    let mut flags = vec![false; keys.len()];
    if !keys.is_empty() {
        flags[0] = true; // first element always starts a segment
        for i in 1..keys.len() {
            flags[i] = keys[i] != keys[i - 1];
        }
    }
    flags
}

// ═══════════════════════════════════════════════════════════════════
// Cumulative ops (min, max, product)
// ═══════════════════════════════════════════════════════════════════

/// Cumulative operation on an `f32` slice (running min / max / product).
pub fn cumulative_op_f32(data: &[f32], op: CumulativeOp) -> Result<Vec<f32>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let n = data.len();
    let mut out = vec![0.0f32; n];
    out[0] = data[0];

    match op {
        CumulativeOp::Min => {
            for i in 1..n {
                out[i] = out[i - 1].min(data[i]);
            }
        }
        CumulativeOp::Max => {
            for i in 1..n {
                out[i] = out[i - 1].max(data[i]);
            }
        }
        CumulativeOp::Product => {
            for i in 1..n {
                out[i] = out[i - 1] * data[i];
            }
        }
    }

    Ok(out)
}

/// Cumulative operation on an `i32` slice.
pub fn cumulative_op_i32(data: &[i32], op: CumulativeOp) -> Result<Vec<i32>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let n = data.len();
    let mut out = vec![0i32; n];
    out[0] = data[0];

    match op {
        CumulativeOp::Min => {
            for i in 1..n {
                out[i] = out[i - 1].min(data[i]);
            }
        }
        CumulativeOp::Max => {
            for i in 1..n {
                out[i] = out[i - 1].max(data[i]);
            }
        }
        CumulativeOp::Product => {
            for i in 1..n {
                out[i] = out[i - 1].wrapping_mul(data[i]);
            }
        }
    }

    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── prefix_sum_f32 ─────────────────────────────────────────

    #[test]
    fn test_f32_inclusive_basic() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_f32_exclusive_basic() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let out = prefix_sum_f32(&data, ScanMode::Exclusive).unwrap();
        assert_eq!(out, vec![0.0, 1.0, 3.0, 6.0]);
    }

    #[test]
    fn test_f32_inclusive_single() {
        let out = prefix_sum_f32(&[42.0], ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![42.0]);
    }

    #[test]
    fn test_f32_exclusive_single() {
        let out = prefix_sum_f32(&[42.0], ScanMode::Exclusive).unwrap();
        assert_eq!(out, vec![0.0]);
    }

    #[test]
    fn test_f32_empty() {
        let out = prefix_sum_f32(&[], ScanMode::Inclusive).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_f32_inclusive_two() {
        let out = prefix_sum_f32(&[3.0, 7.0], ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![3.0, 10.0]);
    }

    #[test]
    fn test_f32_exclusive_two() {
        let out = prefix_sum_f32(&[3.0, 7.0], ScanMode::Exclusive).unwrap();
        assert_eq!(out, vec![0.0, 3.0]);
    }

    #[test]
    fn test_f32_all_zeros() {
        let data = vec![0.0f32; 16];
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![0.0; 16]);
    }

    #[test]
    fn test_f32_negatives() {
        let data = [1.0, -2.0, 3.0, -4.0];
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![1.0, -1.0, 2.0, -2.0]);
    }

    #[test]
    fn test_f32_inclusive_exact_8() {
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        let expected: Vec<f32> = vec![1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_f32_inclusive_9_elements() {
        let data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        let mut expected = Vec::new();
        let mut acc = 0.0;
        for &v in &data {
            acc += v;
            expected.push(acc);
        }
        assert_eq!(out, expected);
    }

    #[test]
    fn test_f32_inclusive_16_elements() {
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        let expected = naive_inclusive_f32(&data);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_f32_exclusive_16_elements() {
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let out = prefix_sum_f32(&data, ScanMode::Exclusive).unwrap();
        let expected = naive_exclusive_f32(&data);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_f32_inclusive_large() {
        let data: Vec<f32> = (0..1024).map(|i| (i % 7) as f32).collect();
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        let expected = naive_inclusive_f32(&data);
        for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-2, "mismatch at index {i}: got {a}, expected {b}");
        }
    }

    #[test]
    fn test_f32_exclusive_large() {
        let data: Vec<f32> = (0..1024).map(|i| (i % 11) as f32).collect();
        let out = prefix_sum_f32(&data, ScanMode::Exclusive).unwrap();
        let expected = naive_exclusive_f32(&data);
        for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-2, "mismatch at index {i}: got {a}, expected {b}");
        }
    }

    #[test]
    fn test_f32_inclusive_7_elements() {
        let data: Vec<f32> = (1..=7).map(|x| x as f32).collect();
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, naive_inclusive_f32(&data));
    }

    #[test]
    fn test_f32_exclusive_7_elements() {
        let data: Vec<f32> = (1..=7).map(|x| x as f32).collect();
        let out = prefix_sum_f32(&data, ScanMode::Exclusive).unwrap();
        assert_eq!(out, naive_exclusive_f32(&data));
    }

    #[test]
    fn test_f32_inclusive_ones() {
        let data = vec![1.0f32; 32];
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        let expected: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn test_f32_large_values() {
        let data = [1e10, 1e10, 1e10, 1e10];
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        assert!((out[3] - 4e10).abs() < 1e3);
    }

    #[test]
    fn test_f32_subnormal() {
        let tiny = f32::MIN_POSITIVE / 2.0;
        let data = [tiny, tiny, tiny, tiny];
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        assert!(out[3] >= 0.0);
    }

    // ── prefix_sum_i32 ─────────────────────────────────────────

    #[test]
    fn test_i32_inclusive_basic() {
        let data = [1, 2, 3, 4];
        let out = prefix_sum_i32(&data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![1, 3, 6, 10]);
    }

    #[test]
    fn test_i32_exclusive_basic() {
        let data = [1, 2, 3, 4];
        let out = prefix_sum_i32(&data, ScanMode::Exclusive).unwrap();
        assert_eq!(out, vec![0, 1, 3, 6]);
    }

    #[test]
    fn test_i32_empty() {
        let out = prefix_sum_i32(&[], ScanMode::Inclusive).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_i32_single() {
        let out = prefix_sum_i32(&[99], ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![99]);
    }

    #[test]
    fn test_i32_exclusive_single() {
        let out = prefix_sum_i32(&[99], ScanMode::Exclusive).unwrap();
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn test_i32_negatives() {
        let data = [10, -3, 5, -7, 2];
        let out = prefix_sum_i32(&data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![10, 7, 12, 5, 7]);
    }

    #[test]
    fn test_i32_all_zeros() {
        let data = vec![0i32; 20];
        let out = prefix_sum_i32(&data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![0; 20]);
    }

    #[test]
    fn test_i32_exact_8() {
        let data: Vec<i32> = (1..=8).collect();
        let out = prefix_sum_i32(&data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![1, 3, 6, 10, 15, 21, 28, 36]);
    }

    #[test]
    fn test_i32_inclusive_16() {
        let data: Vec<i32> = (1..=16).collect();
        let out = prefix_sum_i32(&data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, naive_inclusive_i32(&data));
    }

    #[test]
    fn test_i32_exclusive_16() {
        let data: Vec<i32> = (1..=16).collect();
        let out = prefix_sum_i32(&data, ScanMode::Exclusive).unwrap();
        assert_eq!(out, naive_exclusive_i32(&data));
    }

    #[test]
    fn test_i32_large_array() {
        let data: Vec<i32> = (0..2048).map(|i| (i % 13) as i32 - 6).collect();
        let out = prefix_sum_i32(&data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, naive_inclusive_i32(&data));
    }

    #[test]
    fn test_i32_wrapping() {
        let data = [i32::MAX, 1];
        let out = prefix_sum_i32(&data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![i32::MAX, i32::MIN]);
    }

    #[test]
    fn test_i32_exclusive_9() {
        let data: Vec<i32> = (1..=9).collect();
        let out = prefix_sum_i32(&data, ScanMode::Exclusive).unwrap();
        assert_eq!(out, naive_exclusive_i32(&data));
    }

    // ── parallel_scan ──────────────────────────────────────────

    #[test]
    fn test_parallel_scan_f32_basic() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let out = parallel_scan_f32(&data).unwrap();
        assert_eq!(out, vec![0.0, 1.0, 3.0, 6.0]);
    }

    #[test]
    fn test_parallel_scan_f32_empty() {
        let out = parallel_scan_f32(&[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_parallel_scan_f32_single() {
        let out = parallel_scan_f32(&[5.0]).unwrap();
        assert_eq!(out, vec![0.0]);
    }

    #[test]
    fn test_parallel_scan_f32_large() {
        let data: Vec<f32> = (0..2048).map(|i| (i % 5) as f32).collect();
        let out = parallel_scan_f32(&data).unwrap();
        let expected = naive_exclusive_f32(&data);
        for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-1, "parallel_scan mismatch at {i}: got {a}, expected {b}");
        }
    }

    #[test]
    fn test_parallel_scan_f32_medium() {
        let data: Vec<f32> = (0..512).map(|i| (i % 3) as f32 + 0.5).collect();
        let out = parallel_scan_f32(&data).unwrap();
        let expected = naive_exclusive_f32(&data);
        for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-1, "mismatch at {i}: got {a}, expected {b}");
        }
    }

    #[test]
    fn test_parallel_scan_i32_basic() {
        let data = [1, 2, 3, 4];
        let out = parallel_scan_i32(&data).unwrap();
        assert_eq!(out, vec![0, 1, 3, 6]);
    }

    #[test]
    fn test_parallel_scan_i32_empty() {
        let out = parallel_scan_i32(&[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_parallel_scan_i32_large() {
        let data: Vec<i32> = (0..2048).map(|i| (i % 7) as i32).collect();
        let out = parallel_scan_i32(&data).unwrap();
        assert_eq!(out, naive_exclusive_i32(&data));
    }

    // ── segmented_scan ─────────────────────────────────────────

    #[test]
    fn test_segmented_f32_inclusive_basic() {
        let data = [1.0, 2.0, 3.0, 10.0, 20.0];
        let flags = [true, false, false, true, false];
        let out = segmented_scan_f32(&data, &flags, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![1.0, 3.0, 6.0, 10.0, 30.0]);
    }

    #[test]
    fn test_segmented_f32_exclusive_basic() {
        let data = [1.0, 2.0, 3.0, 10.0, 20.0];
        let flags = [true, false, false, true, false];
        let out = segmented_scan_f32(&data, &flags, ScanMode::Exclusive).unwrap();
        assert_eq!(out, vec![0.0, 1.0, 3.0, 0.0, 10.0]);
    }

    #[test]
    fn test_segmented_f32_single_segment() {
        let data = [1.0, 2.0, 3.0];
        let flags = [true, false, false];
        let out = segmented_scan_f32(&data, &flags, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![1.0, 3.0, 6.0]);
    }

    #[test]
    fn test_segmented_f32_all_segments() {
        let data = [5.0, 10.0, 15.0];
        let flags = [true, true, true];
        let out = segmented_scan_f32(&data, &flags, ScanMode::Inclusive).unwrap();
        // Every element starts a new segment, so each is its own value.
        assert_eq!(out, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_segmented_f32_empty() {
        let out = segmented_scan_f32(&[], &[], ScanMode::Inclusive).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_segmented_f32_mismatched_lengths() {
        let result = segmented_scan_f32(&[1.0, 2.0], &[true], ScanMode::Inclusive);
        assert!(result.is_err());
    }

    #[test]
    fn test_segmented_i32_inclusive() {
        let data = [1, 2, 3, 10, 20];
        let flags = [true, false, false, true, false];
        let out = segmented_scan_i32(&data, &flags, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![1, 3, 6, 10, 30]);
    }

    #[test]
    fn test_segmented_i32_exclusive() {
        let data = [1, 2, 3, 10, 20];
        let flags = [true, false, false, true, false];
        let out = segmented_scan_i32(&data, &flags, ScanMode::Exclusive).unwrap();
        assert_eq!(out, vec![0, 1, 3, 0, 10]);
    }

    #[test]
    fn test_segmented_i32_mismatched() {
        let result = segmented_scan_i32(&[1, 2], &[true], ScanMode::Inclusive);
        assert!(result.is_err());
    }

    #[test]
    fn test_segmented_i32_all_flags() {
        let data = [3, 7, 11];
        let flags = [true, true, true];
        let out = segmented_scan_i32(&data, &flags, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![3, 7, 11]);
    }

    #[test]
    fn test_segmented_f32_two_segments_exclusive() {
        let data = [2.0, 4.0, 6.0, 8.0];
        let flags = [true, false, true, false];
        let out = segmented_scan_f32(&data, &flags, ScanMode::Exclusive).unwrap();
        assert_eq!(out, vec![0.0, 2.0, 0.0, 6.0]);
    }

    #[test]
    fn test_segmented_i32_empty() {
        let out = segmented_scan_i32(&[], &[], ScanMode::Inclusive).unwrap();
        assert!(out.is_empty());
    }

    // ── scan_by_key ────────────────────────────────────────────

    #[test]
    fn test_scan_by_key_f32_inclusive() {
        let keys = [0, 0, 0, 1, 1];
        let data = [1.0, 2.0, 3.0, 10.0, 20.0];
        let out = scan_by_key_f32(&keys, &data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![1.0, 3.0, 6.0, 10.0, 30.0]);
    }

    #[test]
    fn test_scan_by_key_f32_exclusive() {
        let keys = [0, 0, 0, 1, 1];
        let data = [1.0, 2.0, 3.0, 10.0, 20.0];
        let out = scan_by_key_f32(&keys, &data, ScanMode::Exclusive).unwrap();
        assert_eq!(out, vec![0.0, 1.0, 3.0, 0.0, 10.0]);
    }

    #[test]
    fn test_scan_by_key_f32_single_key() {
        let keys = [5, 5, 5, 5];
        let data = [1.0, 2.0, 3.0, 4.0];
        let out = scan_by_key_f32(&keys, &data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_scan_by_key_f32_all_different_keys() {
        let keys = [0, 1, 2, 3];
        let data = [10.0, 20.0, 30.0, 40.0];
        let out = scan_by_key_f32(&keys, &data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_scan_by_key_f32_empty() {
        let out = scan_by_key_f32(&[], &[], ScanMode::Inclusive).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_scan_by_key_f32_mismatched() {
        let result = scan_by_key_f32(&[0, 1], &[1.0], ScanMode::Inclusive);
        assert!(result.is_err());
    }

    #[test]
    fn test_scan_by_key_i32_inclusive() {
        let keys = [0, 0, 1, 1, 1];
        let data = [1, 2, 10, 20, 30];
        let out = scan_by_key_i32(&keys, &data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![1, 3, 10, 30, 60]);
    }

    #[test]
    fn test_scan_by_key_i32_exclusive() {
        let keys = [0, 0, 1, 1, 1];
        let data = [1, 2, 10, 20, 30];
        let out = scan_by_key_i32(&keys, &data, ScanMode::Exclusive).unwrap();
        assert_eq!(out, vec![0, 1, 0, 10, 30]);
    }

    #[test]
    fn test_scan_by_key_i32_empty() {
        let out = scan_by_key_i32(&[], &[], ScanMode::Inclusive).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_scan_by_key_i32_mismatched() {
        let result = scan_by_key_i32(&[0], &[1, 2], ScanMode::Inclusive);
        assert!(result.is_err());
    }

    #[test]
    fn test_scan_by_key_i32_three_segments() {
        let keys = [0, 0, 1, 1, 2, 2];
        let data = [1, 1, 1, 1, 1, 1];
        let out = scan_by_key_i32(&keys, &data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![1, 2, 1, 2, 1, 2]);
    }

    // ── cumulative ops ─────────────────────────────────────────

    #[test]
    fn test_cumulative_min_f32() {
        let data = [5.0, 3.0, 7.0, 1.0, 4.0];
        let out = cumulative_op_f32(&data, CumulativeOp::Min).unwrap();
        assert_eq!(out, vec![5.0, 3.0, 3.0, 1.0, 1.0]);
    }

    #[test]
    fn test_cumulative_max_f32() {
        let data = [1.0, 5.0, 3.0, 8.0, 2.0];
        let out = cumulative_op_f32(&data, CumulativeOp::Max).unwrap();
        assert_eq!(out, vec![1.0, 5.0, 5.0, 8.0, 8.0]);
    }

    #[test]
    fn test_cumulative_product_f32() {
        let data = [2.0, 3.0, 4.0];
        let out = cumulative_op_f32(&data, CumulativeOp::Product).unwrap();
        assert_eq!(out, vec![2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_cumulative_min_f32_empty() {
        let out = cumulative_op_f32(&[], CumulativeOp::Min).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_cumulative_min_f32_single() {
        let out = cumulative_op_f32(&[42.0], CumulativeOp::Min).unwrap();
        assert_eq!(out, vec![42.0]);
    }

    #[test]
    fn test_cumulative_max_f32_descending() {
        let data = [10.0, 8.0, 6.0, 4.0, 2.0];
        let out = cumulative_op_f32(&data, CumulativeOp::Max).unwrap();
        assert_eq!(out, vec![10.0, 10.0, 10.0, 10.0, 10.0]);
    }

    #[test]
    fn test_cumulative_product_f32_with_zero() {
        let data = [3.0, 0.0, 5.0];
        let out = cumulative_op_f32(&data, CumulativeOp::Product).unwrap();
        assert_eq!(out, vec![3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cumulative_product_f32_negatives() {
        let data = [2.0, -3.0, 4.0];
        let out = cumulative_op_f32(&data, CumulativeOp::Product).unwrap();
        assert_eq!(out, vec![2.0, -6.0, -24.0]);
    }

    #[test]
    fn test_cumulative_min_i32() {
        let data = [5, 3, 7, 1, 4];
        let out = cumulative_op_i32(&data, CumulativeOp::Min).unwrap();
        assert_eq!(out, vec![5, 3, 3, 1, 1]);
    }

    #[test]
    fn test_cumulative_max_i32() {
        let data = [1, 5, 3, 8, 2];
        let out = cumulative_op_i32(&data, CumulativeOp::Max).unwrap();
        assert_eq!(out, vec![1, 5, 5, 8, 8]);
    }

    #[test]
    fn test_cumulative_product_i32() {
        let data = [2, 3, 4];
        let out = cumulative_op_i32(&data, CumulativeOp::Product).unwrap();
        assert_eq!(out, vec![2, 6, 24]);
    }

    #[test]
    fn test_cumulative_min_i32_empty() {
        let out = cumulative_op_i32(&[], CumulativeOp::Min).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_cumulative_max_i32_single() {
        let out = cumulative_op_i32(&[7], CumulativeOp::Max).unwrap();
        assert_eq!(out, vec![7]);
    }

    #[test]
    fn test_cumulative_product_i32_wrapping() {
        // Large values that would overflow — wrapping_mul semantics.
        let data = [i32::MAX, 2];
        let out = cumulative_op_i32(&data, CumulativeOp::Product).unwrap();
        assert_eq!(out[0], i32::MAX);
        assert_eq!(out[1], i32::MAX.wrapping_mul(2));
    }

    // ── scalar fallback agreement ──────────────────────────────

    #[test]
    fn test_scalar_f32_inclusive_matches_naive() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
        let out = prefix_sum_f32_scalar(&data, ScanMode::Inclusive);
        let expected = naive_inclusive_f32(&data);
        for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "scalar mismatch at {i}: got {a}, expected {b}");
        }
    }

    #[test]
    fn test_scalar_f32_exclusive_matches_naive() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
        let out = prefix_sum_f32_scalar(&data, ScanMode::Exclusive);
        let expected = naive_exclusive_f32(&data);
        for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "scalar mismatch at {i}: got {a}, expected {b}");
        }
    }

    #[test]
    fn test_scalar_i32_inclusive_matches_naive() {
        let data: Vec<i32> = (0..100).collect();
        let out = prefix_sum_i32_scalar(&data, ScanMode::Inclusive);
        assert_eq!(out, naive_inclusive_i32(&data));
    }

    #[test]
    fn test_scalar_i32_exclusive_matches_naive() {
        let data: Vec<i32> = (0..100).collect();
        let out = prefix_sum_i32_scalar(&data, ScanMode::Exclusive);
        assert_eq!(out, naive_exclusive_i32(&data));
    }

    // ── large array stress tests ───────────────────────────────

    #[test]
    fn test_f32_inclusive_4096() {
        let data: Vec<f32> = (0..4096).map(|i| (i % 17) as f32).collect();
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        let expected = naive_inclusive_f32(&data);
        for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1.0, "4096-elem mismatch at {i}: got {a}, expected {b}");
        }
    }

    #[test]
    fn test_i32_inclusive_4096() {
        let data: Vec<i32> = (0..4096).map(|i| (i % 17) as i32).collect();
        let out = prefix_sum_i32(&data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, naive_inclusive_i32(&data));
    }

    #[test]
    fn test_f32_exclusive_4096() {
        let data: Vec<f32> = (0..4096).map(|i| (i % 23) as f32).collect();
        let out = prefix_sum_f32(&data, ScanMode::Exclusive).unwrap();
        let expected = naive_exclusive_f32(&data);
        for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1.0, "4096-elem excl mismatch at {i}: got {a}, expected {b}");
        }
    }

    #[test]
    fn test_parallel_scan_f32_4096() {
        let data: Vec<f32> = (0..4096).map(|i| (i % 7) as f32).collect();
        let out = parallel_scan_f32(&data).unwrap();
        let expected = naive_exclusive_f32(&data);
        for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1.0, "parallel 4096 mismatch at {i}: got {a}, expected {b}");
        }
    }

    #[test]
    fn test_parallel_scan_i32_4096() {
        let data: Vec<i32> = (0..4096).map(|i| (i % 11) as i32).collect();
        let out = parallel_scan_i32(&data).unwrap();
        assert_eq!(out, naive_exclusive_i32(&data));
    }

    // ── numerical accuracy ─────────────────────────────────────

    #[test]
    fn test_f32_accuracy_small_values() {
        let data = vec![0.1f32; 100];
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        // Allow some FP accumulation error.
        assert!((out[99] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_f32_accuracy_alternating() {
        let data: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        // Alternating +1/-1: even indices → 0 except first, odd → 1.
        assert!((out[63] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_f32_monotone_inclusive() {
        // Inclusive prefix sum of positive data must be monotonically
        // non-decreasing.
        let data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 + 0.1).collect();
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        for i in 1..out.len() {
            assert!(out[i] >= out[i - 1], "not monotone at {i}: {} < {}", out[i], out[i - 1]);
        }
    }

    #[test]
    fn test_f32_exclusive_first_zero() {
        // Exclusive prefix sum must always start with 0.
        let data: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let out = prefix_sum_f32(&data, ScanMode::Exclusive).unwrap();
        assert_eq!(out[0], 0.0);
    }

    // ── keys_to_flags ──────────────────────────────────────────

    #[test]
    fn test_keys_to_flags_basic() {
        let keys = [0, 0, 1, 1, 2];
        let flags = keys_to_flags(&keys);
        assert_eq!(flags, vec![true, false, true, false, true]);
    }

    #[test]
    fn test_keys_to_flags_all_same() {
        let keys = [5, 5, 5, 5];
        let flags = keys_to_flags(&keys);
        assert_eq!(flags, vec![true, false, false, false]);
    }

    #[test]
    fn test_keys_to_flags_all_different() {
        let keys = [1, 2, 3, 4];
        let flags = keys_to_flags(&keys);
        assert_eq!(flags, vec![true, true, true, true]);
    }

    #[test]
    fn test_keys_to_flags_empty() {
        let flags = keys_to_flags(&[]);
        assert!(flags.is_empty());
    }

    // ── edge case: length = 3 (not a multiple of 8) ────────────

    #[test]
    fn test_f32_inclusive_3() {
        let out = prefix_sum_f32(&[10.0, 20.0, 30.0], ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![10.0, 30.0, 60.0]);
    }

    #[test]
    fn test_f32_exclusive_3() {
        let out = prefix_sum_f32(&[10.0, 20.0, 30.0], ScanMode::Exclusive).unwrap();
        assert_eq!(out, vec![0.0, 10.0, 30.0]);
    }

    // ── cumulative op edge cases ───────────────────────────────

    #[test]
    fn test_cumulative_min_f32_already_sorted() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let out = cumulative_op_f32(&data, CumulativeOp::Min).unwrap();
        assert_eq!(out, vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_cumulative_max_f32_already_sorted() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let out = cumulative_op_f32(&data, CumulativeOp::Max).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_cumulative_product_f32_ones() {
        let data = vec![1.0f32; 10];
        let out = cumulative_op_f32(&data, CumulativeOp::Product).unwrap();
        assert_eq!(out, vec![1.0; 10]);
    }

    #[test]
    fn test_cumulative_min_i32_negative() {
        let data = [0, -1, -2, -3, -4];
        let out = cumulative_op_i32(&data, CumulativeOp::Min).unwrap();
        assert_eq!(out, vec![0, -1, -2, -3, -4]);
    }

    #[test]
    fn test_cumulative_max_i32_all_same() {
        let data = [7, 7, 7, 7];
        let out = cumulative_op_i32(&data, CumulativeOp::Max).unwrap();
        assert_eq!(out, vec![7, 7, 7, 7]);
    }

    #[test]
    fn test_cumulative_product_i32_with_zero() {
        let data = [5, 0, 3];
        let out = cumulative_op_i32(&data, CumulativeOp::Product).unwrap();
        assert_eq!(out, vec![5, 0, 0]);
    }

    // ── segmented scan edge: single-element segments ───────────

    #[test]
    fn test_segmented_f32_single_element_segments() {
        let data = [1.0, 2.0, 3.0];
        let flags = [true, true, true];
        let out = segmented_scan_f32(&data, &flags, ScanMode::Exclusive).unwrap();
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_segmented_i32_single_element_segments() {
        let data = [1, 2, 3];
        let flags = [true, true, true];
        let out = segmented_scan_i32(&data, &flags, ScanMode::Exclusive).unwrap();
        assert_eq!(out, vec![0, 0, 0]);
    }

    // ── scan_by_key larger example ─────────────────────────────

    #[test]
    fn test_scan_by_key_f32_repeated_keys() {
        // keys: [A A B B B A A]
        let keys = [0, 0, 1, 1, 1, 0, 0];
        let data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let out = scan_by_key_f32(&keys, &data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_scan_by_key_i32_repeated_keys() {
        let keys = [0, 0, 1, 1, 1, 0, 0];
        let data = [1, 1, 1, 1, 1, 1, 1];
        let out = scan_by_key_i32(&keys, &data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, vec![1, 2, 1, 2, 3, 1, 2]);
    }

    // ── additional SIMD boundary tests ─────────────────────────

    #[test]
    fn test_f32_inclusive_15() {
        let data: Vec<f32> = (1..=15).map(|x| x as f32).collect();
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, naive_inclusive_f32(&data));
    }

    #[test]
    fn test_f32_inclusive_17() {
        let data: Vec<f32> = (1..=17).map(|x| x as f32).collect();
        let out = prefix_sum_f32(&data, ScanMode::Inclusive).unwrap();
        let expected = naive_inclusive_f32(&data);
        for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "17-elem mismatch at {i}: got {a}, expected {b}");
        }
    }

    #[test]
    fn test_i32_inclusive_15() {
        let data: Vec<i32> = (1..=15).collect();
        let out = prefix_sum_i32(&data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, naive_inclusive_i32(&data));
    }

    #[test]
    fn test_i32_inclusive_17() {
        let data: Vec<i32> = (1..=17).collect();
        let out = prefix_sum_i32(&data, ScanMode::Inclusive).unwrap();
        assert_eq!(out, naive_inclusive_i32(&data));
    }

    // ── helpers ────────────────────────────────────────────────

    fn naive_inclusive_f32(data: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(data.len());
        let mut acc = 0.0f32;
        for &v in data {
            acc += v;
            out.push(acc);
        }
        out
    }

    fn naive_exclusive_f32(data: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(data.len());
        let mut acc = 0.0f32;
        for &v in data {
            out.push(acc);
            acc += v;
        }
        out
    }

    fn naive_inclusive_i32(data: &[i32]) -> Vec<i32> {
        let mut out = Vec::with_capacity(data.len());
        let mut acc = 0i32;
        for &v in data {
            acc = acc.wrapping_add(v);
            out.push(acc);
        }
        out
    }

    fn naive_exclusive_i32(data: &[i32]) -> Vec<i32> {
        let mut out = Vec::with_capacity(data.len());
        let mut acc = 0i32;
        for &v in data {
            out.push(acc);
            acc = acc.wrapping_add(v);
        }
        out
    }
}
