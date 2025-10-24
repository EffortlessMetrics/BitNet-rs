#!/usr/bin/env python3
"""
Apply FMA tiling optimization to QK256 AVX2 dequantization kernel.

Phase 2: Replace _mm256_mul_ps with _mm256_fmadd_ps and add 8-tile unrolling.
"""

import re

def apply_fma_tiling(content):
    """Apply FMA tiling optimization to the dequantization kernel."""

    # Step 1: Update the target_feature attribute to include "fma"
    content = re.sub(
        r'#\[target_feature\(enable = "avx2"\)\]\s*#\[allow\(unsafe_op_in_unsafe_fn\)\]\s*unsafe fn dequantize_qk256_avx2\(',
        '#[target_feature(enable = "avx2", enable = "fma")]\n    #[allow(unsafe_op_in_unsafe_fn)]\n    unsafe fn dequantize_qk256_avx2(',
        content
    )

    # Step 2: Update the docstring to mention Phase 2
    content = re.sub(
        r'# Algorithm \(Phase 1: Nibble LUT Unpack\)',
        '# Algorithm (Phase 1+2: Nibble LUT Unpack + FMA Tiling)',
        content
    )

    content = re.sub(
        r'- Apply scales with AVX2 FMA',
        '- Apply scales with AVX2 FMA and 8-tile unrolling for ILP',
        content
    )

    # Step 3: Replace the scale application loop with FMA-tiled version
    pattern = r'(\s+)// SIMD conversion: codes → f32 using LUT, then scale\s+let scale_vec = _mm256_set1_ps\(\*scale\);.*?elem_idx \+= 8;\s+\}'

    replacement = r'''\1// Phase 2: FMA-tiled SIMD conversion with 8-way unrolling
\1// Replaces simple _mm256_mul_ps with FMA for better instruction-level parallelism
\1let scale_vec = _mm256_set1_ps(*scale);
\1let zero = _mm256_setzero_ps();

\1let mut elem_idx = 0;
\1const TILE_SIZE: usize = 64; // 8 vectors × 8 elements for ILP

\1// Process in 64-element tiles (8 vectors) for better ILP and FMA utilization
\1while elem_idx + TILE_SIZE <= QK256 {
\1    // Unroll 8 iterations with separate accumulators to expose parallelism

\1    // Tile 0: elements [elem_idx..elem_idx+8]
\1    let weights0 = [
\1        LUT[codes[elem_idx] as usize],
\1        LUT[codes[elem_idx + 1] as usize],
\1        LUT[codes[elem_idx + 2] as usize],
\1        LUT[codes[elem_idx + 3] as usize],
\1        LUT[codes[elem_idx + 4] as usize],
\1        LUT[codes[elem_idx + 5] as usize],
\1        LUT[codes[elem_idx + 6] as usize],
\1        LUT[codes[elem_idx + 7] as usize],
\1    ];
\1    let w_vec0 = _mm256_loadu_ps(weights0.as_ptr());
\1    let scaled0 = _mm256_fmadd_ps(w_vec0, scale_vec, zero); // w * s + 0

\1    // Tile 1: elements [elem_idx+8..elem_idx+16]
\1    let weights1 = [
\1        LUT[codes[elem_idx + 8] as usize],
\1        LUT[codes[elem_idx + 9] as usize],
\1        LUT[codes[elem_idx + 10] as usize],
\1        LUT[codes[elem_idx + 11] as usize],
\1        LUT[codes[elem_idx + 12] as usize],
\1        LUT[codes[elem_idx + 13] as usize],
\1        LUT[codes[elem_idx + 14] as usize],
\1        LUT[codes[elem_idx + 15] as usize],
\1    ];
\1    let w_vec1 = _mm256_loadu_ps(weights1.as_ptr());
\1    let scaled1 = _mm256_fmadd_ps(w_vec1, scale_vec, zero);

\1    // Tile 2: elements [elem_idx+16..elem_idx+24]
\1    let weights2 = [
\1        LUT[codes[elem_idx + 16] as usize],
\1        LUT[codes[elem_idx + 17] as usize],
\1        LUT[codes[elem_idx + 18] as usize],
\1        LUT[codes[elem_idx + 19] as usize],
\1        LUT[codes[elem_idx + 20] as usize],
\1        LUT[codes[elem_idx + 21] as usize],
\1        LUT[codes[elem_idx + 22] as usize],
\1        LUT[codes[elem_idx + 23] as usize],
\1    ];
\1    let w_vec2 = _mm256_loadu_ps(weights2.as_ptr());
\1    let scaled2 = _mm256_fmadd_ps(w_vec2, scale_vec, zero);

\1    // Tile 3: elements [elem_idx+24..elem_idx+32]
\1    let weights3 = [
\1        LUT[codes[elem_idx + 24] as usize],
\1        LUT[codes[elem_idx + 25] as usize],
\1        LUT[codes[elem_idx + 26] as usize],
\1        LUT[codes[elem_idx + 27] as usize],
\1        LUT[codes[elem_idx + 28] as usize],
\1        LUT[codes[elem_idx + 29] as usize],
\1        LUT[codes[elem_idx + 30] as usize],
\1        LUT[codes[elem_idx + 31] as usize],
\1    ];
\1    let w_vec3 = _mm256_loadu_ps(weights3.as_ptr());
\1    let scaled3 = _mm256_fmadd_ps(w_vec3, scale_vec, zero);

\1    // Tile 4: elements [elem_idx+32..elem_idx+40]
\1    let weights4 = [
\1        LUT[codes[elem_idx + 32] as usize],
\1        LUT[codes[elem_idx + 33] as usize],
\1        LUT[codes[elem_idx + 34] as usize],
\1        LUT[codes[elem_idx + 35] as usize],
\1        LUT[codes[elem_idx + 36] as usize],
\1        LUT[codes[elem_idx + 37] as usize],
\1        LUT[codes[elem_idx + 38] as usize],
\1        LUT[codes[elem_idx + 39] as usize],
\1    ];
\1    let w_vec4 = _mm256_loadu_ps(weights4.as_ptr());
\1    let scaled4 = _mm256_fmadd_ps(w_vec4, scale_vec, zero);

\1    // Tile 5: elements [elem_idx+40..elem_idx+48]
\1    let weights5 = [
\1        LUT[codes[elem_idx + 40] as usize],
\1        LUT[codes[elem_idx + 41] as usize],
\1        LUT[codes[elem_idx + 42] as usize],
\1        LUT[codes[elem_idx + 43] as usize],
\1        LUT[codes[elem_idx + 44] as usize],
\1        LUT[codes[elem_idx + 45] as usize],
\1        LUT[codes[elem_idx + 46] as usize],
\1        LUT[codes[elem_idx + 47] as usize],
\1    ];
\1    let w_vec5 = _mm256_loadu_ps(weights5.as_ptr());
\1    let scaled5 = _mm256_fmadd_ps(w_vec5, scale_vec, zero);

\1    // Tile 6: elements [elem_idx+48..elem_idx+56]
\1    let weights6 = [
\1        LUT[codes[elem_idx + 48] as usize],
\1        LUT[codes[elem_idx + 49] as usize],
\1        LUT[codes[elem_idx + 50] as usize],
\1        LUT[codes[elem_idx + 51] as usize],
\1        LUT[codes[elem_idx + 52] as usize],
\1        LUT[codes[elem_idx + 53] as usize],
\1        LUT[codes[elem_idx + 54] as usize],
\1        LUT[codes[elem_idx + 55] as usize],
\1    ];
\1    let w_vec6 = _mm256_loadu_ps(weights6.as_ptr());
\1    let scaled6 = _mm256_fmadd_ps(w_vec6, scale_vec, zero);

\1    // Tile 7: elements [elem_idx+56..elem_idx+64]
\1    let weights7 = [
\1        LUT[codes[elem_idx + 56] as usize],
\1        LUT[codes[elem_idx + 57] as usize],
\1        LUT[codes[elem_idx + 58] as usize],
\1        LUT[codes[elem_idx + 59] as usize],
\1        LUT[codes[elem_idx + 60] as usize],
\1        LUT[codes[elem_idx + 61] as usize],
\1        LUT[codes[elem_idx + 62] as usize],
\1        LUT[codes[elem_idx + 63] as usize],
\1    ];
\1    let w_vec7 = _mm256_loadu_ps(weights7.as_ptr());
\1    let scaled7 = _mm256_fmadd_ps(w_vec7, scale_vec, zero);

\1    // Store all 8 tiles (64 elements total)
\1    let out_ptr = output.as_mut_ptr().add(block_start + elem_idx);
\1    _mm256_storeu_ps(out_ptr, scaled0);
\1    _mm256_storeu_ps(out_ptr.add(8), scaled1);
\1    _mm256_storeu_ps(out_ptr.add(16), scaled2);
\1    _mm256_storeu_ps(out_ptr.add(24), scaled3);
\1    _mm256_storeu_ps(out_ptr.add(32), scaled4);
\1    _mm256_storeu_ps(out_ptr.add(40), scaled5);
\1    _mm256_storeu_ps(out_ptr.add(48), scaled6);
\1    _mm256_storeu_ps(out_ptr.add(56), scaled7);

\1    elem_idx += TILE_SIZE;
\1}

\1// Process remaining elements in smaller 8-element chunks with FMA
\1while elem_idx + 8 <= QK256 {
\1    let weights = [
\1        LUT[codes[elem_idx] as usize],
\1        LUT[codes[elem_idx + 1] as usize],
\1        LUT[codes[elem_idx + 2] as usize],
\1        LUT[codes[elem_idx + 3] as usize],
\1        LUT[codes[elem_idx + 4] as usize],
\1        LUT[codes[elem_idx + 5] as usize],
\1        LUT[codes[elem_idx + 6] as usize],
\1        LUT[codes[elem_idx + 7] as usize],
\1    ];
\1    let w_vec = _mm256_loadu_ps(weights.as_ptr());
\1    let scaled = _mm256_fmadd_ps(w_vec, scale_vec, zero);
\1    let out_ptr = output.as_mut_ptr().add(block_start + elem_idx);
\1    _mm256_storeu_ps(out_ptr, scaled);
\1    elem_idx += 8;
\1}'''

    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    return content

def main():
    input_file = 'crates/bitnet-kernels/src/cpu/x86.rs'

    with open(input_file, 'r') as f:
        content = f.read()

    modified_content = apply_fma_tiling(content)

    with open(input_file, 'w') as f:
        f.write(modified_content)

    print("✅ FMA tiling optimization applied successfully!")
    print("Changes:")
    print("  - Updated #[target_feature] to include 'fma'")
    print("  - Updated docstring to mention Phase 2")
    print("  - Replaced _mm256_mul_ps with _mm256_fmadd_ps")
    print("  - Added 8-tile unrolling for better ILP")

if __name__ == '__main__':
    main()
