use std::arch::x86_64::*;

#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
unsafe fn test_avx512() {
    let scale_factor_vec = _mm512_set1_ps(128.0);
    let offset_vec = _mm512_set1_ps(128.0);

    let mut data: [f32; 16] = [1.0; 16];
    let data_vec = _mm512_loadu_ps(data.as_ptr());
    let scaled = _mm512_mul_ps(data_vec, scale_factor_vec);
    let offset = _mm512_add_ps(scaled, offset_vec);
    let indices = _mm512_cvtps_epi32(offset);

    let mut result: [i32; 16] = [0; 16];
    _mm512_storeu_si512(result.as_mut_ptr() as *mut _, indices);
}

fn main() {
    unsafe {
        test_avx512();
    }
}
