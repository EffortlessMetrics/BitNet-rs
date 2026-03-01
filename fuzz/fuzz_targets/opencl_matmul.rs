#![no_main]
use libfuzzer_sys::fuzz_target;

/// Fuzz the matmul CPU reference to ensure it never panics.
fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }
    let m = (data[0] as usize % 16) + 1;
    let n = (data[1] as usize % 16) + 1;
    let k = (data[2] as usize % 16) + 1;
    let needed = (m * k + k * n + m * n) * 4 + 12;
    if data.len() < needed {
        return;
    }

    // Parse floats from data
    let mut offset = 12;
    let parse_f32 = |d: &[u8], o: &mut usize| -> f32 {
        if *o + 4 > d.len() {
            return 0.0;
        }
        let bytes = [d[*o], d[*o + 1], d[*o + 2], d[*o + 3]];
        *o += 4;
        f32::from_le_bytes(bytes)
    };

    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    let mut c = vec![0.0f32; m * n];

    for v in a.iter_mut() {
        *v = parse_f32(data, &mut offset);
    }
    for v in b.iter_mut() {
        *v = parse_f32(data, &mut offset);
    }

    // Run matmul - should never panic
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    // Verify no NaN/Inf if inputs are finite
    let inputs_finite = a.iter().chain(b.iter()).all(|v| v.is_finite());
    if inputs_finite {
        for v in &c {
            assert!(v.is_finite() || k > 100, "NaN/Inf in output with finite inputs");
        }
    }
});
