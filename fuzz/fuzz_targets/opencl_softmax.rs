#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let cols = (data[0] as usize % 32) + 1;
    let needed = cols * 4 + 8;
    if data.len() < needed {
        return;
    }

    let mut input = vec![0.0f32; cols];
    let mut output = vec![0.0f32; cols];
    let mut offset = 8;

    for v in input.iter_mut() {
        if offset + 4 > data.len() {
            break;
        }
        let bytes = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
        *v = f32::from_le_bytes(bytes);
        offset += 4;
    }

    // Skip if inputs contain NaN/Inf
    if !input.iter().all(|v| v.is_finite()) {
        return;
    }

    // Run softmax
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for i in 0..cols {
        let e = (input[i] - max_val).exp();
        output[i] = e;
        sum += e;
    }
    if sum > 0.0 {
        for v in output.iter_mut() {
            *v /= sum;
        }
    }

    // Verify: sum ≈ 1.0, all non-negative
    if sum > 0.0 {
        let total: f32 = output.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "Softmax sum = {total}");
        for v in &output {
            assert!(*v >= 0.0, "Negative softmax output");
        }
    }
});
