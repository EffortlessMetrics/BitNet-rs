#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled, add_residual_with_dropout};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ResidualInput {
    len_hint: u8,
    scale: u8,
    a_raw: Vec<u8>,
    b_raw: Vec<u8>,
    mask_raw: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    data.chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: ResidualInput| {
    let n = (input.len_hint as usize % 64) + 1;
    let a = bytes_to_f32(&input.a_raw, n);
    let b = bytes_to_f32(&input.b_raw, n);

    if a.len() < n || b.len() < n {
        return;
    }
    if !a.iter().chain(b.iter()).all(|x| x.is_finite()) {
        return;
    }

    // --- add_residual ---
    {
        let mut out = a[..n].to_vec();
        if let Ok(()) = add_residual(&mut out, &b[..n]) {
            assert_eq!(out.len(), n);
            for (i, &v) in out.iter().enumerate() {
                assert!(v.is_finite(), "add_residual non-finite at {i}: {v}");
            }
        }
    }

    // --- add_residual_scaled ---
    {
        let scale = (input.scale as f32 - 128.0) / 32.0;
        if scale.is_finite() {
            let mut out = a[..n].to_vec();
            if let Ok(()) = add_residual_scaled(&mut out, &b[..n], scale) {
                assert_eq!(out.len(), n);
            }
        }
    }

    // --- add_residual_with_dropout ---
    {
        let mask: Vec<bool> = input.mask_raw.iter().take(n).map(|&b| b & 1 != 0).collect();
        if mask.len() >= n {
            let mut out = a[..n].to_vec();
            if let Ok(()) = add_residual_with_dropout(&mut out, &b[..n], &mask[..n]) {
                assert_eq!(out.len(), n);
                for (i, &v) in out.iter().enumerate() {
                    assert!(v.is_finite(), "dropout residual non-finite at {i}: {v}");
                }
            }
        }
    }

    // --- length mismatch should return Err ---
    if n > 1 {
        let mut short = vec![0.0f32; n - 1];
        assert!(add_residual(&mut short, &b[..n]).is_err());
    }
});
