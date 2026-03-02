#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::gating::{GatingType, apply_gating, geglu, reglu, swiglu};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct GatingInput {
    len_hint: u8,
    gating_type: u8,
    gate_raw: Vec<u8>,
    up_raw: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    data.chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: GatingInput| {
    let n = (input.len_hint as usize % 64) + 1;
    let gate = bytes_to_f32(&input.gate_raw, n);
    let up = bytes_to_f32(&input.up_raw, n);

    if gate.len() < n || up.len() < n {
        return;
    }
    if !gate[..n].iter().chain(up[..n].iter()).all(|x| x.is_finite()) {
        return;
    }

    let gate = &gate[..n];
    let up = &up[..n];
    let output = vec![0.0f32; n];

    // --- swiglu ---
    {
        let mut out = output.clone();
        if let Ok(()) = swiglu(gate, up, &mut out) {
            assert_eq!(out.len(), n);
            for (i, &v) in out.iter().enumerate() {
                assert!(v.is_finite(), "swiglu non-finite at {i}: {v}");
            }
        }
    }

    // --- geglu ---
    {
        let mut out = output.clone();
        if let Ok(()) = geglu(gate, up, &mut out) {
            assert_eq!(out.len(), n);
            for (i, &v) in out.iter().enumerate() {
                assert!(v.is_finite(), "geglu non-finite at {i}: {v}");
            }
        }
    }

    // --- reglu ---
    {
        let mut out = output.clone();
        if let Ok(()) = reglu(gate, up, &mut out) {
            assert_eq!(out.len(), n);
            for (i, &v) in out.iter().enumerate() {
                assert!(v.is_finite(), "reglu non-finite at {i}: {v}");
                // ReGLU output should be >= 0 when gate >= 0
                // (ReLU(gate) * up >= 0 when both >= 0)
            }
        }
    }

    // --- apply_gating (dispatch by type) ---
    {
        let gating_type = match input.gating_type % 3 {
            0 => GatingType::SwiGLU,
            1 => GatingType::GeGLU,
            _ => GatingType::ReGLU,
        };
        let mut out = output.clone();
        if let Ok(()) = apply_gating(gating_type, gate, up, &mut out) {
            assert_eq!(out.len(), n);
        }
    }

    // --- mismatched lengths should error ---
    if n > 1 {
        let short_gate = &gate[..n - 1];
        let mut out = vec![0.0f32; n];
        assert!(swiglu(short_gate, up, &mut out).is_err());
    }
});
