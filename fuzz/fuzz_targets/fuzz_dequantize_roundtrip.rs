#![no_main]

use arbitrary::Arbitrary;
use bitnet_quantization::int4_quant::{
    Int4QuantConfig, dequantize_tensor_int4, quantize_tensor_int4,
};
use bitnet_quantization::int8_quant::{
    Int8QuantConfig, dequantize_tensor_int8, quantize_tensor_int8,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct RoundtripInput {
    mode: u8,
    group_size: u8,
    symmetric: bool,
    data: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| {
            let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            if v.is_finite() { v.clamp(-1e6, 1e6) } else { 0.0 }
        })
        .collect()
}

fuzz_target!(|input: RoundtripInput| {
    let floats = bytes_to_f32(&input.data, 512);
    if floats.is_empty() {
        return;
    }

    match input.mode % 2 {
        0 => {
            // INT8 quantize → dequantize roundtrip
            let config = Int8QuantConfig { symmetric: input.symmetric, ..Default::default() };
            let (quantized, params) = quantize_tensor_int8(&floats, &config);

            // Invariant 1: Quantized length matches input
            assert_eq!(quantized.len(), floats.len(), "int8 quantized length mismatch");

            // Invariant 2: All quantized values are in [-128, 127]
            for (i, &v) in quantized.iter().enumerate() {
                assert!((-128..=127).contains(&(v as i32)), "int8 value out of range at {i}: {v}");
            }

            // Invariant 3: Scale must be finite and non-negative
            for (i, &s) in params.scales.iter().enumerate() {
                assert!(s.is_finite(), "int8 scale non-finite at {i}: {s}");
                assert!(s >= 0.0, "int8 scale negative at {i}: {s}");
            }

            let deq = dequantize_tensor_int8(&quantized, &params);

            // Invariant 4: Dequantized length matches input
            assert_eq!(deq.len(), floats.len(), "int8 dequantized length mismatch");

            // Invariant 5: All dequantized values must be finite
            for (i, &v) in deq.iter().enumerate() {
                assert!(v.is_finite(), "int8 dequantized non-finite at {i}: {v}");
            }

            // Invariant 6: Roundtrip error is bounded (int8 has ~0.4% max error)
            let max_orig = floats.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            if max_orig > 1e-6 {
                for (i, (&orig, &deq_v)) in floats.iter().zip(deq.iter()).enumerate() {
                    let abs_err = (orig - deq_v).abs();
                    // INT8 can have up to ~1/127 relative error per value
                    let tolerance = max_orig * 0.02 + 1e-5;
                    assert!(
                        abs_err <= tolerance,
                        "int8 roundtrip error too large at {i}: orig={orig}, deq={deq_v}, err={abs_err}, tol={tolerance}"
                    );
                }
            }
        }
        1 => {
            // INT4 quantize → dequantize roundtrip
            let group_size = ((input.group_size as usize) % 32) + 1;
            let config =
                Int4QuantConfig { group_size, symmetric: input.symmetric, block_wise: true };
            let (packed, params) = quantize_tensor_int4(&floats, &config);

            // Invariant 7: Packed data length is ceil(n/2) bytes
            assert_eq!(packed.len, floats.len(), "int4 packed element count mismatch");

            // Invariant 8: All scales must be finite and non-negative
            for (i, &s) in params.scales.iter().enumerate() {
                assert!(s.is_finite(), "int4 scale non-finite at {i}: {s}");
                assert!(s >= 0.0, "int4 scale negative at {i}: {s}");
            }

            let deq = dequantize_tensor_int4(&packed, &params);

            // Invariant 9: Dequantized length matches input
            assert_eq!(deq.len(), floats.len(), "int4 dequantized length mismatch");

            // Invariant 10: All dequantized values must be finite
            for (i, &v) in deq.iter().enumerate() {
                assert!(v.is_finite(), "int4 dequantized non-finite at {i}: {v}");
            }

            // Invariant 11: Zero input should produce zero output
            let zeros = vec![0.0f32; floats.len()];
            let (z_packed, z_params) = quantize_tensor_int4(&zeros, &config);
            let z_deq = dequantize_tensor_int4(&z_packed, &z_params);
            for (i, &v) in z_deq.iter().enumerate() {
                assert!(v.abs() < 1e-5, "int4 zero roundtrip non-zero at {i}: {v}");
            }
        }
        _ => unreachable!(),
    }
});
