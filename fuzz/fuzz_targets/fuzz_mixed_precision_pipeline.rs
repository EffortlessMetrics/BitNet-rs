#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

/// Fuzz mixed precision conversion chains, verifying that conversions between
/// f32, f16, bf16, and i8 preserve invariants and don't produce invalid values.
#[derive(Arbitrary, Debug)]
struct MixedPrecisionInput {
    data_bytes: Vec<u8>,
    conversion_chain: Vec<ConversionStep>,
    clamp_range_lo: i8,
    clamp_range_hi: i8,
}

#[derive(Arbitrary, Debug, Clone, Copy)]
enum ConversionStep {
    F32ToF16,
    F16ToF32,
    F32ToBf16,
    Bf16ToF32,
    F32ToI8 { scale_byte: u8 },
    I8ToF32 { scale_byte: u8 },
    Clamp,
    RoundToNearest,
    ScaleBy { factor_byte: u8 },
}

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf/NaN → f16 Inf/NaN
        return ((sign << 15) | 0x7C00 | if mantissa != 0 { 0x200 } else { 0 }) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | 0x7C00) as u16; // overflow → Inf
    }
    if new_exp <= 0 {
        return (sign << 15) as u16; // underflow → 0
    }

    let new_mantissa = mantissa >> 13;
    ((sign << 15) | ((new_exp as u32) << 10) | new_mantissa) as u16
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0x1F {
        let f32_bits = (sign << 31) | 0x7F800000 | if mantissa != 0 { 0x400000 } else { 0 };
        return f32::from_bits(f32_bits);
    }
    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let shift = mantissa.leading_zeros() - 22;
        let new_exp = 127 - 15 - shift + 1;
        let new_mantissa = (mantissa << shift) & 0x3FF;
        let f32_bits = (sign << 31) | (new_exp << 23) | (new_mantissa << 13);
        return f32::from_bits(f32_bits);
    }

    let new_exp = exp + 127 - 15;
    let f32_bits = (sign << 31) | (new_exp << 23) | (mantissa << 13);
    f32::from_bits(f32_bits)
}

fn f32_to_bf16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    // Simple truncation (round-to-nearest-even not needed for fuzzing)
    (bits >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn apply_chain(data: &mut Vec<f32>, chain: &[ConversionStep], clamp_lo: f32, clamp_hi: f32) {
    for step in chain {
        match step {
            ConversionStep::F32ToF16 => {
                for v in data.iter_mut() {
                    let bits = f32_to_f16_bits(*v);
                    *v = f16_bits_to_f32(bits);
                }
            }
            ConversionStep::F16ToF32 => {
                // Already f32 representation; this is a no-op identity.
            }
            ConversionStep::F32ToBf16 => {
                for v in data.iter_mut() {
                    let bits = f32_to_bf16_bits(*v);
                    *v = bf16_bits_to_f32(bits);
                }
            }
            ConversionStep::Bf16ToF32 => {
                // Already f32 representation; identity.
            }
            ConversionStep::F32ToI8 { scale_byte } => {
                let scale = (*scale_byte as f32 / 255.0) * 127.0 + 1.0;
                for v in data.iter_mut() {
                    let quantized = (*v * scale).round().clamp(-128.0, 127.0) as i8;
                    *v = quantized as f32 / scale;
                }
            }
            ConversionStep::I8ToF32 { scale_byte } => {
                let scale = (*scale_byte as f32 / 255.0) * 127.0 + 1.0;
                for v in data.iter_mut() {
                    let quantized = (*v * scale).round().clamp(-128.0, 127.0) as i8;
                    *v = quantized as f32 / scale;
                }
            }
            ConversionStep::Clamp => {
                for v in data.iter_mut() {
                    *v = v.clamp(clamp_lo, clamp_hi);
                }
            }
            ConversionStep::RoundToNearest => {
                for v in data.iter_mut() {
                    *v = v.round();
                }
            }
            ConversionStep::ScaleBy { factor_byte } => {
                let factor = (*factor_byte as f32 / 128.0) - 1.0; // range [-1, ~1)
                for v in data.iter_mut() {
                    *v *= factor;
                }
            }
        }
    }
}

fuzz_target!(|input: MixedPrecisionInput| {
    if input.data_bytes.len() < 4 {
        return;
    }

    let aligned = (input.data_bytes.len() / 4) * 4;
    let mut data: Vec<f32> = input.data_bytes[..aligned]
        .chunks_exact(4)
        .take(128)
        .map(|b| {
            let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            if v.is_finite() { v.clamp(-1e4, 1e4) } else { 0.0 }
        })
        .collect();

    if data.is_empty() {
        return;
    }

    let clamp_lo = (input.clamp_range_lo as f32).min(input.clamp_range_hi as f32);
    let clamp_hi = (input.clamp_range_lo as f32).max(input.clamp_range_hi as f32);
    let original_len = data.len();

    // Apply conversion chain (limited to 32 steps).
    let chain: Vec<ConversionStep> = input.conversion_chain.iter().copied().take(32).collect();
    apply_chain(&mut data, &chain, clamp_lo, clamp_hi);

    // Invariant 1: Length is preserved through all conversions.
    assert_eq!(data.len(), original_len, "conversion chain changed length");

    // Invariant 2: All outputs are finite (no NaN/Inf introduced).
    for (i, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "non-finite value at index {i}: {v}");
    }

    // Invariant 3: f32→f16→f32 round-trip doesn't increase magnitude beyond precision.
    let mut test_vals = data.clone();
    apply_chain(&mut test_vals, &[ConversionStep::F32ToF16], clamp_lo, clamp_hi);
    for (i, (&orig, &converted)) in data.iter().zip(test_vals.iter()).enumerate() {
        if orig.is_finite() && converted.is_finite() {
            // f16 has ~3 decimal digits of precision
            let max_err = orig.abs() * 1e-3 + 1e-4;
            assert!(
                (orig - converted).abs() <= max_err,
                "f16 round-trip error at {i}: {orig} vs {converted}"
            );
        }
    }

    // Invariant 4: bf16 round-trip preserves approximate value.
    let mut bf_vals = data.clone();
    apply_chain(&mut bf_vals, &[ConversionStep::F32ToBf16], clamp_lo, clamp_hi);
    for (i, (&orig, &converted)) in data.iter().zip(bf_vals.iter()).enumerate() {
        if orig.is_finite() && converted.is_finite() {
            // bf16 has ~2-3 decimal digits of precision
            let max_err = orig.abs() * 1e-2 + 1e-3;
            assert!(
                (orig - converted).abs() <= max_err,
                "bf16 round-trip error at {i}: {orig} vs {converted}"
            );
        }
    }

    // Invariant 5: Double conversion is idempotent (f16→f16 == f16).
    let mut double_f16 = data.clone();
    apply_chain(
        &mut double_f16,
        &[ConversionStep::F32ToF16, ConversionStep::F32ToF16],
        clamp_lo,
        clamp_hi,
    );
    let mut single_f16 = data.clone();
    apply_chain(&mut single_f16, &[ConversionStep::F32ToF16], clamp_lo, clamp_hi);
    for (i, (&d, &s)) in double_f16.iter().zip(single_f16.iter()).enumerate() {
        assert_eq!(d.to_bits(), s.to_bits(), "f16 not idempotent at {i}: {d} vs {s}");
    }

    // Invariant 6: Clamp respects bounds.
    let mut clamped = data.clone();
    apply_chain(&mut clamped, &[ConversionStep::Clamp], clamp_lo, clamp_hi);
    for (i, &v) in clamped.iter().enumerate() {
        assert!(
            v >= clamp_lo && v <= clamp_hi,
            "clamp violated at {i}: {v} not in [{clamp_lo}, {clamp_hi}]"
        );
    }
});
