#![no_main]

use arbitrary::Arbitrary;
use bitnet_quantization::tl2::VectorizedLookupTable;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Tl2Input {
    min_val: f32,
    max_val: f32,
    bits: u8,
    /// Values to quantize and round-trip.
    values: Vec<f32>,
}

fuzz_target!(|input: Tl2Input| {
    if !input.min_val.is_finite() || !input.max_val.is_finite() {
        return;
    }
    if input.min_val >= input.max_val {
        return;
    }

    let bits = input.bits.clamp(1, 8);

    // VectorizedLookupTable::new must never panic.
    let table = VectorizedLookupTable::new(input.min_val, input.max_val, bits);

    // Table length accessors must not panic.
    let _ = table.forward_len();
    let _ = table.reverse_len();

    // Quantize + dequantize each value — must never panic.
    for &v in input.values.iter().take(256) {
        let q = table.quantize(v);
        let dq = table.dequantize(q);

        if v.is_finite() {
            assert!(
                dq.is_finite(),
                "dequantize produced non-finite {dq} from quantized {q} (input {v})",
            );
        }
    }
});
