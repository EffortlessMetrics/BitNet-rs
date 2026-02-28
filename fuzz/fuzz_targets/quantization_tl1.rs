#![no_main]

use arbitrary::Arbitrary;
use bitnet_quantization::tl1::LookupTable;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Tl1Input {
    min_val: f32,
    max_val: f32,
    bits: u8,
    use_asymmetric: bool,
    /// Values to quantize and round-trip.
    values: Vec<f32>,
}

fuzz_target!(|input: Tl1Input| {
    // Skip degenerate floats that would make table construction meaningless.
    if !input.min_val.is_finite() || !input.max_val.is_finite() {
        return;
    }
    if input.min_val >= input.max_val {
        return;
    }

    // Clamp bits to a reasonable range (1..=8); the constructor must not panic.
    let bits = input.bits.clamp(1, 8);

    // LookupTable::new must never panic.
    let table = LookupTable::new(input.min_val, input.max_val, bits, input.use_asymmetric);

    // Quantize + dequantize each value — must never panic.
    for &v in input.values.iter().take(256) {
        let q = table.quantize(v);
        let dq = table.dequantize(q);

        // Dequantized value must be finite when input is finite.
        if v.is_finite() {
            assert!(
                dq.is_finite(),
                "dequantize produced non-finite {dq} from quantized {q} (input {v})",
            );
        }
    }
});
