#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    dim: u8,
    seq_len: u8,
    base: f32,
    /// Optional base for `resolve_base` – exercises both Some and None paths.
    resolve_base_arg: Option<f32>,
}

fuzz_target!(|input: Input| {
    use bitnet_rope::{build_tables, resolve_base};

    // Exercise resolve_base with both None and Some variants.
    let _resolved_none = resolve_base(None);
    let _resolved_some = resolve_base(input.resolve_base_arg);

    let dim = input.dim as usize;
    let seq_len = input.seq_len as usize;

    match build_tables(dim, seq_len, input.base) {
        Ok(tables) => {
            // Shape invariant: both vecs must have exactly seq_len * half_dim entries.
            assert_eq!(tables.sin.len(), seq_len * tables.half_dim);
            assert_eq!(tables.cos.len(), seq_len * tables.half_dim);

            // Trig identity: sin² + cos² ≈ 1 for every pair.
            for (s, c) in tables.sin.iter().zip(&tables.cos) {
                let norm = s * s + c * c;
                assert!((norm - 1.0).abs() < 1e-5, "sin²+cos²={norm} != 1.0 (sin={s}, cos={c})");
            }
        }
        // Errors are expected for invalid inputs; just ensure no panic occurred.
        Err(_) => {}
    }
});
