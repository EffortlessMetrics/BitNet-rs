#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct RopeInput {
    dim: u8,
    seq_len: u8,
    base: f32,
}

fuzz_target!(|input: RopeInput| {
    use bitnet_rope::{build_tables, resolve_base};

    let dim = input.dim as usize;
    let seq_len = input.seq_len as usize;

    // Exercise resolve_base with valid and edge-case values.
    let resolved = resolve_base(Some(input.base));
    assert!(resolved.is_finite(), "resolve_base returned non-finite: {resolved}");

    let default = resolve_base(None);
    assert!(default > 0.0, "default base must be positive");

    match build_tables(dim, seq_len, input.base) {
        Ok(tables) => {
            // Shape invariant: both vecs have seq_len * half_dim entries.
            assert_eq!(tables.sin.len(), seq_len * tables.half_dim);
            assert_eq!(tables.cos.len(), seq_len * tables.half_dim);

            for (i, (s, c)) in tables.sin.iter().zip(&tables.cos).enumerate().take(256) {
                // cos/sin values must be in [-1, 1].
                assert!(*s >= -1.0 && *s <= 1.0, "sin[{i}] = {s} out of [-1, 1]");
                assert!(*c >= -1.0 && *c <= 1.0, "cos[{i}] = {c} out of [-1, 1]");

                // Trig identity: sin² + cos² ≈ 1.
                let norm = s * s + c * c;
                assert!(
                    (norm - 1.0).abs() < 1e-5,
                    "sin²+cos² = {norm} != 1.0 at index {i} (sin={s}, cos={c})"
                );
            }
        }
        // Invalid configs (odd dim, zero dim, non-finite base) return errors — no panic.
        Err(_) => {}
    }
});
