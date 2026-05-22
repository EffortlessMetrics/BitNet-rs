/// Scale logits by `1 / temperature`.
///
/// * `temperature == 0.0` → no-op (handled externally via greedy path).
/// * `temperature == 1.0` → no-op (identity scaling).
/// * Values in `(0, 1)` sharpen the distribution (lower entropy).
/// * Values `> 1` flatten it (higher entropy / more randomness).
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if logits.is_empty() {
        return;
    }
    #[allow(clippy::float_cmp)]
    if temperature == 0.0 || temperature == 1.0 {
        return;
    }
    let inv = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv;
    }
}
