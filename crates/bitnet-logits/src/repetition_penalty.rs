/// Apply a multiplicative repetition penalty to previously-seen tokens.
pub fn apply_repetition_penalty(logits: &mut [f32], token_ids: &[u32], penalty: f32) {
    #[allow(clippy::float_cmp)]
    if penalty <= 0.0 || !penalty.is_finite() || penalty == 1.0 || token_ids.is_empty() {
        return;
    }
    let inv_penalty = 1.0 / penalty;
    for &id in token_ids {
        let idx = id as usize;
        if let Some(logit) = logits.get_mut(idx) {
            if *logit > 0.0 {
                *logit *= inv_penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
}
