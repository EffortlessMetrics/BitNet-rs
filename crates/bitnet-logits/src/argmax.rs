use std::cmp::Ordering;

/// Return the index of the maximum value (argmax).
pub fn argmax(logits: &[f32]) -> usize {
    logits.iter().enumerate().max_by(|(_, a), (_, b)| f32_ascending(**a, **b)).map_or(0, |(i, _)| i)
}

#[inline]
fn f32_ascending(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}
