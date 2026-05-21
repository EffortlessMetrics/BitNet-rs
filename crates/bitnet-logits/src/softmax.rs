/// Convert raw logits to a probability distribution in-place via softmax.
pub fn softmax_in_place(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::INFINITY {
        let positive_infinity_count =
            logits.iter().filter(|&&value| value == f32::INFINITY).count();
        let probability = 1.0 / positive_infinity_count as f32;
        for l in logits.iter_mut() {
            *l = if *l == f32::INFINITY { probability } else { 0.0 };
        }
        return;
    }

    let mut sum = 0.0f64;
    for l in logits.iter_mut() {
        let v = *l;
        if v == f32::NEG_INFINITY {
            *l = 0.0;
        } else {
            let exp = f64::from(v - max).exp();
            *l = exp as f32;
            sum += exp;
        }
    }
    if sum > 0.0 && sum.is_finite() {
        let inv_sum = (1.0 / sum) as f32;
        for l in logits.iter_mut() {
            *l *= inv_sum;
        }
    } else {
        #[allow(clippy::cast_precision_loss)]
        let uniform = 1.0_f32 / logits.len() as f32;
        for l in logits.iter_mut() {
            *l = uniform;
        }
    }
}
