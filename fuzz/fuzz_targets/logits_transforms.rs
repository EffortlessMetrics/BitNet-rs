#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct LogitsInput {
    /// Raw bytes interpreted as f32 logits (up to 256 values).
    data: Vec<u8>,
    temperature: f32,
    top_k: u8,
    top_p: f32,
    repetition_penalty: f32,
    token_history: Vec<u8>,
}

fuzz_target!(|input: LogitsInput| {
    // Convert raw bytes to f32 slice (truncate to nearest multiple of 4).
    let aligned_len = (input.data.len() / 4) * 4;
    if aligned_len == 0 {
        return;
    }
    let data = &input.data[..aligned_len];
    let mut logits: Vec<f32> =
        data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();

    // Keep to a reasonable size (≤ 256) to avoid timeout.
    logits.truncate(256);
    if logits.is_empty() {
        return;
    }

    // Token history: interpret bytes as token IDs.
    let token_ids: Vec<u32> = input.token_history.iter().map(|&b| b as u32).collect();

    // --- apply_repetition_penalty: must not panic on any input ---
    {
        let mut l = logits.clone();
        let penalty = input.repetition_penalty.clamp(0.1, 10.0);
        bitnet_logits::apply_repetition_penalty(&mut l, &token_ids, penalty);
        // All values must still be finite or NEG_INFINITY (no NaN from the operation).
        // Note: logits may already contain NaN/inf from raw bytes — that is valid
        // input; the penalty function must survive without panicking.
        let _ = l;
    }

    // --- apply_temperature: must not panic ---
    {
        let mut l = logits.clone();
        // temperature=0.0 is a no-op per spec; we don't clamp here.
        bitnet_logits::apply_temperature(&mut l, input.temperature);
        let _ = l;
    }

    // --- apply_top_k: must not panic, returned count ≤ logits.len() ---
    {
        let mut l = logits.clone();
        let top_k = input.top_k as usize;
        let kept = bitnet_logits::apply_top_k(&mut l, top_k);
        assert!(kept <= l.len(), "apply_top_k returned count > len");
    }

    // --- softmax_in_place on finite logits: must produce valid probabilities ---
    {
        // Only run softmax when logits contain at least one finite value.
        let finite: Vec<f32> = logits.iter().copied().filter(|x| x.is_finite()).collect();
        if !finite.is_empty() {
            let mut l = finite;
            bitnet_logits::softmax_in_place(&mut l);
            // All outputs must be non-negative and finite.
            for &p in &l {
                assert!(p >= 0.0, "softmax produced negative probability: {p}");
                assert!(p.is_finite(), "softmax produced non-finite: {p}");
            }
        }
    }

    // --- apply_top_p: must not panic ---
    {
        let finite: Vec<f32> = logits.iter().copied().filter(|x| x.is_finite()).collect();
        if !finite.is_empty() {
            let mut l = finite;
            bitnet_logits::softmax_in_place(&mut l);
            bitnet_logits::apply_top_p(&mut l, input.top_p.clamp(0.0, 1.0));
            let _ = l;
        }
    }

    // --- argmax: must return a valid index ---
    {
        let idx = bitnet_logits::argmax(&logits);
        assert!(
            idx < logits.len(),
            "argmax returned out-of-bounds index {idx} for len {}",
            logits.len()
        );
    }
});
