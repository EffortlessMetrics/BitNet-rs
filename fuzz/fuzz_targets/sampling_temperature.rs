#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct SamplingInput {
    logits: Vec<f32>,
    temperature: f32,
    top_k: u16,
}

fuzz_target!(|input: SamplingInput| {
    let logits: Vec<f32> = input.logits.into_iter().take(256).collect();
    if logits.is_empty() {
        return;
    }
    let temp = input.temperature.abs().clamp(0.01, 100.0);
    let top_k = (input.top_k as usize).clamp(1, logits.len());
    // Apply temperature scaling - should never panic
    let _scaled: Vec<f32> = logits.iter().map(|l| l / temp).collect();
    let _ = top_k;
});
