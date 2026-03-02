#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::QuantizationType;
use bitnet_kernels::KernelManager;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct DispatchInput {
    /// Input values for quantize dispatch.
    values: Vec<f32>,
    /// Matrix dimensions for matmul dispatch (clamped in body).
    m: u8,
    n: u8,
    k: u8,
    /// Signed activations for matmul_i2s.
    activations: Vec<i8>,
    /// Packed weight bytes for matmul_i2s.
    weights: Vec<u8>,
    /// Quantization type selector.
    qtype_selector: u8,
}

const MAX_DIM: usize = 32;

fuzz_target!(|input: DispatchInput| {
    let mgr = KernelManager::new();

    // Invariant: at least one provider is always available.
    let providers = mgr.list_available_providers();
    assert!(!providers.is_empty(), "must have at least one kernel provider");

    let provider = match mgr.select_best() {
        Ok(p) => p,
        Err(_) => return,
    };

    // Provider name must be non-empty.
    assert!(!provider.name().is_empty());

    // ── matmul_i2s dispatch with adversarial dims ──────────────
    let m = (input.m as usize % MAX_DIM) + 1;
    let n = (input.n as usize % MAX_DIM) + 1;
    let k = (input.k as usize % MAX_DIM) + 1;

    let a_len = m * k;
    let b_len = k * n;
    let c_len = m * n;

    let a: Vec<i8> = input.activations.iter().copied().cycle().take(a_len).collect();
    let b: Vec<u8> = input.weights.iter().copied().cycle().take(b_len).collect();
    let mut c = vec![0.0f32; c_len];

    // Must not panic; errors are acceptable.
    let _ = provider.matmul_i2s(&a, &b, &mut c, m, n, k);

    // ── quantize dispatch ─────────────────────────────────────
    let values: Vec<f32> = input
        .values
        .iter()
        .take(256)
        .map(|&x| if x.is_nan() || x.is_infinite() { 0.0 } else { x.clamp(-1e6, 1e6) })
        .collect();

    if !values.is_empty() {
        let qtype = match input.qtype_selector % 3 {
            0 => QuantizationType::I2S,
            1 => QuantizationType::TL1,
            _ => QuantizationType::TL2,
        };

        let mut output = vec![0u8; values.len() * 4];
        let mut scales = vec![0.0f32; (values.len() / 32).max(1)];

        // Must not panic; errors are fine.
        let _ = provider.quantize(&values, &mut output, &mut scales, qtype);
    }
});
