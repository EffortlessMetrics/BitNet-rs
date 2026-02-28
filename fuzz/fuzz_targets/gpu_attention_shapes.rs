#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct AttentionShapeInput {
    batch: u8,
    seq_len: u8,
    num_heads: u8,
    head_dim: u8,
}

fuzz_target!(|input: AttentionShapeInput| {
    use bitnet_common::{BitNetTensor, Device, Tensor};

    // Clamp dimensions to small values to prevent timeout.
    let batch = (input.batch as usize).clamp(1, 4);
    let seq_len = (input.seq_len as usize).clamp(1, 32);
    let num_heads = (input.num_heads as usize).clamp(1, 8);
    let head_dim = (input.head_dim as usize).clamp(1, 16);

    // Guard against overflow.
    let Some(total) = batch
        .checked_mul(seq_len)
        .and_then(|v| v.checked_mul(num_heads))
        .and_then(|v| v.checked_mul(head_dim))
    else {
        return;
    };
    if total == 0 || total > 256 * 256 {
        return;
    }

    let shape = vec![batch, seq_len, num_heads, head_dim];

    // Construct Q, K, V tensors with zeros — must not panic.
    let data = vec![0.0f32; total];
    let q = match BitNetTensor::from_slice(&data, &shape, &Device::Cpu) {
        Ok(t) => t,
        Err(_) => return,
    };
    let k = match BitNetTensor::from_slice(&data, &shape, &Device::Cpu) {
        Ok(t) => t,
        Err(_) => return,
    };
    let v = match BitNetTensor::from_slice(&data, &shape, &Device::Cpu) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Shape invariants.
    assert_eq!(q.shape(), &shape, "Q shape mismatch");
    assert_eq!(k.shape(), &shape, "K shape mismatch");
    assert_eq!(v.shape(), &shape, "V shape mismatch");

    // Verify shapes are consistent for attention: Q, K, V must have same dims.
    assert_eq!(q.shape(), k.shape(), "Q/K shape mismatch");
    assert_eq!(k.shape(), v.shape(), "K/V shape mismatch");

    // head_dim must divide hidden_size (num_heads * head_dim).
    let hidden_size = num_heads * head_dim;
    assert_eq!(
        hidden_size % head_dim,
        0,
        "hidden_size {hidden_size} not divisible by head_dim {head_dim}"
    );
    assert_eq!(
        hidden_size / head_dim,
        num_heads,
        "num_heads mismatch: {} != {num_heads}",
        hidden_size / head_dim
    );

    // Verify from_slice rejects mismatched shape/data.
    let bad_shape = vec![batch, seq_len, num_heads, head_dim + 1];
    assert!(
        BitNetTensor::from_slice(&data, &bad_shape, &Device::Cpu).is_err(),
        "from_slice should reject mismatched shape/data"
    );
});
