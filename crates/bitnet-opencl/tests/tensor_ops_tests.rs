use bitnet_opencl::{
    Backend, GpuTensorOps, Tensor, TensorOpsDispatcher, TensorShape, tensor_ops_cpu,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn t1d(data: Vec<f32>) -> Tensor {
    let n = data.len();
    Tensor::new(TensorShape::new(&[n]), data).unwrap()
}

fn t2d(rows: usize, cols: usize, data: Vec<f32>) -> Tensor {
    Tensor::new(TensorShape::new(&[rows, cols]), data).unwrap()
}

fn approx_eq(a: &[f32], b: &[f32], eps: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < eps)
}

fn dispatcher() -> TensorOpsDispatcher {
    TensorOpsDispatcher::cpu()
}

// ===================================================================
// matmul
// ===================================================================

#[test]
fn matmul_2x2_known() {
    let a = t2d(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let b = t2d(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
    let c = dispatcher().matmul(&a, &b).unwrap();
    assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn matmul_identity() {
    let a = t2d(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let eye = t2d(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let c = dispatcher().matmul(&a, &eye).unwrap();
    assert!(approx_eq(&c.data, &a.data, 1e-6));
}

#[test]
fn matmul_1x1() {
    let a = t2d(1, 1, vec![3.0]);
    let b = t2d(1, 1, vec![4.0]);
    let c = dispatcher().matmul(&a, &b).unwrap();
    assert_eq!(c.data, vec![12.0]);
}

#[test]
fn matmul_shape_mismatch() {
    let a = t2d(2, 3, vec![0.0; 6]);
    let b = t2d(2, 2, vec![0.0; 4]);
    assert!(dispatcher().matmul(&a, &b).is_err());
}

// ===================================================================
// add
// ===================================================================

#[test]
fn add_known() {
    let a = t1d(vec![1.0, 2.0, 3.0]);
    let b = t1d(vec![4.0, 5.0, 6.0]);
    let c = dispatcher().add(&a, &b).unwrap();
    assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn add_commutative() {
    let a = t1d(vec![1.0, -2.0, 3.5]);
    let b = t1d(vec![4.0, 5.0, -6.0]);
    let ab = dispatcher().add(&a, &b).unwrap();
    let ba = dispatcher().add(&b, &a).unwrap();
    assert_eq!(ab.data, ba.data);
}

#[test]
fn add_shape_mismatch() {
    let a = t1d(vec![1.0, 2.0]);
    let b = t1d(vec![1.0, 2.0, 3.0]);
    assert!(dispatcher().add(&a, &b).is_err());
}

// ===================================================================
// mul
// ===================================================================

#[test]
fn mul_known() {
    let a = t1d(vec![2.0, 3.0, 4.0]);
    let b = t1d(vec![5.0, 6.0, 7.0]);
    let c = dispatcher().mul(&a, &b).unwrap();
    assert_eq!(c.data, vec![10.0, 18.0, 28.0]);
}

#[test]
fn mul_by_zero() {
    let a = t1d(vec![1.0, 2.0, 3.0]);
    let z = t1d(vec![0.0, 0.0, 0.0]);
    let c = dispatcher().mul(&a, &z).unwrap();
    assert!(c.data.iter().all(|&v| v == 0.0));
}

#[test]
fn mul_shape_mismatch() {
    let a = t1d(vec![1.0]);
    let b = t1d(vec![1.0, 2.0]);
    assert!(dispatcher().mul(&a, &b).is_err());
}

// ===================================================================
// softmax
// ===================================================================

#[test]
fn softmax_sum_to_one() {
    let t = t1d(vec![1.0, 2.0, 3.0]);
    let s = dispatcher().softmax(&t, 0).unwrap();
    let sum: f32 = s.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn softmax_2d_rows() {
    let t = t2d(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let s = dispatcher().softmax(&t, 1).unwrap();
    // Each row sums to 1.
    let row0: f32 = s.data[..3].iter().sum();
    let row1: f32 = s.data[3..].iter().sum();
    assert!((row0 - 1.0).abs() < 1e-5);
    assert!((row1 - 1.0).abs() < 1e-5);
}

#[test]
fn softmax_large_values() {
    let t = t1d(vec![1000.0, 1001.0, 1002.0]);
    let s = dispatcher().softmax(&t, 0).unwrap();
    let sum: f32 = s.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn softmax_negative_values() {
    let t = t1d(vec![-1.0, -2.0, -3.0]);
    let s = dispatcher().softmax(&t, 0).unwrap();
    let sum: f32 = s.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // Largest value gets largest probability.
    assert!(s.data[0] > s.data[1]);
    assert!(s.data[1] > s.data[2]);
}

#[test]
fn softmax_empty_tensor() {
    let t = Tensor::new(TensorShape::new(&[0]), vec![]).unwrap();
    assert!(dispatcher().softmax(&t, 0).is_err());
}

#[test]
fn softmax_invalid_dim() {
    let t = t1d(vec![1.0, 2.0]);
    assert!(dispatcher().softmax(&t, 5).is_err());
}

// ===================================================================
// rmsnorm
// ===================================================================

#[test]
fn rmsnorm_unit_rms() {
    let input = t2d(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
    let weight = t1d(vec![1.0, 1.0, 1.0, 1.0]);
    let out = dispatcher().rmsnorm(&input, &weight, 1e-5).unwrap();
    // After rmsnorm with unit weight, the RMS of output ≈ 1.
    let rms: f32 = (out.data.iter().map(|x| x * x).sum::<f32>() / out.data.len() as f32).sqrt();
    assert!((rms - 1.0).abs() < 1e-4);
}

#[test]
fn rmsnorm_with_weight() {
    let input = t1d(vec![2.0, 2.0]);
    let weight = t1d(vec![3.0, 3.0]);
    let out = dispatcher().rmsnorm(&input, &weight, 1e-5).unwrap();
    // rms = sqrt((4+4)/2) = 2, normalised = [1,1], scaled = [3,3]
    assert!(approx_eq(&out.data, &[3.0, 3.0], 1e-4));
}

#[test]
fn rmsnorm_weight_mismatch() {
    let input = t1d(vec![1.0, 2.0, 3.0]);
    let weight = t1d(vec![1.0, 1.0]);
    assert!(dispatcher().rmsnorm(&input, &weight, 1e-5).is_err());
}

#[test]
fn rmsnorm_empty() {
    let input = Tensor::new(TensorShape::new(&[0]), vec![]).unwrap();
    let weight = Tensor::new(TensorShape::new(&[0]), vec![]).unwrap();
    assert!(dispatcher().rmsnorm(&input, &weight, 1e-5).is_err());
}

// ===================================================================
// rope
// ===================================================================

#[test]
fn rope_zero_freqs() {
    // Zero frequencies → cos=1, sin=0 → output equals input.
    let input = t2d(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
    let freqs = t2d(1, 2, vec![0.0, 0.0]);
    let out = dispatcher().rope(&input, &freqs).unwrap();
    assert!(approx_eq(&out.data, &input.data, 1e-6));
}

#[test]
fn rope_odd_dim_error() {
    let input = t2d(2, 3, vec![0.0; 6]);
    let freqs = t2d(2, 1, vec![0.0; 2]);
    assert!(dispatcher().rope(&input, &freqs).is_err());
}

// ===================================================================
// silu
// ===================================================================

#[test]
fn silu_zero() {
    let t = t1d(vec![0.0, 0.0, 0.0]);
    let s = dispatcher().silu(&t).unwrap();
    assert!(s.data.iter().all(|&v| v.abs() < 1e-9));
}

#[test]
fn silu_known_value() {
    // silu(1.0) = 1.0 * sigmoid(1.0) ≈ 0.7310586
    let t = t1d(vec![1.0]);
    let s = dispatcher().silu(&t).unwrap();
    assert!((s.data[0] - 0.7310586).abs() < 1e-5);
}

#[test]
fn silu_negative() {
    // silu(x) for large negative x → 0.
    let t = t1d(vec![-100.0]);
    let s = dispatcher().silu(&t).unwrap();
    assert!(s.data[0].abs() < 1e-5);
}

// ===================================================================
// attention
// ===================================================================

#[test]
fn attention_identity_pattern() {
    // Q == K → uniform attention → output ≈ mean of V rows.
    let q = t2d(2, 2, vec![1.0, 0.0, 1.0, 0.0]);
    let k = t2d(2, 2, vec![1.0, 0.0, 1.0, 0.0]);
    let v = t2d(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let out = dispatcher().attention(&q, &k, &v, None).unwrap();
    // Equal attention weights → each output row ≈ [2.0, 3.0].
    assert!(approx_eq(&out.data[..2], &[2.0, 3.0], 1e-4));
}

#[test]
fn attention_causal_mask() {
    let q = t2d(2, 2, vec![1.0, 0.0, 1.0, 0.0]);
    let k = t2d(2, 2, vec![1.0, 0.0, 1.0, 0.0]);
    let v = t2d(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    // Causal mask: position 0 can only see position 0.
    let mask = t2d(2, 2, vec![0.0, -1e9, 0.0, 0.0]);
    let out = dispatcher().attention(&q, &k, &v, Some(&mask)).unwrap();
    // Row 0 should attend only to V[0].
    assert!(approx_eq(&out.data[..2], &[1.0, 2.0], 1e-3));
}

#[test]
fn attention_qk_mismatch() {
    let q = t2d(2, 3, vec![0.0; 6]);
    let k = t2d(2, 2, vec![0.0; 4]);
    let v = t2d(2, 2, vec![0.0; 4]);
    assert!(dispatcher().attention(&q, &k, &v, None).is_err());
}

#[test]
fn attention_mask_shape_error() {
    let q = t2d(2, 2, vec![0.0; 4]);
    let k = t2d(2, 2, vec![0.0; 4]);
    let v = t2d(2, 2, vec![0.0; 4]);
    let bad_mask = t2d(3, 3, vec![0.0; 9]);
    assert!(dispatcher().attention(&q, &k, &v, Some(&bad_mask)).is_err());
}

// ===================================================================
// embedding
// ===================================================================

#[test]
fn embedding_known() {
    let table = t2d(3, 2, vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
    let out = dispatcher().embedding(&[2, 0], &table).unwrap();
    assert_eq!(out.data, vec![30.0, 31.0, 10.0, 11.0]);
}

#[test]
fn embedding_out_of_range() {
    let table = t2d(2, 2, vec![0.0; 4]);
    assert!(dispatcher().embedding(&[5], &table).is_err());
}

#[test]
fn embedding_single() {
    let table = t2d(4, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let out = dispatcher().embedding(&[1], &table).unwrap();
    assert_eq!(out.data, vec![4.0, 5.0, 6.0]);
}

// ===================================================================
// linear
// ===================================================================

#[test]
fn linear_no_bias() {
    let input = t2d(1, 2, vec![1.0, 2.0]);
    let weight = t2d(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let out = dispatcher().linear(&input, &weight, None).unwrap();
    assert_eq!(out.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn linear_with_bias() {
    let input = t2d(1, 2, vec![1.0, 2.0]);
    let weight = t2d(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let bias = t1d(vec![10.0, 20.0]);
    let out = dispatcher().linear(&input, &weight, Some(&bias)).unwrap();
    assert_eq!(out.data, vec![11.0, 22.0]);
}

#[test]
fn linear_batch() {
    let input = t2d(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let weight = t2d(2, 2, vec![3.0, 4.0, 5.0, 6.0]);
    let out = dispatcher().linear(&input, &weight, None).unwrap();
    assert_eq!(out.data, vec![3.0, 5.0, 4.0, 6.0]);
}

#[test]
fn linear_weight_mismatch() {
    let input = t2d(1, 3, vec![1.0, 2.0, 3.0]);
    let weight = t2d(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    assert!(dispatcher().linear(&input, &weight, None).is_err());
}

#[test]
fn linear_bias_mismatch() {
    let input = t2d(1, 2, vec![1.0, 2.0]);
    let weight = t2d(3, 2, vec![0.0; 6]);
    let bad_bias = t1d(vec![1.0, 2.0]);
    assert!(dispatcher().linear(&input, &weight, Some(&bad_bias)).is_err());
}

// ===================================================================
// dispatcher
// ===================================================================

#[test]
fn dispatcher_auto_is_cpu() {
    // GPU is not available → auto should pick CPU.
    let d = TensorOpsDispatcher::auto();
    assert_eq!(d.backend(), Backend::Cpu);
}

// ===================================================================
// tensor construction
// ===================================================================

#[test]
fn tensor_data_length_mismatch() {
    let r = Tensor::new(TensorShape::new(&[2, 3]), vec![1.0; 7]);
    assert!(r.is_err());
}

#[test]
fn tensor_zeros() {
    let t = Tensor::zeros(TensorShape::new(&[2, 3]));
    assert_eq!(t.numel(), 6);
    assert!(t.data.iter().all(|&v| v == 0.0));
}

// ===================================================================
// CPU reference direct calls (bypassing dispatcher)
// ===================================================================

#[test]
fn cpu_matmul_rect() {
    let a = t2d(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = t2d(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = tensor_ops_cpu::matmul(&a, &b).unwrap();
    assert_eq!(c.shape.dims(), &[2, 2]);
    assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn cpu_softmax_single() {
    let t = t1d(vec![0.0]);
    let s = tensor_ops_cpu::softmax(&t, 0).unwrap();
    assert!((s.data[0] - 1.0).abs() < 1e-6);
}

#[test]
fn cpu_silu_positive() {
    let t = t1d(vec![10.0]);
    let s = tensor_ops_cpu::silu(&t).unwrap();
    // sigmoid(10) ≈ 1 → silu(10) ≈ 10
    assert!((s.data[0] - 10.0).abs() < 0.01);
}
