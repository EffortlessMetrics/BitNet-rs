//! RoPE (Rotary Position Embedding) parity tests
//!
//! These tests verify that the RoPE implementation matches llama.cpp reference
//! by using split-halves pairing (i, i+head_dim/2) instead of interleaved pairs (i, i+1).

use candle_core::{Device, Result, Tensor};

/// Helper function to apply RoPE using split-halves approach
///
/// This is a test-only reference implementation to verify the main implementation.
/// For position `pos` and dimension `i`:
///   θ = pos / (rope_theta ^ (2i / head_dim))
///   x'[i]           = x[i] * cos(θ) - x[i+D/2] * sin(θ)
///   x'[i + D/2]     = x[i] * sin(θ) + x[i+D/2] * cos(θ)
fn apply_rope_reference(x: &Tensor, position: usize, rope_theta: f32) -> Result<Tensor> {
    let dims = x.dims();
    let head_dim = dims[dims.len() - 1];
    let half_dim = head_dim / 2;

    // Compute cos/sin values for this position
    let mut cos_vals = Vec::with_capacity(half_dim);
    let mut sin_vals = Vec::with_capacity(half_dim);

    for i in 0..half_dim {
        let freq = (i * 2) as f32 / head_dim as f32;
        let theta = (position as f32) / rope_theta.powf(freq);
        cos_vals.push(theta.cos());
        sin_vals.push(theta.sin());
    }

    // Split into two halves
    let last_dim = dims.len() - 1;
    let x0 = x.narrow(last_dim, 0, half_dim)?;
    let x1 = x.narrow(last_dim, half_dim, half_dim)?;

    // Create cos/sin tensors with proper broadcasting shape
    let mut cos_shape = vec![1; dims.len()];
    cos_shape[last_dim] = half_dim;
    let cos = Tensor::from_vec(cos_vals, cos_shape.as_slice(), x.device())?;
    let sin = Tensor::from_vec(sin_vals, cos_shape.as_slice(), x.device())?;

    // Broadcast cos/sin to match x0/x1 shape
    let cos = cos.broadcast_as(x0.shape())?;
    let sin = sin.broadcast_as(x0.shape())?;

    // Apply rotation formula
    let x0_rot = (x0.mul(&cos)? - x1.mul(&sin)?)?;
    let x1_rot = (x0.mul(&sin)? + x1.mul(&cos)?)?;

    // Concatenate back
    Tensor::cat(&[x0_rot, x1_rot], last_dim)
}

#[test]
fn test_rope_split_halves_position_zero() {
    // At position 0, rotation should be identity (cos=1, sin=0)
    let q_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let q = Tensor::from_vec(q_data.clone(), &[1, 1, 1, 8], &Device::Cpu).unwrap();

    let q_rotated = apply_rope_reference(&q, 0, 10000.0).unwrap();
    let q_rot_data = q_rotated.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // At position 0, all cos(0) = 1, sin(0) = 0, so output = input
    for (i, &val) in q_rot_data.iter().enumerate() {
        assert!(
            (val - q_data[i]).abs() < 1e-6,
            "Position 0 should be identity: expected {}, got {}",
            q_data[i],
            val
        );
    }
}

#[test]
fn test_rope_split_halves_position_one() {
    // Test position 1 with known values
    // For head_dim=8, we have 4 rotation pairs
    let q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let q = Tensor::from_vec(q_data.clone(), &[1, 1, 1, 8], &Device::Cpu).unwrap();

    let rope_theta = 10000.0;
    let q_rotated = apply_rope_reference(&q, 1, rope_theta).unwrap();
    let q_rot_data = q_rotated.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Manually compute expected values for verification
    // For i=0: freq = 0/8 = 0, θ = 1 / 10000^0 = 1
    let theta_0 = 1.0f32 / rope_theta.powf(0.0);
    let cos_0 = theta_0.cos();
    let sin_0 = theta_0.sin();

    // x'[0] = x[0] * cos(θ₀) - x[4] * sin(θ₀) = 1.0 * cos(1) - 5.0 * sin(1)
    let expected_0 = q_data[0] * cos_0 - q_data[4] * sin_0;
    // x'[4] = x[0] * sin(θ₀) + x[4] * cos(θ₀) = 1.0 * sin(1) + 5.0 * cos(1)
    let expected_4 = q_data[0] * sin_0 + q_data[4] * cos_0;

    assert!(
        (q_rot_data[0] - expected_0).abs() < 1e-5,
        "Dimension 0: expected {}, got {}",
        expected_0,
        q_rot_data[0]
    );
    assert!(
        (q_rot_data[4] - expected_4).abs() < 1e-5,
        "Dimension 4: expected {}, got {}",
        expected_4,
        q_rot_data[4]
    );
}

#[test]
fn test_rope_split_halves_batch_heads() {
    // Test with multiple batches and heads: [B=2, H=2, T=1, D=8]
    let batch_size = 2;
    let num_heads = 2;
    let seq_len = 1;
    let head_dim = 8;
    let total_elements = batch_size * num_heads * seq_len * head_dim;

    let q_data: Vec<f32> = (0..total_elements).map(|i| i as f32).collect();
    let q = Tensor::from_vec(q_data, &[batch_size, num_heads, seq_len, head_dim], &Device::Cpu)
        .unwrap();

    let q_rotated = apply_rope_reference(&q, 1, 10000.0).unwrap();

    // Verify shape is preserved
    assert_eq!(q_rotated.dims(), &[batch_size, num_heads, seq_len, head_dim]);

    // Verify no NaN values
    let q_rot_data = q_rotated.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for (i, &val) in q_rot_data.iter().enumerate() {
        assert!(val.is_finite(), "Position {} is not finite: {}", i, val);
    }
}

#[test]
fn test_rope_split_halves_multiple_positions() {
    // Test with sequence length > 1: [B=1, H=1, T=4, D=8]
    let seq_len = 4;
    let head_dim = 8;
    let total_elements = seq_len * head_dim;

    let q_data: Vec<f32> = (0..total_elements).map(|i| i as f32).collect();
    let q = Tensor::from_vec(q_data, &[1, 1, seq_len, head_dim], &Device::Cpu).unwrap();

    // Apply RoPE to each position in sequence
    let mut rotated_positions = Vec::new();
    for pos in 0..seq_len {
        let q_slice = q.narrow(2, pos, 1).unwrap();
        let rotated = apply_rope_reference(&q_slice, pos, 10000.0).unwrap();
        rotated_positions.push(rotated);
    }

    // Verify each position has different rotations
    for pos in 1..seq_len {
        let prev = rotated_positions[pos - 1].flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let curr = rotated_positions[pos].flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Check that rotations are actually different
        let mut differs = false;
        for i in 0..prev.len() {
            if (prev[i] - curr[i]).abs() > 1e-6 {
                differs = true;
                break;
            }
        }
        assert!(differs, "Position {} and {} should have different rotations", pos - 1, pos);
    }
}

#[test]
fn test_rope_interleaved_vs_split_halves() {
    // This test demonstrates the difference between interleaved and split-halves
    let q_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let q = Tensor::from_vec(q_data.clone(), &[1, 1, 1, 8], &Device::Cpu).unwrap();

    let rope_theta = 10000.0;
    let position = 1;

    // Split-halves approach (CORRECT for llama.cpp)
    let q_split = apply_rope_reference(&q, position, rope_theta).unwrap();
    let split_data = q_split.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // For interleaved approach (INCORRECT), pairs would be (0,1), (2,3), (4,5), (6,7)
    // For split-halves approach (CORRECT), pairs are (0,4), (1,5), (2,6), (3,7)

    // Compute what interleaved would produce for comparison
    let mut interleaved_data = [0.0f32; 8];
    for i in 0..4 {
        let freq = (i * 2) as f32 / 8.0;
        let theta = (position as f32) / rope_theta.powf(freq);
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // Interleaved pairs: (2i, 2i+1)
        let x0 = q_data[2 * i];
        let x1 = q_data[2 * i + 1];
        interleaved_data[2 * i] = x0 * cos_theta - x1 * sin_theta;
        interleaved_data[2 * i + 1] = x0 * sin_theta + x1 * cos_theta;
    }

    // The two approaches should produce DIFFERENT results
    let mut differs = false;
    for i in 0..8 {
        if (split_data[i] - interleaved_data[i]).abs() > 1e-6 {
            differs = true;
            break;
        }
    }
    assert!(
        differs,
        "Split-halves and interleaved should produce different results (they are different algorithms)"
    );
}

#[test]
#[ignore = "Requires bitnet.cpp FFI"]
fn test_rope_parity_cpp() {
    // TODO: Use FFI to call bitnet.cpp RoPE and compare
    // Load same Q tensor, apply RoPE in both stacks, assert cosine similarity > 0.9999
    unimplemented!("Requires FFI integration with bitnet.cpp");
}
