//! Property-based tests — wave 18.
//!
//! Covers CPU kernel modules that lack proptest coverage:
//! residual, gating, dequant, concat, embedding (bag/pack/positional),
//! fusion (rmsnorm_linear, gelu_linear, softmax_mask), attention_mask
//! (sliding window, padding, combine), linear, transpose (flatten/squeeze),
//! loss (smooth_l1, contrastive, kl_divergence), and quantize (ternary/binary).

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::attention_mask::{
    combine_masks, create_causal_mask, create_padding_mask, create_sliding_window_mask,
};
use bitnet_kernels::cpu::concat::ConcatKernel;
use bitnet_kernels::cpu::dequant::{
    dequant_i2s_block, dequant_i2s_row, dequant_ternary, pack_ternary,
};
use bitnet_kernels::cpu::embedding::{
    embedding_lookup, normalize_embeddings, pack_embedding_table, positional_embedding,
    unpack_embedding_lookup,
};
use bitnet_kernels::cpu::fusion::{
    fused_add_normalize, fused_gelu_linear, fused_rmsnorm_linear, fused_scale_add,
    fused_softmax_mask,
};
use bitnet_kernels::cpu::gating::{GatingType, apply_gating, geglu, reglu, swiglu};
use bitnet_kernels::cpu::linear::{LinearConfig, linear_cpu};
use bitnet_kernels::cpu::loss::{
    LossReduction, contrastive_loss, l1_loss, mse_loss, smooth_l1_loss,
};
use bitnet_kernels::cpu::quantize::{
    dequantize_symmetric_i8, quantize_binary, quantize_symmetric_i8, quantize_ternary,
};
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled, add_residual_with_dropout};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, compute_frequencies};
use bitnet_kernels::cpu::transpose::TransposeKernel;
use proptest::prelude::*;

fn proptest_config() -> ProptestConfig {
    ProptestConfig { cases: 80, ..ProptestConfig::default() }
}

// ── Residual properties ─────────────────────────────────────────────

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn prop_residual_add_zero_is_identity(
        vals in proptest::collection::vec(-1e4f32..1e4, 1..512)
    ) {
        let zeros = vec![0.0f32; vals.len()];
        let mut output = vals.clone();
        add_residual(&mut output, &zeros).unwrap();
        prop_assert_eq!(&output, &vals);
    }

    #[test]
    fn prop_residual_add_commutative(
        a in proptest::collection::vec(-1e3f32..1e3, 1..256),
        b in proptest::collection::vec(-1e3f32..1e3, 1..256),
    ) {
        let n = a.len().min(b.len());
        let (a, b) = (&a[..n], &b[..n]);
        let mut ab = a.to_vec();
        add_residual(&mut ab, b).unwrap();
        let mut ba = b.to_vec();
        add_residual(&mut ba, a).unwrap();
        for (x, y) in ab.iter().zip(ba.iter()) {
            prop_assert!((x - y).abs() < 1e-4, "commutativity failed: {} vs {}", x, y);
        }
    }

    #[test]
    fn prop_residual_scaled_zero_scale_is_identity(
        vals in proptest::collection::vec(-1e4f32..1e4, 1..512),
        residual in proptest::collection::vec(-1e4f32..1e4, 1..512),
    ) {
        let n = vals.len().min(residual.len());
        let mut output = vals[..n].to_vec();
        let original = output.clone();
        add_residual_scaled(&mut output, &residual[..n], 0.0).unwrap();
        prop_assert_eq!(&output, &original);
    }

    #[test]
    fn prop_residual_scaled_unit_equals_add(
        a in proptest::collection::vec(-1e3f32..1e3, 1..256),
        b in proptest::collection::vec(-1e3f32..1e3, 1..256),
    ) {
        let n = a.len().min(b.len());
        let mut via_add = a[..n].to_vec();
        add_residual(&mut via_add, &b[..n]).unwrap();
        let mut via_scaled = a[..n].to_vec();
        add_residual_scaled(&mut via_scaled, &b[..n], 1.0).unwrap();
        for (x, y) in via_add.iter().zip(via_scaled.iter()) {
            prop_assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn prop_residual_dropout_all_true_equals_add(
        a in proptest::collection::vec(-1e3f32..1e3, 1..256),
        b in proptest::collection::vec(-1e3f32..1e3, 1..256),
    ) {
        let n = a.len().min(b.len());
        let mask = vec![true; n];
        let mut via_add = a[..n].to_vec();
        add_residual(&mut via_add, &b[..n]).unwrap();
        let mut via_dropout = a[..n].to_vec();
        add_residual_with_dropout(&mut via_dropout, &b[..n], &mask).unwrap();
        prop_assert_eq!(&via_dropout, &via_add);
    }

    #[test]
    fn prop_residual_dropout_all_false_is_identity(
        vals in proptest::collection::vec(-1e4f32..1e4, 1..256),
        residual in proptest::collection::vec(-1e4f32..1e4, 1..256),
    ) {
        let n = vals.len().min(residual.len());
        let mask = vec![false; n];
        let mut output = vals[..n].to_vec();
        let original = output.clone();
        add_residual_with_dropout(&mut output, &residual[..n], &mask).unwrap();
        prop_assert_eq!(&output, &original);
    }
}

// ── Gating properties ───────────────────────────────────────────────

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn prop_swiglu_output_length(
        gate in proptest::collection::vec(-10.0f32..10.0, 1..256),
        up in proptest::collection::vec(-10.0f32..10.0, 1..256),
    ) {
        let n = gate.len().min(up.len());
        let mut output = vec![0.0f32; n];
        swiglu(&gate[..n], &up[..n], &mut output).unwrap();
        prop_assert_eq!(output.len(), n);
    }

    #[test]
    fn prop_reglu_nonneg_gate_preserves_sign(
        gate in proptest::collection::vec(0.0f32..10.0, 1..128),
        up in proptest::collection::vec(0.0f32..10.0, 1..128),
    ) {
        let n = gate.len().min(up.len());
        let mut output = vec![0.0f32; n];
        reglu(&gate[..n], &up[..n], &mut output).unwrap();
        for &v in &output[..n] {
            prop_assert!(
                v >= 0.0,
                "ReGLU with non-negative inputs should be non-negative, got {}",
                v
            );
        }
    }

    #[test]
    fn prop_geglu_finite(
        gate in proptest::collection::vec(-10.0f32..10.0, 1..128),
        up in proptest::collection::vec(-10.0f32..10.0, 1..128),
    ) {
        let n = gate.len().min(up.len());
        let mut output = vec![0.0f32; n];
        geglu(&gate[..n], &up[..n], &mut output).unwrap();
        for &v in &output[..n] {
            prop_assert!(v.is_finite(), "GeGLU output must be finite, got {}", v);
        }
    }

    #[test]
    fn prop_swiglu_zero_up_gives_zero(
        gate in proptest::collection::vec(-10.0f32..10.0, 1..128),
    ) {
        let n = gate.len();
        let up = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];
        swiglu(&gate, &up, &mut output).unwrap();
        for &v in &output {
            prop_assert!(v.abs() < 1e-7, "SwiGLU with zero up should be zero, got {}", v);
        }
    }

    #[test]
    fn prop_apply_gating_dispatch_matches_direct(
        gate in proptest::collection::vec(-5.0f32..5.0, 1..64),
        up in proptest::collection::vec(-5.0f32..5.0, 1..64),
    ) {
        let n = gate.len().min(up.len());
        let (g, u) = (&gate[..n], &up[..n]);

        let mut direct = vec![0.0f32; n];
        swiglu(g, u, &mut direct).unwrap();
        let mut dispatched = vec![0.0f32; n];
        apply_gating(GatingType::SwiGLU, g, u, &mut dispatched).unwrap();
        prop_assert_eq!(&dispatched, &direct);
    }
}

// ── Dequantization properties ───────────────────────────────────────

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn prop_dequant_i2s_output_length(
        block_size in 1usize..128,
        scale in -10.0f32..10.0,
    ) {
        let bytes_needed = block_size.div_ceil(4);
        let packed = vec![0u8; bytes_needed];
        let result = dequant_i2s_block(&packed, scale, block_size).unwrap();
        prop_assert_eq!(result.len(), block_size);
    }

    #[test]
    fn prop_dequant_i2s_values_in_ternary_set(
        packed in proptest::collection::vec(0u8..=255, 1..32),
    ) {
        let block_size = packed.len() * 4;
        let result = dequant_i2s_block(&packed, 1.0, block_size).unwrap();
        for &v in &result {
            prop_assert!(
                v == -1.0 || v == 0.0 || v == 1.0,
                "I2S with scale=1.0 should produce {{-1,0,1}}, got {}",
                v
            );
        }
    }

    #[test]
    fn prop_dequant_ternary_output_length(
        packed in proptest::collection::vec(0u8..=255, 1..64),
        scale in 0.1f32..10.0,
    ) {
        let result = dequant_ternary(&packed, scale);
        prop_assert_eq!(result.len(), packed.len() * 4);
    }

    #[test]
    fn prop_dequant_ternary_values_bounded(
        packed in proptest::collection::vec(0u8..=255, 1..64),
        scale in 0.1f32..10.0,
    ) {
        let result = dequant_ternary(&packed, scale);
        for &v in &result {
            prop_assert!(v.abs() <= scale + 1e-6, "ternary value {} exceeds scale {}", v, scale);
        }
    }

    #[test]
    fn prop_pack_ternary_roundtrip_signs(
        vals in proptest::collection::vec(-5.0f32..5.0, 4..128),
    ) {
        let threshold = 0.5;
        let (packed, scale) = pack_ternary(&vals, threshold);
        let decoded = dequant_ternary(&packed, scale);
        for (i, (&orig, &dec)) in vals.iter().zip(decoded.iter()).enumerate() {
            if orig.abs() <= threshold {
                prop_assert!(dec.abs() < 1e-6, "idx {}: expected 0, got {}", i, dec);
            } else {
                prop_assert!(
                    orig.signum() == dec.signum() || dec == 0.0,
                    "idx {}: sign mismatch orig={} dec={}",
                    i,
                    orig,
                    dec
                );
            }
        }
    }

    #[test]
    fn prop_dequant_i2s_row_output_length(
        num_blocks in 1usize..8,
        block_size_factor in 1usize..16,
        scale in 0.1f32..5.0,
    ) {
        // Ensure block_size is a multiple of 4 so packed bytes align cleanly.
        let block_size = block_size_factor * 4;
        let bytes_per_block = block_size / 4;
        let packed = vec![0u8; num_blocks * bytes_per_block];
        let scales = vec![scale; num_blocks];
        let result = dequant_i2s_row(&packed, &scales, block_size).unwrap();
        prop_assert_eq!(result.len(), num_blocks * block_size);
    }
}

// ── Concat / split properties ───────────────────────────────────────

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn prop_concat_output_length_equals_sum(
        a in proptest::collection::vec(-10.0f32..10.0, 1..64),
        b in proptest::collection::vec(-10.0f32..10.0, 1..64),
    ) {
        let inputs: Vec<&[f32]> = vec![&a, &b];
        let sa = [a.len()];
        let sb = [b.len()];
        let shapes: Vec<&[usize]> = vec![&sa, &sb];
        let result = ConcatKernel::concat(&inputs, &shapes, 0).unwrap();
        prop_assert_eq!(result.len(), a.len() + b.len());
    }

    #[test]
    fn prop_concat_preserves_elements(
        a in proptest::collection::vec(-10.0f32..10.0, 1..64),
        b in proptest::collection::vec(-10.0f32..10.0, 1..64),
    ) {
        let inputs: Vec<&[f32]> = vec![&a, &b];
        let sa = [a.len()];
        let sb = [b.len()];
        let shapes: Vec<&[usize]> = vec![&sa, &sb];
        let result = ConcatKernel::concat(&inputs, &shapes, 0).unwrap();
        prop_assert_eq!(&result[..a.len()], &a[..]);
        prop_assert_eq!(&result[a.len()..], &b[..]);
    }

    #[test]
    fn prop_split_roundtrip(
        data in proptest::collection::vec(-10.0f32..10.0, 4..128),
    ) {
        let n = data.len();
        let even_n = n - (n % 2);
        if even_n >= 2 {
            let d = &data[..even_n];
            let shape = [even_n];
            let parts = ConcatKernel::split(d, &shape, 0, 2).unwrap();
            prop_assert_eq!(parts.len(), 2);
            prop_assert_eq!(parts[0].len() + parts[1].len(), even_n);
        }
    }

    #[test]
    fn prop_concat_output_shape(
        rows in 1usize..8,
        cols_a in 1usize..16,
        cols_b in 1usize..16,
    ) {
        let shape_a = [rows, cols_a];
        let shape_b = [rows, cols_b];
        let shapes: Vec<&[usize]> = vec![&shape_a, &shape_b];
        let result = ConcatKernel::concat_output_shape(&shapes, 1).unwrap();
        prop_assert_eq!(result, vec![rows, cols_a + cols_b]);
    }
}

// ── Attention mask properties ───────────────────────────────────────

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn prop_sliding_window_mask_size(seq_len in 1usize..64, window in 0usize..128) {
        let mask = create_sliding_window_mask(seq_len, window);
        prop_assert_eq!(mask.len(), seq_len * seq_len);
    }

    #[test]
    fn prop_sliding_window_subsumes_causal(seq_len in 1usize..32) {
        let causal = create_causal_mask(seq_len);
        let sliding = create_sliding_window_mask(seq_len, seq_len);
        for (c, s) in causal.iter().zip(sliding.iter()) {
            prop_assert_eq!(c, s);
        }
    }

    #[test]
    fn prop_sliding_window_values_are_valid(
        seq_len in 1usize..32,
        window in 1usize..64,
    ) {
        let mask = create_sliding_window_mask(seq_len, window);
        for &v in &mask {
            prop_assert!(
                v == 0.0 || v == f32::NEG_INFINITY,
                "mask value must be 0.0 or -inf, got {}",
                v
            );
        }
    }

    #[test]
    fn prop_padding_mask_valid_positions_are_zero(
        lengths in proptest::collection::vec(0usize..16, 1..8),
    ) {
        let max_len = 16;
        let mask = create_padding_mask(&lengths, max_len);
        for (b, &len) in lengths.iter().enumerate() {
            let valid = len.min(max_len);
            for j in 0..valid {
                prop_assert_eq!(
                    mask[b * max_len + j],
                    0.0,
                    "valid position ({},{}) should be 0.0",
                    b,
                    j
                );
            }
        }
    }

    #[test]
    fn prop_combine_masks_blocks_either(seq_len in 2usize..16) {
        let causal = create_causal_mask(seq_len);
        let zero_mask = vec![0.0f32; seq_len * seq_len];
        let combined = combine_masks(&causal, &zero_mask, seq_len);
        for (c, r) in causal.iter().zip(combined.iter()) {
            if c.is_finite() {
                prop_assert!((c - r).abs() < 1e-6);
            } else {
                prop_assert!(r.is_infinite() && r.is_sign_negative());
            }
        }
    }

    #[test]
    fn prop_causal_mask_lower_triangle_zero(seq_len in 1usize..32) {
        let mask = create_causal_mask(seq_len);
        for i in 0..seq_len {
            for j in 0..=i {
                prop_assert_eq!(
                    mask[i * seq_len + j],
                    0.0,
                    "lower triangle at ({},{}) should be 0.0",
                    i,
                    j
                );
            }
        }
    }
}

// ── Embedding properties ────────────────────────────────────────────

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn prop_positional_embedding_shape(
        seq_len in 1usize..32,
        dim in (1usize..32).prop_map(|d| d * 2),
    ) {
        let pe = positional_embedding(seq_len, dim);
        prop_assert_eq!(pe.len(), seq_len * dim);
    }

    #[test]
    fn prop_positional_embedding_values_bounded(
        seq_len in 1usize..16,
        dim in (1usize..16).prop_map(|d| d * 2),
    ) {
        let pe = positional_embedding(seq_len, dim);
        for &v in &pe {
            prop_assert!(
                (-1.0..=1.0).contains(&v),
                "positional embedding value {} outside [-1,1]",
                v
            );
        }
    }

    #[test]
    fn prop_pack_unpack_embedding_preserves_shape(
        vocab in 1usize..16,
        dim in 1usize..32,
    ) {
        let table = vec![0.5f32; vocab * dim];
        let packed = pack_embedding_table(&table, vocab, dim);
        prop_assert_eq!(packed.vocab_size, vocab);
        prop_assert_eq!(packed.embed_dim, dim);
        prop_assert_eq!(packed.data.len(), vocab * dim);
        prop_assert_eq!(packed.scales.len(), vocab);
    }

    #[test]
    fn prop_pack_unpack_embedding_roundtrip(
        vocab in 2usize..8,
        dim in 2usize..16,
    ) {
        let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32 * 0.1) - 1.0).collect();
        let packed = pack_embedding_table(&table, vocab, dim);
        let indices: Vec<u32> = (0..vocab as u32).collect();
        let unpacked = unpack_embedding_lookup(&packed, &indices).unwrap();
        for (orig, recon) in table.iter().zip(unpacked.iter()) {
            let err = (orig - recon).abs();
            prop_assert!(
                err < 0.1,
                "roundtrip error {} too large (orig={}, recon={})",
                err,
                orig,
                recon
            );
        }
    }

    #[test]
    fn prop_normalize_embeddings_unit_norm(
        n_vecs in 1usize..8,
        dim in 2usize..16,
    ) {
        let mut emb: Vec<f32> = (0..n_vecs * dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
        normalize_embeddings(&mut emb, dim);
        for chunk in emb.chunks(dim) {
            let norm: f32 = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assert!((norm - 1.0).abs() < 1e-4, "norm should be 1.0, got {}", norm);
        }
    }

    #[test]
    fn prop_embedding_lookup_correct_dim(
        vocab in 2usize..16,
        dim in 1usize..32,
        num_lookups in 1usize..8,
    ) {
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
        let indices: Vec<u32> = (0..num_lookups).map(|i| (i % vocab) as u32).collect();
        let result = embedding_lookup(&table, &indices, dim).unwrap();
        prop_assert_eq!(result.len(), num_lookups * dim);
    }
}

// ── Fusion properties ───────────────────────────────────────────────

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn prop_fused_scale_add_commutative_with_unit_scale(
        a in proptest::collection::vec(-10.0f32..10.0, 1..128),
        b in proptest::collection::vec(-10.0f32..10.0, 1..128),
    ) {
        let n = a.len().min(b.len());
        let r1 = fused_scale_add(&a[..n], &b[..n], 1.0).unwrap();
        let r2 = fused_scale_add(&b[..n], &a[..n], 1.0).unwrap();
        for (x, y) in r1.iter().zip(r2.iter()) {
            prop_assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn prop_fused_scale_add_zero_scale_returns_a(
        a in proptest::collection::vec(-10.0f32..10.0, 1..128),
        b in proptest::collection::vec(-10.0f32..10.0, 1..128),
    ) {
        let n = a.len().min(b.len());
        let result = fused_scale_add(&a[..n], &b[..n], 0.0).unwrap();
        for (r, ai) in result.iter().zip(a[..n].iter()) {
            prop_assert!((r - ai).abs() < 1e-6);
        }
    }

    #[test]
    fn prop_fused_rmsnorm_linear_output_dim(
        n in 2usize..16,
        out_dim in 1usize..8,
    ) {
        let input = vec![1.0f32; n];
        let gamma = vec![1.0f32; n];
        let weight = vec![0.1f32; out_dim * n];
        let result = fused_rmsnorm_linear(&input, &weight, &gamma, 1e-5).unwrap();
        prop_assert_eq!(result.len(), out_dim);
    }

    #[test]
    fn prop_fused_rmsnorm_linear_finite(
        n in 2usize..16,
        out_dim in 1usize..8,
    ) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let gamma = vec![1.0f32; n];
        let weight: Vec<f32> = (0..out_dim * n).map(|i| (i as f32) * 0.01).collect();
        let result = fused_rmsnorm_linear(&input, &weight, &gamma, 1e-5).unwrap();
        for &v in &result {
            prop_assert!(v.is_finite(), "fused_rmsnorm_linear output must be finite");
        }
    }

    #[test]
    fn prop_fused_gelu_linear_output_dim(
        n in 2usize..16,
        out_dim in 1usize..8,
    ) {
        let input = vec![1.0f32; n];
        let weight = vec![0.1f32; out_dim * n];
        let bias = vec![0.0f32; out_dim];
        let result = fused_gelu_linear(&input, &weight, &bias).unwrap();
        prop_assert_eq!(result.len(), out_dim);
    }

    #[test]
    fn prop_fused_gelu_linear_no_bias_finite(
        n in 2usize..16,
        out_dim in 1usize..8,
    ) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let weight: Vec<f32> = (0..out_dim * n).map(|i| (i as f32) * 0.02).collect();
        let result = fused_gelu_linear(&input, &weight, &[]).unwrap();
        for &v in &result {
            prop_assert!(v.is_finite());
        }
    }

    #[test]
    fn prop_fused_softmax_mask_sums_to_one(
        scores in proptest::collection::vec(-5.0f32..5.0, 1..64),
    ) {
        let n = scores.len();
        let mask = vec![0.0f32; n];
        let result = fused_softmax_mask(&scores, &mask, 1.0).unwrap();
        let sum: f32 = result.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-4, "softmax should sum to 1.0, got {}", sum);
    }

    #[test]
    fn prop_fused_softmax_mask_output_nonneg(
        scores in proptest::collection::vec(-5.0f32..5.0, 1..64),
    ) {
        let n = scores.len();
        let mask = vec![0.0f32; n];
        let result = fused_softmax_mask(&scores, &mask, 1.0).unwrap();
        for &v in &result {
            prop_assert!(v >= 0.0, "softmax output must be non-negative, got {}", v);
        }
    }

    #[test]
    fn prop_fused_add_normalize_output_length(
        n in 2usize..64,
    ) {
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b = vec![0.1f32; n];
        let gamma = vec![1.0f32; n];
        let result = fused_add_normalize(&a, &b, &gamma, 1e-5).unwrap();
        prop_assert_eq!(result.len(), n);
    }
}

// ── Linear layer properties ─────────────────────────────────────────

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn prop_linear_output_shape(
        batch in 1usize..4,
        in_f in 1usize..16,
        out_f in 1usize..16,
    ) {
        let config = LinearConfig::new(batch, in_f, out_f).unwrap();
        let x = vec![0.1f32; batch * in_f];
        let w = vec![0.1f32; out_f * in_f];
        let mut output = vec![0.0f32; batch * out_f];
        linear_cpu(&x, &w, None, &mut output, &config).unwrap();
        prop_assert_eq!(output.len(), batch * out_f);
    }

    #[test]
    fn prop_linear_zero_weight_zero_bias_gives_zero(
        batch in 1usize..4,
        in_f in 1usize..16,
        out_f in 1usize..16,
    ) {
        let config = LinearConfig::new(batch, in_f, out_f).unwrap();
        let x: Vec<f32> = (0..batch * in_f).map(|i| i as f32 * 0.1).collect();
        let w = vec![0.0f32; out_f * in_f];
        let mut output = vec![999.0f32; batch * out_f];
        linear_cpu(&x, &w, None, &mut output, &config).unwrap();
        for &v in &output {
            prop_assert!(v.abs() < 1e-6, "zero weight should give zero output, got {}", v);
        }
    }

    #[test]
    fn prop_linear_with_bias_adds_bias(
        in_f in 1usize..8,
        out_f in 1usize..8,
    ) {
        let config = LinearConfig::new(1, in_f, out_f).unwrap();
        let x = vec![0.0f32; in_f];
        let w = vec![0.0f32; out_f * in_f];
        let bias: Vec<f32> = (0..out_f).map(|i| i as f32 + 1.0).collect();
        let mut output = vec![0.0f32; out_f];
        linear_cpu(&x, &w, Some(&bias), &mut output, &config).unwrap();
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(
                (v - bias[i]).abs() < 1e-6,
                "output[{}]={} should equal bias={}",
                i,
                v,
                bias[i]
            );
        }
    }
}

// ── Transpose / reshape properties ──────────────────────────────────

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn prop_flatten_preserves_elements(
        data in proptest::collection::vec(-10.0f32..10.0, 1..128),
    ) {
        let n = data.len();
        let shape = [n];
        let (flat_data, flat_shape) = TransposeKernel::flatten(&data, &shape, 0, 0).unwrap();
        prop_assert_eq!(&flat_data, &data);
        prop_assert_eq!(flat_shape, vec![n]);
    }

    #[test]
    fn prop_squeeze_removes_ones(dims in proptest::collection::vec(1usize..4, 1..6)) {
        let mut shape = dims.clone();
        if shape.len() > 1 {
            shape.insert(1, 1);
        }
        shape.push(1);
        let squeezed = TransposeKernel::squeeze(&shape);
        for &d in &squeezed {
            prop_assert!(
                d > 1 || squeezed.len() == 1,
                "squeezed shape should not contain 1s (except scalar): {:?}",
                squeezed
            );
        }
    }

    #[test]
    fn prop_unsqueeze_adds_dim(
        shape in proptest::collection::vec(1usize..8, 1..4),
    ) {
        let original_len = shape.len();
        let unsqueezed = TransposeKernel::unsqueeze(&shape, 0).unwrap();
        prop_assert_eq!(unsqueezed.len(), original_len + 1);
        prop_assert_eq!(unsqueezed[0], 1);
    }

    #[test]
    fn prop_contiguous_strides_last_is_one(
        shape in proptest::collection::vec(1usize..8, 1..5),
    ) {
        let strides = TransposeKernel::contiguous_strides(&shape);
        prop_assert_eq!(strides.len(), shape.len());
        if !strides.is_empty() {
            prop_assert_eq!(*strides.last().unwrap(), 1);
        }
    }

    #[test]
    fn prop_is_contiguous_for_contiguous_strides(
        shape in proptest::collection::vec(1usize..8, 1..5),
    ) {
        let strides = TransposeKernel::contiguous_strides(&shape);
        prop_assert!(TransposeKernel::is_contiguous(&shape, &strides));
    }
}

// ── Loss function properties ────────────────────────────────────────

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn prop_smooth_l1_nonnegative(
        a in proptest::collection::vec(-10.0f32..10.0, 1..64),
        b in proptest::collection::vec(-10.0f32..10.0, 1..64),
    ) {
        let n = a.len().min(b.len());
        let result = smooth_l1_loss(&a[..n], &b[..n], 1.0, LossReduction::Mean);
        if let Ok(loss) = result {
            prop_assert!(loss >= 0.0, "smooth L1 loss must be non-negative, got {}", loss);
        }
    }

    #[test]
    fn prop_smooth_l1_zero_for_identical(
        vals in proptest::collection::vec(-10.0f32..10.0, 1..64),
    ) {
        let loss = smooth_l1_loss(&vals, &vals, 1.0, LossReduction::Mean).unwrap();
        prop_assert!(loss.abs() < 1e-6, "smooth L1 of identical should be 0, got {}", loss);
    }

    #[test]
    fn prop_smooth_l1_leq_l1(
        a in proptest::collection::vec(-5.0f32..5.0, 1..64),
        b in proptest::collection::vec(-5.0f32..5.0, 1..64),
    ) {
        let n = a.len().min(b.len());
        let sl1 = smooth_l1_loss(&a[..n], &b[..n], 1.0, LossReduction::Sum).unwrap();
        let l1_val = l1_loss(&a[..n], &b[..n], LossReduction::Sum).unwrap();
        prop_assert!(sl1 <= l1_val + 1e-4, "smooth L1 {} should be <= L1 {}", sl1, l1_val);
    }

    #[test]
    fn prop_contrastive_loss_nonnegative(
        a in proptest::collection::vec(-5.0f32..5.0, 2..32),
        b in proptest::collection::vec(-5.0f32..5.0, 2..32),
        label in 0.0f32..1.0,
        margin in 0.5f32..3.0,
    ) {
        let n = a.len().min(b.len());
        let loss = contrastive_loss(&a[..n], &b[..n], label, margin).unwrap();
        prop_assert!(loss >= 0.0, "contrastive loss must be non-negative, got {}", loss);
    }

    #[test]
    fn prop_mse_triangle_inequality(
        a in proptest::collection::vec(-5.0f32..5.0, 2..32),
        b in proptest::collection::vec(-5.0f32..5.0, 2..32),
    ) {
        let n = a.len().min(b.len());
        let mse_ab = mse_loss(&a[..n], &b[..n], LossReduction::Sum).unwrap();
        let mse_ba = mse_loss(&b[..n], &a[..n], LossReduction::Sum).unwrap();
        prop_assert!((mse_ab - mse_ba).abs() < 1e-3, "MSE should be symmetric");
    }
}

// ── Quantize properties ─────────────────────────────────────────────

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn prop_quantize_ternary_values_in_set(
        vals in proptest::collection::vec(-10.0f32..10.0, 1..128),
        threshold in 0.01f32..5.0,
    ) {
        let result = quantize_ternary(&vals, threshold);
        for &v in &result {
            prop_assert!(
                v == -1 || v == 0 || v == 1,
                "ternary quantize must produce {{-1,0,1}}, got {}",
                v
            );
        }
    }

    #[test]
    fn prop_quantize_binary_values_in_set(
        vals in proptest::collection::vec(-10.0f32..10.0, 1..128),
    ) {
        let result = quantize_binary(&vals);
        for &v in &result {
            prop_assert!(
                v == -1 || v == 1,
                "binary quantize must produce {{-1,1}}, got {}",
                v
            );
        }
    }

    #[test]
    fn prop_symmetric_i8_roundtrip_preserves_length(
        vals in proptest::collection::vec(-100.0f32..100.0, 1..128),
        bits in 2u8..=8,
    ) {
        let (quantized, scale) = quantize_symmetric_i8(&vals, bits);
        let dequantized = dequantize_symmetric_i8(&quantized, scale);
        prop_assert_eq!(dequantized.len(), vals.len());
    }

    #[test]
    fn prop_symmetric_i8_roundtrip_bounded_error(
        vals in proptest::collection::vec(-10.0f32..10.0, 1..64),
    ) {
        let (quantized, scale) = quantize_symmetric_i8(&vals, 8);
        let dequantized = dequantize_symmetric_i8(&quantized, scale);
        for (orig, recon) in vals.iter().zip(dequantized.iter()) {
            let err = (orig - recon).abs();
            prop_assert!(err < 1.0, "roundtrip error {} too large", err);
        }
    }

    #[test]
    fn prop_quantize_ternary_respects_threshold(
        vals in proptest::collection::vec(-10.0f32..10.0, 1..128),
        threshold in 0.01f32..5.0,
    ) {
        let result = quantize_ternary(&vals, threshold);
        for (&orig, &q) in vals.iter().zip(result.iter()) {
            if orig.abs() <= threshold {
                prop_assert_eq!(q, 0, "value {} within threshold {} should be 0", orig, threshold);
            }
        }
    }
}

// ── RoPE additional properties ──────────────────────────────────────

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn prop_rope_frequencies_length(
        head_dim in (1usize..16).prop_map(|d| d * 2),
        max_seq in 1usize..32,
    ) {
        let config = RopeConfig::new(head_dim, max_seq);
        let freqs = compute_frequencies(&config);
        prop_assert_eq!(freqs.len(), max_seq * head_dim);
    }

    #[test]
    fn prop_rope_apply_preserves_norm(
        head_dim in (1usize..8).prop_map(|d| d * 2),
    ) {
        let max_seq = 4;
        let config = RopeConfig::new(head_dim, max_seq);
        let freqs = compute_frequencies(&config);
        let mut data: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope(&mut data, 0, head_dim, &freqs);
        let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "RoPE should preserve norm: before={}, after={}",
            norm_before,
            norm_after
        );
    }
}
