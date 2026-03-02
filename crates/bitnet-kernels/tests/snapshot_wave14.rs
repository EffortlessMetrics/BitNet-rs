//! Wave 14 snapshot tests for `bitnet-kernels` — CPU kernel output and
//! error-message snapshots for modules added after wave 11.
//!
//! Covers: residual, concat, dequant, attention_mask, gating.

// =========================================================================
// Section 1 — Residual connection snapshots
// =========================================================================

mod residual_snapshots {
    use bitnet_kernels::cpu::residual::{
        add_residual, add_residual_scaled, add_residual_with_dropout,
    };

    #[test]
    fn residual_add_basic_result() {
        let mut output = vec![1.0_f32, 2.0, 3.0, 4.0];
        let residual = vec![0.5, -0.5, 1.0, -1.0];
        add_residual(&mut output, &residual).unwrap();
        insta::assert_debug_snapshot!(output);
    }

    #[test]
    fn residual_scaled_result() {
        let mut output = vec![0.0_f32; 4];
        let residual = vec![1.0, 2.0, 3.0, 4.0];
        add_residual_scaled(&mut output, &residual, 0.5).unwrap();
        insta::assert_debug_snapshot!(output);
    }

    #[test]
    fn residual_with_dropout_result() {
        let mut output = vec![1.0_f32, 2.0, 3.0, 4.0];
        let residual = vec![10.0, 20.0, 30.0, 40.0];
        let mask = vec![true, false, true, false];
        add_residual_with_dropout(&mut output, &residual, &mask).unwrap();
        insta::assert_debug_snapshot!(output);
    }

    #[test]
    fn residual_length_mismatch_error() {
        let mut output = vec![1.0_f32, 2.0];
        let err = add_residual(&mut output, &[1.0]).unwrap_err();
        insta::assert_snapshot!(err.to_string());
    }

    #[test]
    fn residual_scaled_length_mismatch_error() {
        let mut output = vec![1.0_f32];
        let err = add_residual_scaled(&mut output, &[1.0, 2.0], 1.0).unwrap_err();
        insta::assert_snapshot!(err.to_string());
    }

    #[test]
    fn residual_dropout_mask_mismatch_error() {
        let mut output = vec![1.0_f32, 2.0];
        let err = add_residual_with_dropout(&mut output, &[1.0, 2.0], &[true]).unwrap_err();
        insta::assert_snapshot!(err.to_string());
    }
}

// =========================================================================
// Section 2 — Concat kernel output-shape snapshots
// =========================================================================

mod concat_snapshots {
    use bitnet_kernels::cpu::concat::ConcatKernel;

    #[test]
    fn concat_output_shape_axis0() {
        let s1: &[usize] = &[2, 3];
        let s2: &[usize] = &[4, 3];
        let shape = ConcatKernel::concat_output_shape(&[s1, s2], 0).unwrap();
        insta::assert_debug_snapshot!(shape);
    }

    #[test]
    fn concat_output_shape_axis1() {
        let s1: &[usize] = &[2, 3];
        let s2: &[usize] = &[2, 5];
        let shape = ConcatKernel::concat_output_shape(&[s1, s2], 1).unwrap();
        insta::assert_debug_snapshot!(shape);
    }

    #[test]
    fn stack_output_shape_axis0() {
        let shape = ConcatKernel::stack_output_shape(&[3, 4], 0, 5).unwrap();
        insta::assert_debug_snapshot!(shape);
    }

    #[test]
    fn stack_output_shape_axis1() {
        let shape = ConcatKernel::stack_output_shape(&[3, 4], 1, 2).unwrap();
        insta::assert_debug_snapshot!(shape);
    }

    #[test]
    fn split_output_shapes_even() {
        let shapes = ConcatKernel::split_output_shapes(&[6, 4], 0, 3).unwrap();
        insta::assert_debug_snapshot!(shapes);
    }

    #[test]
    fn concat_axis_out_of_range_error() {
        let s1: &[usize] = &[2, 3];
        let err = ConcatKernel::concat_output_shape(&[s1], 5).unwrap_err();
        insta::assert_snapshot!(err.to_string());
    }

    #[test]
    fn concat_2d_values_axis0() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [5.0_f32, 6.0, 7.0, 8.0];
        let sa: &[usize] = &[2, 2];
        let sb: &[usize] = &[2, 2];
        let result = ConcatKernel::concat(&[&a, &b], &[sa, sb], 0).unwrap();
        insta::assert_debug_snapshot!(result);
    }
}

// =========================================================================
// Section 3 — Dequantization snapshots
// =========================================================================

mod dequant_snapshots {
    use bitnet_kernels::cpu::dequant::{
        dequant_i2s_block, dequant_i2s_row, dequant_ternary, pack_ternary,
    };

    fn pack4(vals: [i8; 4]) -> u8 {
        let mut byte = 0u8;
        for (i, &v) in vals.iter().enumerate() {
            let code: u8 = match v {
                1 => 0b01,
                -1 => 0b11,
                _ => 0b00,
            };
            byte |= code << (i * 2);
        }
        byte
    }

    #[test]
    fn dequant_i2s_block_mixed_values() {
        let packed = vec![pack4([1, -1, 0, 1])];
        let result = dequant_i2s_block(&packed, 2.0, 4).unwrap();
        insta::assert_debug_snapshot!(result);
    }

    #[test]
    fn dequant_ternary_two_bytes() {
        let packed = vec![pack4([1, -1, 0, 1]), pack4([-1, 0, -1, 1])];
        let result = dequant_ternary(&packed, 1.5);
        insta::assert_debug_snapshot!(result);
    }

    #[test]
    fn dequant_i2s_row_multi_block() {
        let packed = vec![pack4([1, -1, 0, 1]), pack4([-1, 1, 1, 0])];
        let scales = vec![2.0, 3.0];
        let result = dequant_i2s_row(&packed, &scales, 4).unwrap();
        insta::assert_debug_snapshot!(result);
    }

    #[test]
    fn pack_ternary_round_trip_snapshot() {
        let values = vec![1.0_f32, -0.8, 0.05, 0.9, -1.2];
        let (packed, scale) = pack_ternary(&values, 0.1);
        let deq = dequant_ternary(&packed, scale);
        insta::assert_debug_snapshot!("packed_bytes", &packed);
        insta::assert_snapshot!("scale", format!("{scale:.6}"));
        insta::assert_debug_snapshot!("round_trip_values", &deq);
    }

    #[test]
    fn dequant_i2s_block_insufficient_bytes_error() {
        let err = dequant_i2s_block(&[0u8], 1.0, 8).unwrap_err();
        insta::assert_snapshot!(err.to_string());
    }

    #[test]
    fn dequant_i2s_row_zero_block_size_error() {
        let err = dequant_i2s_row(&[0u8; 4], &[1.0], 0).unwrap_err();
        insta::assert_snapshot!(err.to_string());
    }

    #[test]
    fn dequant_i2s_row_insufficient_scales_error() {
        let err = dequant_i2s_row(&[0u8; 4], &[1.0, 2.0], 4).unwrap_err();
        insta::assert_snapshot!(err.to_string());
    }
}

// =========================================================================
// Section 4 — Attention mask output snapshots
// =========================================================================

mod attention_mask_snapshots {
    use bitnet_kernels::cpu::attention_mask::{
        combine_masks, create_causal_mask, create_padding_mask, create_sliding_window_mask,
    };

    /// Replace -inf with a readable string for snapshot stability.
    fn mask_display(mask: &[f32], cols: usize) -> String {
        mask.chunks(cols)
            .map(|row| {
                row.iter()
                    .map(|&v| {
                        if v == f32::NEG_INFINITY {
                            " -inf".to_string()
                        } else {
                            format!("{v:5.1}")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn causal_mask_4x4() {
        let mask = create_causal_mask(4);
        insta::assert_snapshot!(mask_display(&mask, 4));
    }

    #[test]
    fn padding_mask_batch3() {
        let mask = create_padding_mask(&[2, 4, 0], 4);
        insta::assert_snapshot!(mask_display(&mask, 4));
    }

    #[test]
    fn sliding_window_mask_4x4_window2() {
        let mask = create_sliding_window_mask(4, 2);
        insta::assert_snapshot!(mask_display(&mask, 4));
    }

    #[test]
    fn sliding_window_mask_4x4_window1() {
        let mask = create_sliding_window_mask(4, 1);
        insta::assert_snapshot!(mask_display(&mask, 4));
    }

    #[test]
    fn combine_causal_and_padding() {
        let causal = create_causal_mask(3);
        let mut pad = vec![0.0_f32; 9];
        // Block column 2 in all rows (simulates padding at position 2).
        for i in 0..3 {
            pad[i * 3 + 2] = f32::NEG_INFINITY;
        }
        let combined = combine_masks(&causal, &pad, 3);
        insta::assert_snapshot!(mask_display(&combined, 3));
    }
}

// =========================================================================
// Section 5 — Gating kernel snapshots
// =========================================================================

mod gating_snapshots {
    use bitnet_kernels::cpu::gating::{GatingType, apply_gating, geglu, reglu, swiglu};

    #[test]
    fn gating_type_all_variants_debug() {
        let types = [GatingType::SwiGLU, GatingType::GeGLU, GatingType::ReGLU];
        insta::assert_debug_snapshot!(types);
    }

    #[test]
    fn swiglu_known_output() {
        let gate = [0.0_f32, 1.0, -1.0, 2.0];
        let up = [1.0_f32, 1.0, 1.0, 0.5];
        let mut out = [0.0_f32; 4];
        swiglu(&gate, &up, &mut out).unwrap();
        let rounded: Vec<String> = out.iter().map(|v| format!("{v:.4}")).collect();
        insta::assert_debug_snapshot!(rounded);
    }

    #[test]
    fn geglu_known_output() {
        let gate = [0.0_f32, 1.0, -1.0, 2.0];
        let up = [1.0_f32, 1.0, 1.0, 0.5];
        let mut out = [0.0_f32; 4];
        geglu(&gate, &up, &mut out).unwrap();
        let rounded: Vec<String> = out.iter().map(|v| format!("{v:.4}")).collect();
        insta::assert_debug_snapshot!(rounded);
    }

    #[test]
    fn reglu_known_output() {
        let gate = [0.0_f32, 1.0, -1.0, 2.0];
        let up = [1.0_f32, 1.0, 1.0, 0.5];
        let mut out = [0.0_f32; 4];
        reglu(&gate, &up, &mut out).unwrap();
        insta::assert_debug_snapshot!(out);
    }

    #[test]
    fn apply_gating_dispatch_results() {
        let gate = [1.0_f32];
        let up = [2.0_f32];
        let mut results = Vec::new();
        for gtype in [GatingType::SwiGLU, GatingType::GeGLU, GatingType::ReGLU] {
            let mut out = [0.0_f32];
            apply_gating(gtype, &gate, &up, &mut out).unwrap();
            results.push(format!("{:?}: {:.4}", gtype, out[0]));
        }
        insta::assert_debug_snapshot!(results);
    }

    #[test]
    fn gating_empty_input_error() {
        let mut out = [0.0_f32; 1];
        let err = swiglu(&[], &[], &mut out).unwrap_err();
        insta::assert_snapshot!(err.to_string());
    }

    #[test]
    fn gating_length_mismatch_error() {
        let mut out = [0.0_f32; 4];
        let err = geglu(&[1.0, 2.0], &[1.0], &mut out).unwrap_err();
        insta::assert_snapshot!(err.to_string());
    }

    #[test]
    fn gating_output_too_short_error() {
        let mut out = [0.0_f32; 1];
        let err = reglu(&[1.0, 2.0], &[1.0, 2.0], &mut out).unwrap_err();
        insta::assert_snapshot!(err.to_string());
    }
}
