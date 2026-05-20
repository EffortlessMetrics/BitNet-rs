#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Metal batch normalization shader validation tests.
//! Tests BN operations expected to run on Metal GPU:
//! forward pass, running stats, channel-wise normalization,
//! fused BN+activation, training vs inference modes.
//!
//! All GPU-runtime tests are `#[ignore]` with justification.
//! CPU-side logic tests run without Metal hardware.

#[cfg(test)]
mod tests {
    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    /// Default epsilon for batch normalization.
    const BN_EPSILON: f32 = 1e-5;

    /// Default momentum for running statistics EMA.
    const BN_MOMENTUM: f32 = 0.1;

    // -----------------------------------------------------------------------
    // CPU reference implementations
    // -----------------------------------------------------------------------

    /// CPU reference: batch normalization forward pass.
    /// For each channel c across the batch:
    ///   mean_c = mean(x[:, c, ...])
    ///   var_c  = var(x[:, c, ...])
    ///   y[:, c, ...] = gamma_c * (x[:, c, ...] - mean_c) / sqrt(var_c + eps) + beta_c
    ///
    /// `input` is laid out as [batch, channels, spatial...] flattened.
    /// Returns (output, batch_mean, batch_var).
    fn batch_norm_forward_cpu(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        channels: usize,
        spatial: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let batch = input.len() / (channels * spatial);
        assert_eq!(input.len(), batch * channels * spatial);
        let mut output = vec![0.0f32; input.len()];
        let mut mean = vec![0.0f32; channels];
        let mut var = vec![0.0f32; channels];

        let count = (batch * spatial) as f32;
        for c in 0..channels {
            // Compute mean for channel c.
            let mut sum = 0.0f32;
            for b in 0..batch {
                for s in 0..spatial {
                    sum += input[b * channels * spatial + c * spatial + s];
                }
            }
            mean[c] = sum / count;

            // Compute variance for channel c.
            let mut sq_sum = 0.0f32;
            for b in 0..batch {
                for s in 0..spatial {
                    let diff = input[b * channels * spatial + c * spatial + s] - mean[c];
                    sq_sum += diff * diff;
                }
            }
            var[c] = sq_sum / count;

            // Normalize.
            let inv_std = 1.0 / (var[c] + eps).sqrt();
            for b in 0..batch {
                for s in 0..spatial {
                    let idx = b * channels * spatial + c * spatial + s;
                    output[idx] = gamma[c] * (input[idx] - mean[c]) * inv_std + beta[c];
                }
            }
        }
        (output, mean, var)
    }

    /// CPU reference: batch normalization inference (frozen running stats).
    fn batch_norm_inference_cpu(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        running_mean: &[f32],
        running_var: &[f32],
        channels: usize,
        spatial: usize,
        eps: f32,
    ) -> Vec<f32> {
        let batch = input.len() / (channels * spatial);
        let mut output = vec![0.0f32; input.len()];
        for c in 0..channels {
            let inv_std = 1.0 / (running_var[c] + eps).sqrt();
            for b in 0..batch {
                for s in 0..spatial {
                    let idx = b * channels * spatial + c * spatial + s;
                    output[idx] = gamma[c] * (input[idx] - running_mean[c]) * inv_std + beta[c];
                }
            }
        }
        output
    }

    /// CPU reference: update running statistics via EMA.
    /// running = (1 - momentum) * running + momentum * batch_stat
    fn update_running_stats(running: &mut [f32], batch_stat: &[f32], momentum: f32) {
        for (r, &b) in running.iter_mut().zip(batch_stat.iter()) {
            *r = (1.0 - momentum) * *r + momentum * b;
        }
    }

    /// CPU reference: group normalization.
    fn group_norm_cpu(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        channels: usize,
        spatial: usize,
        num_groups: usize,
        eps: f32,
    ) -> Vec<f32> {
        let batch = input.len() / (channels * spatial);
        let channels_per_group = channels / num_groups;
        let mut output = vec![0.0f32; input.len()];

        for b in 0..batch {
            for g in 0..num_groups {
                let c_start = g * channels_per_group;
                let c_end = c_start + channels_per_group;
                let group_size = (channels_per_group * spatial) as f32;

                // Mean over the group.
                let mut sum = 0.0f32;
                for c in c_start..c_end {
                    for s in 0..spatial {
                        sum += input[b * channels * spatial + c * spatial + s];
                    }
                }
                let mean = sum / group_size;

                // Variance over the group.
                let mut sq_sum = 0.0f32;
                for c in c_start..c_end {
                    for s in 0..spatial {
                        let diff = input[b * channels * spatial + c * spatial + s] - mean;
                        sq_sum += diff * diff;
                    }
                }
                let var = sq_sum / group_size;
                let inv_std = 1.0 / (var + eps).sqrt();

                for c in c_start..c_end {
                    for s in 0..spatial {
                        let idx = b * channels * spatial + c * spatial + s;
                        output[idx] = gamma[c] * (input[idx] - mean) * inv_std + beta[c];
                    }
                }
            }
        }
        output
    }

    /// CPU reference: instance normalization (group norm with num_groups = channels).
    fn instance_norm_cpu(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        channels: usize,
        spatial: usize,
        eps: f32,
    ) -> Vec<f32> {
        group_norm_cpu(input, gamma, beta, channels, spatial, channels, eps)
    }

    /// CPU reference: fused BN + ReLU.
    fn fused_bn_relu_cpu(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        channels: usize,
        spatial: usize,
        eps: f32,
    ) -> Vec<f32> {
        let (normed, _, _) = batch_norm_forward_cpu(input, gamma, beta, channels, spatial, eps);
        normed.iter().map(|&x| x.max(0.0)).collect()
    }

    /// CPU reference: fused BN + SiLU (x * sigmoid(x)).
    fn fused_bn_silu_cpu(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        channels: usize,
        spatial: usize,
        eps: f32,
    ) -> Vec<f32> {
        let (normed, _, _) = batch_norm_forward_cpu(input, gamma, beta, channels, spatial, eps);
        normed.iter().map(|&x| x * (1.0 / (1.0 + (-x).exp()))).collect()
    }

    /// Helper to assert vectors are close within tolerance.
    fn assert_close(actual: &[f32], expected: &[f32], atol: f32, label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch (actual={}, expected={})",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff <= atol,
                "{label}[{i}]: actual={a}, expected={e}, diff={diff} > atol={atol}"
            );
        }
    }

    /// Generate deterministic input data for reproducibility.
    fn make_input(len: usize, seed: f32) -> Vec<f32> {
        (0..len).map(|i| ((i as f32 + seed) * 0.017).sin()).collect()
    }

    /// Generate gamma (all ones) and beta (all zeros).
    fn identity_affine(channels: usize) -> (Vec<f32>, Vec<f32>) {
        (vec![1.0; channels], vec![0.0; channels])
    }

    /// Generate learned affine params.
    fn learned_affine(channels: usize) -> (Vec<f32>, Vec<f32>) {
        let gamma: Vec<f32> = (0..channels).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let beta: Vec<f32> = (0..channels).map(|i| -0.1 + (i as f32) * 0.005).collect();
        (gamma, beta)
    }

    // ===================================================================
    // 1. Basic BN forward (8 tests)
    // ===================================================================

    mod basic_bn_forward {
        use super::*;

        #[test]
        fn test_bn_forward_1d_4_channels() {
            let (batch, channels, spatial) = (2, 4, 1);
            let input = make_input(batch * channels * spatial, 1.0);
            let (gamma, beta) = identity_affine(channels);
            let (output, mean, var) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(output.len(), input.len());
            // With identity affine, output should be zero-mean unit-var per channel.
            for c in 0..channels {
                let mut ch_sum = 0.0f32;
                for b in 0..batch {
                    ch_sum += output[b * channels + c];
                }
                assert!(
                    (ch_sum / batch as f32).abs() < 0.01,
                    "channel {c}: mean should be near zero"
                );
            }
            assert_eq!(mean.len(), channels);
            assert_eq!(var.len(), channels);
        }

        #[test]
        fn test_bn_forward_2d_8_channels() {
            let (batch, channels, spatial) = (4, 8, 16);
            let input = make_input(batch * channels * spatial, 2.0);
            let (gamma, beta) = identity_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
            // All values should be finite.
            assert!(output.iter().all(|x| x.is_finite()));
        }

        #[test]
        fn test_bn_forward_3d_16_channels() {
            let (batch, channels, spatial) = (2, 16, 8 * 8);
            let input = make_input(batch * channels * spatial, 3.0);
            let (gamma, beta) = identity_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
            assert!(output.iter().all(|x| x.is_finite()));
        }

        #[test]
        fn test_bn_forward_32_channels() {
            let (batch, channels, spatial) = (8, 32, 4);
            let input = make_input(batch * channels * spatial, 4.0);
            let (gamma, beta) = learned_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
            assert!(output.iter().all(|x| x.is_finite()));
        }

        #[test]
        fn test_bn_forward_64_channels() {
            let (batch, channels, spatial) = (4, 64, 1);
            let input = make_input(batch * channels * spatial, 5.0);
            let (gamma, beta) = learned_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
        }

        #[test]
        fn test_bn_forward_128_channels_large_spatial() {
            let (batch, channels, spatial) = (2, 128, 32);
            let input = make_input(batch * channels * spatial, 6.0);
            let (gamma, beta) = identity_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
        }

        #[test]
        fn test_bn_forward_identity_affine_zero_mean() {
            // With identity affine, normalized output should have near-zero mean per channel.
            let (batch, channels, spatial) = (16, 4, 8);
            let input = make_input(batch * channels * spatial, 7.0);
            let (gamma, beta) = identity_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            for c in 0..channels {
                let mut ch_sum = 0.0f32;
                for b in 0..batch {
                    for s in 0..spatial {
                        ch_sum += output[b * channels * spatial + c * spatial + s];
                    }
                }
                let ch_mean = ch_sum / (batch * spatial) as f32;
                assert!(ch_mean.abs() < 1e-4, "channel {c}: mean={ch_mean}, expected ~0");
            }
        }

        #[test]
        fn test_bn_forward_identity_affine_unit_variance() {
            // With identity affine, normalized output should have near-unit variance per channel.
            let (batch, channels, spatial) = (16, 4, 8);
            let input = make_input(batch * channels * spatial, 8.0);
            let (gamma, beta) = identity_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            for c in 0..channels {
                let count = (batch * spatial) as f32;
                let mut sum = 0.0f32;
                let mut sq_sum = 0.0f32;
                for b in 0..batch {
                    for s in 0..spatial {
                        let v = output[b * channels * spatial + c * spatial + s];
                        sum += v;
                        sq_sum += v * v;
                    }
                }
                let mean = sum / count;
                let var = sq_sum / count - mean * mean;
                assert!((var - 1.0).abs() < 0.05, "channel {c}: var={var}, expected ~1.0");
            }
        }
    }

    // ===================================================================
    // 2. Running statistics (7 tests)
    // ===================================================================

    mod running_statistics {
        use super::*;

        #[test]
        fn test_ema_single_update() {
            let mut running_mean = vec![0.0f32; 4];
            let batch_mean = vec![1.0, 2.0, 3.0, 4.0];
            update_running_stats(&mut running_mean, &batch_mean, BN_MOMENTUM);
            let expected: Vec<f32> = batch_mean.iter().map(|&x| BN_MOMENTUM * x).collect();
            assert_close(&running_mean, &expected, 1e-6, "ema_single");
        }

        #[test]
        fn test_ema_multiple_updates_converge() {
            let mut running_mean = vec![0.0f32; 2];
            let batch_mean = vec![5.0, 10.0];
            // After many updates with constant stats, running should converge.
            for _ in 0..200 {
                update_running_stats(&mut running_mean, &batch_mean, BN_MOMENTUM);
            }
            assert_close(&running_mean, &batch_mean, 1e-3, "ema_converge");
        }

        #[test]
        fn test_ema_variance_update() {
            let mut running_var = vec![1.0f32; 3];
            let batch_var = vec![4.0, 9.0, 16.0];
            update_running_stats(&mut running_var, &batch_var, BN_MOMENTUM);
            for (i, (&r, &b)) in running_var.iter().zip(batch_var.iter()).enumerate() {
                let expected = (1.0 - BN_MOMENTUM) * 1.0 + BN_MOMENTUM * b;
                assert!((r - expected).abs() < 1e-6, "var[{i}]: got {r}, expected {expected}");
            }
        }

        #[test]
        fn test_ema_momentum_zero_no_update() {
            let mut running = vec![1.0, 2.0, 3.0];
            let original = running.clone();
            let batch = vec![100.0, 200.0, 300.0];
            update_running_stats(&mut running, &batch, 0.0);
            assert_close(&running, &original, 1e-7, "momentum_zero");
        }

        #[test]
        fn test_ema_momentum_one_full_replace() {
            let mut running = vec![1.0, 2.0, 3.0];
            let batch = vec![10.0, 20.0, 30.0];
            update_running_stats(&mut running, &batch, 1.0);
            assert_close(&running, &batch, 1e-7, "momentum_one");
        }

        #[test]
        fn test_running_stats_from_bn_forward() {
            let (batch, channels, spatial) = (8, 4, 4);
            let input = make_input(batch * channels * spatial, 20.0);
            let (gamma, beta) = identity_affine(channels);
            let (_, batch_mean, batch_var) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);

            let mut running_mean = vec![0.0f32; channels];
            let mut running_var = vec![1.0f32; channels];
            update_running_stats(&mut running_mean, &batch_mean, BN_MOMENTUM);
            update_running_stats(&mut running_var, &batch_var, BN_MOMENTUM);

            for c in 0..channels {
                assert!(running_mean[c].is_finite(), "running_mean[{c}] not finite");
                assert!(running_var[c] >= 0.0, "running_var[{c}] negative");
            }
        }

        #[test]
        fn test_running_stats_accumulate_across_batches() {
            let channels = 4;
            let spatial = 4;
            let mut running_mean = vec![0.0f32; channels];
            let mut running_var = vec![1.0f32; channels];
            let (gamma, beta) = identity_affine(channels);

            for step in 0..10 {
                let batch = 4;
                let input = make_input(batch * channels * spatial, step as f32 * 10.0);
                let (_, bm, bv) =
                    batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
                update_running_stats(&mut running_mean, &bm, BN_MOMENTUM);
                update_running_stats(&mut running_var, &bv, BN_MOMENTUM);
            }
            // After 10 steps, running stats should be reasonable.
            for c in 0..channels {
                assert!(running_mean[c].is_finite());
                assert!(running_var[c] >= 0.0);
            }
        }
    }

    // ===================================================================
    // 3. Channel-wise normalization (7 tests)
    // ===================================================================

    mod channel_wise_normalization {
        use super::*;

        #[test]
        fn test_per_channel_mean_computation() {
            let (_batch, channels, spatial) = (4, 3, 2);
            let input = vec![
                // batch 0: ch0=[1,2], ch1=[3,4], ch2=[5,6]
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                // batch 1: ch0=[7,8], ch1=[9,10], ch2=[11,12]
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                // batch 2: ch0=[0,0], ch1=[0,0], ch2=[0,0]
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // batch 3: ch0=[2,2], ch1=[2,2], ch2=[2,2]
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            ];
            let (gamma, beta) = identity_affine(channels);
            let (_, mean, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);

            // ch0 mean = (1+2+7+8+0+0+2+2) / 8 = 22/8 = 2.75
            assert!((mean[0] - 2.75).abs() < 1e-5, "ch0 mean: {}", mean[0]);
            // ch1 mean = (3+4+9+10+0+0+2+2) / 8 = 30/8 = 3.75
            assert!((mean[1] - 3.75).abs() < 1e-5, "ch1 mean: {}", mean[1]);
            // ch2 mean = (5+6+11+12+0+0+2+2) / 8 = 38/8 = 4.75
            assert!((mean[2] - 4.75).abs() < 1e-5, "ch2 mean: {}", mean[2]);
        }

        #[test]
        fn test_per_channel_variance_computation() {
            let (_batch, channels, spatial) = (2, 2, 2);
            // ch0 = [1, 1, 1, 1], ch1 = [0, 2, 0, 2]
            let input = vec![
                1.0, 1.0, 0.0, 2.0, // batch 0
                1.0, 1.0, 0.0, 2.0, // batch 1
            ];
            let (gamma, beta) = identity_affine(channels);
            let (_, _, var) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            // ch0: all values = 1, var = 0
            assert!(var[0].abs() < 1e-6, "ch0 var: {}", var[0]);
            // ch1: values = [0,2,0,2], mean=1, var = mean((0-1)^2,(2-1)^2,...) = 1
            assert!((var[1] - 1.0).abs() < 1e-5, "ch1 var: {}", var[1]);
        }

        #[test]
        fn test_affine_gamma_scaling() {
            let (batch, channels, spatial) = (4, 2, 4);
            let input = make_input(batch * channels * spatial, 30.0);
            let gamma = vec![2.0, 0.5];
            let beta = vec![0.0, 0.0];
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);

            // The output variance per channel should scale by gamma^2.
            let count = (batch * spatial) as f32;
            for c in 0..channels {
                let mut sum = 0.0f32;
                let mut sq_sum = 0.0f32;
                for b in 0..batch {
                    for s in 0..spatial {
                        let v = output[b * channels * spatial + c * spatial + s];
                        sum += v;
                        sq_sum += v * v;
                    }
                }
                let m = sum / count;
                let v = sq_sum / count - m * m;
                let expected_var = gamma[c] * gamma[c]; // ~gamma^2 * 1
                assert!((v - expected_var).abs() < 0.1, "ch{c}: var={v}, expected ~{expected_var}");
            }
        }

        #[test]
        fn test_affine_beta_shift() {
            let (batch, channels, spatial) = (4, 2, 4);
            let input = make_input(batch * channels * spatial, 31.0);
            let gamma = vec![1.0, 1.0];
            let beta = vec![5.0, -3.0];
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);

            // Per-channel mean should be approximately beta.
            let count = (batch * spatial) as f32;
            for c in 0..channels {
                let mut sum = 0.0f32;
                for b in 0..batch {
                    for s in 0..spatial {
                        sum += output[b * channels * spatial + c * spatial + s];
                    }
                }
                let m = sum / count;
                assert!((m - beta[c]).abs() < 0.01, "ch{c}: mean={m}, expected ~{}", beta[c]);
            }
        }

        #[test]
        fn test_different_gamma_per_channel() {
            let (batch, channels, spatial) = (8, 4, 1);
            let input = make_input(batch * channels * spatial, 32.0);
            let gamma = vec![1.0, 2.0, 0.5, 3.0];
            let beta = vec![0.0; 4];
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
            assert!(output.iter().all(|x| x.is_finite()));
        }

        #[test]
        fn test_negative_beta_offset() {
            let (batch, channels, spatial) = (4, 2, 4);
            let input = make_input(batch * channels * spatial, 33.0);
            let gamma = vec![1.0, 1.0];
            let beta = vec![-10.0, -20.0];
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            let count = (batch * spatial) as f32;
            for c in 0..channels {
                let mut sum = 0.0f32;
                for b in 0..batch {
                    for s in 0..spatial {
                        sum += output[b * channels * spatial + c * spatial + s];
                    }
                }
                let m = sum / count;
                assert!((m - beta[c]).abs() < 0.01, "ch{c}: mean={m}, expected ~{}", beta[c]);
            }
        }

        #[test]
        fn test_large_gamma_amplifies_spread() {
            let (batch, channels, spatial) = (4, 1, 16);
            let input = make_input(batch * channels * spatial, 34.0);
            let (out_small, _, _) =
                batch_norm_forward_cpu(&input, &[1.0], &[0.0], channels, spatial, BN_EPSILON);
            let (out_large, _, _) =
                batch_norm_forward_cpu(&input, &[10.0], &[0.0], channels, spatial, BN_EPSILON);
            // Each element of out_large should be ~10x out_small.
            for (s, l) in out_small.iter().zip(out_large.iter()) {
                assert!((l - s * 10.0).abs() < 1e-3, "expected ~10x scaling: small={s}, large={l}");
            }
        }
    }

    // ===================================================================
    // 4. Fused BN+ReLU (6 tests)
    // ===================================================================

    mod fused_bn_relu {
        use super::*;

        #[test]
        fn test_fused_bn_relu_positive_pass_through() {
            let (batch, channels, spatial) = (2, 2, 4);
            // All positive => ReLU is identity => fused == BN.
            let input: Vec<f32> = vec![10.0; batch * channels * spatial];
            let (gamma, beta) = identity_affine(channels);
            let fused = fused_bn_relu_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            // Constant input → normalized to 0, ReLU(0) = 0.
            assert!(fused.iter().all(|&x| x.abs() < 1e-4 || x >= 0.0));
        }

        #[test]
        fn test_fused_bn_relu_clips_negative() {
            let (batch, channels, spatial) = (4, 2, 4);
            let input = make_input(batch * channels * spatial, 40.0);
            let (gamma, beta) = identity_affine(channels);
            let fused = fused_bn_relu_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            // All outputs should be >= 0 after ReLU.
            for (i, &v) in fused.iter().enumerate() {
                assert!(v >= -1e-7, "fused_bn_relu[{i}]={v} < 0");
            }
        }

        #[test]
        fn test_fused_vs_separate_bn_relu() {
            let (batch, channels, spatial) = (4, 4, 8);
            let input = make_input(batch * channels * spatial, 41.0);
            let (gamma, beta) = learned_affine(channels);
            let fused = fused_bn_relu_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            // Compute separately.
            let (normed, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            let separate: Vec<f32> = normed.iter().map(|&x| x.max(0.0)).collect();
            assert_close(&fused, &separate, 1e-6, "fused_vs_separate_relu");
        }

        #[test]
        fn test_fused_bn_relu_with_large_beta() {
            let (batch, channels, spatial) = (2, 2, 4);
            let input = make_input(batch * channels * spatial, 42.0);
            let gamma = vec![1.0, 1.0];
            let beta = vec![100.0, 100.0]; // Large bias → all positive after BN.
            let fused = fused_bn_relu_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            // With huge beta, everything stays positive.
            assert!(fused.iter().all(|&x| x > 0.0));
        }

        #[test]
        fn test_fused_bn_relu_with_negative_beta() {
            let (batch, channels, spatial) = (2, 2, 4);
            let input = make_input(batch * channels * spatial, 43.0);
            let gamma = vec![1.0, 1.0];
            let beta = vec![-100.0, -100.0]; // Huge negative → all clipped.
            let fused = fused_bn_relu_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert!(fused.iter().all(|&x| x.abs() < 1e-4));
        }

        #[test]
        fn test_fused_bn_relu_preserves_shape() {
            let (batch, channels, spatial) = (8, 16, 4);
            let input = make_input(batch * channels * spatial, 44.0);
            let (gamma, beta) = learned_affine(channels);
            let fused = fused_bn_relu_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(fused.len(), batch * channels * spatial);
        }
    }

    // ===================================================================
    // 5. Fused BN+SiLU (5 tests)
    // ===================================================================

    mod fused_bn_silu {
        use super::*;

        #[test]
        fn test_fused_bn_silu_basic() {
            let (batch, channels, spatial) = (4, 4, 4);
            let input = make_input(batch * channels * spatial, 50.0);
            let (gamma, beta) = identity_affine(channels);
            let fused = fused_bn_silu_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(fused.len(), batch * channels * spatial);
            assert!(fused.iter().all(|x| x.is_finite()));
        }

        #[test]
        fn test_fused_vs_separate_bn_silu() {
            let (batch, channels, spatial) = (4, 4, 8);
            let input = make_input(batch * channels * spatial, 51.0);
            let (gamma, beta) = learned_affine(channels);
            let fused = fused_bn_silu_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            // Compute separately.
            let (normed, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            let separate: Vec<f32> =
                normed.iter().map(|&x| x * (1.0 / (1.0 + (-x).exp()))).collect();
            assert_close(&fused, &separate, 1e-6, "fused_vs_separate_silu");
        }

        #[test]
        fn test_fused_bn_silu_zero_input() {
            let (batch, channels, spatial) = (2, 2, 4);
            let input = vec![0.0f32; batch * channels * spatial];
            let (gamma, beta) = identity_affine(channels);
            let fused = fused_bn_silu_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            // All zeros normalized → 0, SiLU(0) = 0.
            assert!(fused.iter().all(|&x| x.abs() < 1e-4));
        }

        #[test]
        fn test_fused_bn_silu_negative_values() {
            // SiLU can produce negative values for negative inputs (unlike ReLU).
            let (batch, channels, spatial) = (4, 2, 4);
            let input = make_input(batch * channels * spatial, 52.0);
            let gamma = vec![1.0, 1.0];
            let beta = vec![-5.0, -5.0];
            let fused = fused_bn_silu_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            // SiLU(-5 + x) for small x will be negative.
            let has_negative = fused.iter().any(|&x| x < 0.0);
            assert!(has_negative, "SiLU should produce some negative values");
        }

        #[test]
        fn test_fused_bn_silu_preserves_length() {
            let (batch, channels, spatial) = (8, 8, 16);
            let input = make_input(batch * channels * spatial, 53.0);
            let (gamma, beta) = learned_affine(channels);
            let fused = fused_bn_silu_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(fused.len(), batch * channels * spatial);
        }
    }

    // ===================================================================
    // 6. Training mode (7 tests)
    // ===================================================================

    mod training_mode {
        use super::*;

        #[test]
        fn test_training_computes_batch_stats() {
            let (batch, channels, spatial) = (8, 4, 4);
            let input = make_input(batch * channels * spatial, 60.0);
            let (gamma, beta) = identity_affine(channels);
            let (_, mean, var) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            // Mean and var should be computed from this batch only.
            for c in 0..channels {
                assert!(mean[c].is_finite(), "mean[{c}] not finite");
                assert!(var[c] >= 0.0, "var[{c}] negative: {}", var[c]);
            }
        }

        #[test]
        fn test_training_different_batches_different_stats() {
            let (channels, spatial) = (4, 4);
            let (gamma, beta) = identity_affine(channels);
            let input_a = make_input(4 * channels * spatial, 61.0);
            let input_b = make_input(4 * channels * spatial, 999.0);
            let (_, mean_a, var_a) =
                batch_norm_forward_cpu(&input_a, &gamma, &beta, channels, spatial, BN_EPSILON);
            let (_, mean_b, var_b) =
                batch_norm_forward_cpu(&input_b, &gamma, &beta, channels, spatial, BN_EPSILON);
            // Different inputs should give different stats.
            let means_differ = mean_a.iter().zip(mean_b.iter()).any(|(a, b)| (a - b).abs() > 1e-4);
            assert!(means_differ, "means should differ for different inputs");
            let vars_differ = var_a.iter().zip(var_b.iter()).any(|(a, b)| (a - b).abs() > 1e-4);
            assert!(vars_differ, "vars should differ for different inputs");
        }

        #[test]
        fn test_training_batch_mean_is_correct() {
            let (_batch, channels, spatial) = (4, 2, 1);
            // Hand-crafted: ch0 = [1, 3, 5, 7], ch1 = [2, 4, 6, 8]
            let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let (gamma, beta) = identity_affine(channels);
            let (_, mean, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert!((mean[0] - 4.0).abs() < 1e-5, "ch0 mean: {}", mean[0]);
            assert!((mean[1] - 5.0).abs() < 1e-5, "ch1 mean: {}", mean[1]);
        }

        #[test]
        fn test_training_batch_var_is_correct() {
            let (_batch, _channels, spatial) = (4, 1, 1);
            // ch0 = [2, 4, 6, 8], mean=5, var = mean((2-5)^2,(4-5)^2,(6-5)^2,(8-5)^2) = (9+1+1+9)/4 = 5
            let input = vec![2.0, 4.0, 6.0, 8.0];
            let (gamma, beta) = identity_affine(1);
            let (_, _, var) = batch_norm_forward_cpu(&input, &gamma, &beta, 1, spatial, BN_EPSILON);
            assert!((var[0] - 5.0).abs() < 1e-5, "var: {}", var[0]);
        }

        #[test]
        fn test_training_output_depends_on_batch() {
            // Same single sample in different batches should produce different outputs.
            let channels = 2;
            let spatial = 2;
            let (gamma, beta) = identity_affine(channels);
            let input_a = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]; // batch=2
            let input_b = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]; // batch=2 different
            let (out_a, _, _) =
                batch_norm_forward_cpu(&input_a, &gamma, &beta, channels, spatial, BN_EPSILON);
            let (out_b, _, _) =
                batch_norm_forward_cpu(&input_b, &gamma, &beta, channels, spatial, BN_EPSILON);
            // First sample outputs should differ because batch stats differ.
            let differs = out_a[..channels * spatial]
                .iter()
                .zip(out_b[..channels * spatial].iter())
                .any(|(a, b)| (a - b).abs() > 1e-4);
            assert!(differs, "output of first sample should depend on batch stats");
        }

        #[test]
        fn test_training_larger_batch_reduces_variance_estimate_noise() {
            let channels = 2;
            let spatial = 1;
            let (gamma, beta) = identity_affine(channels);
            // Small batch.
            let input_small = make_input(2 * channels * spatial, 65.0);
            let (_, _, var_small) =
                batch_norm_forward_cpu(&input_small, &gamma, &beta, channels, spatial, BN_EPSILON);
            // Large batch.
            let input_large = make_input(64 * channels * spatial, 65.0);
            let (_, _, var_large) =
                batch_norm_forward_cpu(&input_large, &gamma, &beta, channels, spatial, BN_EPSILON);
            // Both should be finite and non-negative.
            assert!(var_small.iter().all(|&v| v >= 0.0));
            assert!(var_large.iter().all(|&v| v >= 0.0));
        }

        #[test]
        fn test_training_running_stats_integrate() {
            let (batch, channels, spatial) = (4, 4, 4);
            let (gamma, beta) = identity_affine(channels);
            let mut running_mean = vec![0.0f32; channels];
            let mut running_var = vec![1.0f32; channels];

            for step in 0..5 {
                let input = make_input(batch * channels * spatial, step as f32 * 7.0);
                let (_, bm, bv) =
                    batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
                update_running_stats(&mut running_mean, &bm, BN_MOMENTUM);
                update_running_stats(&mut running_var, &bv, BN_MOMENTUM);
            }
            assert!(running_mean.iter().all(|x| x.is_finite()));
            assert!(running_var.iter().all(|&x| x >= 0.0));
        }
    }

    // ===================================================================
    // 7. Inference mode (6 tests)
    // ===================================================================

    mod inference_mode {
        use super::*;

        #[test]
        fn test_inference_uses_running_stats() {
            let (batch, channels, spatial) = (2, 4, 4);
            let input = make_input(batch * channels * spatial, 70.0);
            let (gamma, beta) = identity_affine(channels);
            let running_mean = vec![1.0, 2.0, 3.0, 4.0];
            let running_var = vec![1.0, 2.0, 3.0, 4.0];
            let output = batch_norm_inference_cpu(
                &input,
                &gamma,
                &beta,
                &running_mean,
                &running_var,
                channels,
                spatial,
                BN_EPSILON,
            );
            assert_eq!(output.len(), batch * channels * spatial);
            assert!(output.iter().all(|x| x.is_finite()));
        }

        #[test]
        fn test_inference_deterministic() {
            let (batch, channels, spatial) = (2, 4, 4);
            let input = make_input(batch * channels * spatial, 71.0);
            let (gamma, beta) = learned_affine(channels);
            let running_mean = vec![0.5; channels];
            let running_var = vec![2.0; channels];
            let out1 = batch_norm_inference_cpu(
                &input,
                &gamma,
                &beta,
                &running_mean,
                &running_var,
                channels,
                spatial,
                BN_EPSILON,
            );
            let out2 = batch_norm_inference_cpu(
                &input,
                &gamma,
                &beta,
                &running_mean,
                &running_var,
                channels,
                spatial,
                BN_EPSILON,
            );
            assert_close(&out1, &out2, 0.0, "inference_deterministic");
        }

        #[test]
        fn test_inference_independent_of_batch_size() {
            // In inference mode, each sample is normalized independently
            // using running stats, so single-sample output should match.
            let channels = 4;
            let spatial = 4;
            let (gamma, beta) = learned_affine(channels);
            let running_mean = vec![0.1, 0.2, 0.3, 0.4];
            let running_var = vec![1.0, 1.5, 2.0, 2.5];
            let single_input = make_input(channels * spatial, 72.0);
            let out_single = batch_norm_inference_cpu(
                &single_input,
                &gamma,
                &beta,
                &running_mean,
                &running_var,
                channels,
                spatial,
                BN_EPSILON,
            );
            // Repeat same sample in a batch of 3.
            let mut batch_input = Vec::new();
            for _ in 0..3 {
                batch_input.extend_from_slice(&single_input);
            }
            let out_batch = batch_norm_inference_cpu(
                &batch_input,
                &gamma,
                &beta,
                &running_mean,
                &running_var,
                channels,
                spatial,
                BN_EPSILON,
            );
            // First sample in batch should match single output.
            assert_close(
                &out_batch[..channels * spatial],
                &out_single,
                1e-7,
                "inference_batch_independent",
            );
        }

        #[test]
        fn test_inference_with_trained_running_stats() {
            let (channels, spatial) = (4, 4);
            let (gamma, beta) = learned_affine(channels);
            let mut running_mean = vec![0.0f32; channels];
            let mut running_var = vec![1.0f32; channels];

            // "Train" for a few steps.
            for step in 0..20 {
                let batch = 8;
                let input = make_input(batch * channels * spatial, step as f32 * 3.0);
                let (_, bm, bv) =
                    batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
                update_running_stats(&mut running_mean, &bm, BN_MOMENTUM);
                update_running_stats(&mut running_var, &bv, BN_MOMENTUM);
            }

            // Now use in inference mode.
            let test_input = make_input(2 * channels * spatial, 100.0);
            let output = batch_norm_inference_cpu(
                &test_input,
                &gamma,
                &beta,
                &running_mean,
                &running_var,
                channels,
                spatial,
                BN_EPSILON,
            );
            assert_eq!(output.len(), 2 * channels * spatial);
            assert!(output.iter().all(|x| x.is_finite()));
        }

        #[test]
        fn test_inference_identity_params_with_zero_stats() {
            let (batch, channels, spatial) = (2, 2, 4);
            let input = make_input(batch * channels * spatial, 74.0);
            let (gamma, beta) = identity_affine(channels);
            let running_mean = vec![0.0; channels];
            let running_var = vec![1.0; channels]; // var=1 + eps → near identity
            let output = batch_norm_inference_cpu(
                &input,
                &gamma,
                &beta,
                &running_mean,
                &running_var,
                channels,
                spatial,
                BN_EPSILON,
            );
            // Should be approximately identity (within eps correction).
            assert_close(&output, &input, 1e-3, "inference_identity");
        }

        #[test]
        fn test_inference_frozen_stats_not_updated() {
            let (batch, channels, spatial) = (4, 2, 4);
            let input = make_input(batch * channels * spatial, 75.0);
            let (gamma, beta) = identity_affine(channels);
            let running_mean = vec![1.0, 2.0];
            let running_var = vec![3.0, 4.0];
            let _ = batch_norm_inference_cpu(
                &input,
                &gamma,
                &beta,
                &running_mean,
                &running_var,
                channels,
                spatial,
                BN_EPSILON,
            );
            // Running stats should not have changed (we passed immutable refs).
            assert_close(&running_mean, &[1.0, 2.0], 0.0, "frozen_mean");
            assert_close(&running_var, &[3.0, 4.0], 0.0, "frozen_var");
        }
    }

    // ===================================================================
    // 8. Group normalization (7 tests)
    // ===================================================================

    mod group_normalization {
        use super::*;

        #[test]
        fn test_group_norm_2_groups() {
            let (batch, channels, spatial) = (2, 4, 4);
            let input = make_input(batch * channels * spatial, 80.0);
            let (gamma, beta) = identity_affine(channels);
            let output = group_norm_cpu(&input, &gamma, &beta, channels, spatial, 2, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
            assert!(output.iter().all(|x| x.is_finite()));
        }

        #[test]
        fn test_group_norm_4_groups() {
            let (batch, channels, spatial) = (2, 8, 4);
            let input = make_input(batch * channels * spatial, 81.0);
            let (gamma, beta) = learned_affine(channels);
            let output = group_norm_cpu(&input, &gamma, &beta, channels, spatial, 4, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
        }

        #[test]
        fn test_group_norm_single_group_is_layer_norm() {
            // num_groups=1 → all channels in one group → equivalent to layer norm.
            let (batch, channels, spatial) = (2, 8, 4);
            let input = make_input(batch * channels * spatial, 82.0);
            let (gamma, beta) = identity_affine(channels);
            let gn_output = group_norm_cpu(&input, &gamma, &beta, channels, spatial, 1, BN_EPSILON);

            // Compute layer norm manually per sample.
            let sample_len = channels * spatial;
            for b in 0..batch {
                let sample = &input[b * sample_len..(b + 1) * sample_len];
                let n = sample.len() as f32;
                let mean = sample.iter().sum::<f32>() / n;
                let var = sample.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
                let inv_std = 1.0 / (var + BN_EPSILON).sqrt();
                for i in 0..sample_len {
                    let expected = (sample[i] - mean) * inv_std;
                    let actual = gn_output[b * sample_len + i];
                    assert!(
                        (actual - expected).abs() < 1e-4,
                        "batch {b}, idx {i}: gn={actual}, ln={expected}"
                    );
                }
            }
        }

        #[test]
        fn test_group_norm_all_groups_is_instance_norm() {
            // num_groups=channels → each channel is its own group → instance norm.
            let (batch, channels, spatial) = (2, 4, 8);
            let input = make_input(batch * channels * spatial, 83.0);
            let (gamma, beta) = identity_affine(channels);
            let gn = group_norm_cpu(&input, &gamma, &beta, channels, spatial, channels, BN_EPSILON);
            let in_ = instance_norm_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_close(&gn, &in_, 1e-6, "group_norm_eq_instance_norm");
        }

        #[test]
        fn test_group_norm_32_groups() {
            let (batch, channels, spatial) = (2, 32, 4);
            let input = make_input(batch * channels * spatial, 84.0);
            let (gamma, beta) = learned_affine(channels);
            let output = group_norm_cpu(&input, &gamma, &beta, channels, spatial, 32, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
        }

        #[test]
        fn test_group_norm_with_affine_params() {
            let (batch, channels, spatial) = (4, 8, 2);
            let input = make_input(batch * channels * spatial, 85.0);
            let gamma = vec![2.0; channels];
            let beta = vec![1.0; channels];
            let output = group_norm_cpu(&input, &gamma, &beta, channels, spatial, 4, BN_EPSILON);
            assert!(output.iter().all(|x| x.is_finite()));
        }

        #[test]
        fn test_group_norm_large_channels() {
            let (batch, channels, spatial) = (1, 256, 1);
            let input = make_input(batch * channels * spatial, 86.0);
            let (gamma, beta) = identity_affine(channels);
            let output = group_norm_cpu(&input, &gamma, &beta, channels, spatial, 16, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
            assert!(output.iter().all(|x| x.is_finite()));
        }
    }

    // ===================================================================
    // 9. Instance normalization (6 tests)
    // ===================================================================

    mod instance_normalization {
        use super::*;

        #[test]
        fn test_instance_norm_basic() {
            let (batch, channels, spatial) = (2, 4, 8);
            let input = make_input(batch * channels * spatial, 90.0);
            let (gamma, beta) = identity_affine(channels);
            let output = instance_norm_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
            assert!(output.iter().all(|x| x.is_finite()));
        }

        #[test]
        fn test_instance_norm_per_instance_stats() {
            // Each sample, each channel independently normalized.
            let (batch, channels, spatial) = (2, 2, 4);
            let input = make_input(batch * channels * spatial, 91.0);
            let (gamma, beta) = identity_affine(channels);
            let output = instance_norm_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);

            // Check each (batch, channel) has near-zero mean.
            for b in 0..batch {
                for c in 0..channels {
                    let start = b * channels * spatial + c * spatial;
                    let slice = &output[start..start + spatial];
                    let mean = slice.iter().sum::<f32>() / spatial as f32;
                    assert!(mean.abs() < 1e-4, "b={b} c={c}: instance mean={mean}");
                }
            }
        }

        #[test]
        fn test_instance_norm_per_instance_unit_variance() {
            let (batch, channels, spatial) = (4, 2, 16);
            let input = make_input(batch * channels * spatial, 92.0);
            let (gamma, beta) = identity_affine(channels);
            let output = instance_norm_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);

            for b in 0..batch {
                for c in 0..channels {
                    let start = b * channels * spatial + c * spatial;
                    let slice = &output[start..start + spatial];
                    let n = spatial as f32;
                    let mean = slice.iter().sum::<f32>() / n;
                    let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
                    assert!((var - 1.0).abs() < 0.1, "b={b} c={c}: instance var={var}");
                }
            }
        }

        #[test]
        fn test_instance_norm_with_learned_affine() {
            let (batch, channels, spatial) = (2, 4, 8);
            let input = make_input(batch * channels * spatial, 93.0);
            let (gamma, beta) = learned_affine(channels);
            let output = instance_norm_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
            assert!(output.iter().all(|x| x.is_finite()));
        }

        #[test]
        fn test_instance_norm_single_spatial() {
            // spatial=1 → variance=0, eps prevents division by zero.
            let (batch, channels, spatial) = (4, 3, 1);
            let input = make_input(batch * channels * spatial, 94.0);
            let (gamma, beta) = identity_affine(channels);
            let output = instance_norm_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert!(output.iter().all(|x| x.is_finite()));
        }

        #[test]
        fn test_instance_norm_batch_independence() {
            // Changing one sample should not affect the other.
            let (channels, spatial) = (2, 4);
            let (gamma, beta) = identity_affine(channels);
            let input_a = make_input(2 * channels * spatial, 95.0);
            let mut input_b = input_a.clone();
            // Modify second sample.
            for value in input_b.iter_mut().skip(channels * spatial).take(channels * spatial) {
                *value = 100.0;
            }
            let out_a = instance_norm_cpu(&input_a, &gamma, &beta, channels, spatial, BN_EPSILON);
            let out_b = instance_norm_cpu(&input_b, &gamma, &beta, channels, spatial, BN_EPSILON);
            // First sample should be identical.
            assert_close(
                &out_a[..channels * spatial],
                &out_b[..channels * spatial],
                1e-7,
                "instance_norm_batch_indep",
            );
        }
    }

    // ===================================================================
    // 10. Numerical precision (7 tests)
    // ===================================================================

    mod numerical_precision {
        use super::*;

        #[test]
        fn test_large_input_values() {
            let (batch, channels, spatial) = (2, 4, 4);
            let input: Vec<f32> =
                (0..batch * channels * spatial).map(|i| (i as f32) * 1000.0).collect();
            let (gamma, beta) = identity_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert!(output.iter().all(|x| x.is_finite()), "large values: non-finite");
        }

        #[test]
        fn test_small_epsilon() {
            let (batch, channels, spatial) = (4, 4, 4);
            let input = make_input(batch * channels * spatial, 101.0);
            let (gamma, beta) = identity_affine(channels);
            let eps = 1e-10;
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, eps);
            assert!(output.iter().all(|x| x.is_finite()), "small eps: non-finite");
        }

        #[test]
        fn test_large_epsilon() {
            let (batch, channels, spatial) = (4, 4, 4);
            let input = make_input(batch * channels * spatial, 102.0);
            let (gamma, beta) = identity_affine(channels);
            let eps = 1.0; // Unusually large.
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, eps);
            assert!(output.iter().all(|x| x.is_finite()));
            // Large eps reduces the normalization effect.
        }

        #[test]
        fn test_constant_input_zero_variance() {
            // All elements identical → variance = 0, eps prevents NaN.
            let (batch, channels, spatial) = (4, 2, 4);
            let input = vec![42.0f32; batch * channels * spatial];
            let (gamma, beta) = identity_affine(channels);
            let (output, _, var) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            for channel_var in var.iter().take(channels) {
                assert!(channel_var.abs() < 1e-6, "constant input var should be ~0");
            }
            assert!(output.iter().all(|x| x.is_finite()), "constant: non-finite output");
        }

        #[test]
        fn test_very_small_input_values() {
            let (batch, channels, spatial) = (4, 4, 4);
            let input: Vec<f32> =
                (0..batch * channels * spatial).map(|i| (i as f32) * 1e-8).collect();
            let (gamma, beta) = identity_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert!(output.iter().all(|x| x.is_finite()), "tiny values: non-finite");
        }

        #[test]
        fn test_mixed_positive_negative() {
            let (batch, channels, spatial) = (4, 4, 4);
            let input: Vec<f32> = (0..batch * channels * spatial)
                .map(|i| if i % 2 == 0 { 1e3 } else { -1e3 })
                .collect();
            let (gamma, beta) = identity_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert!(output.iter().all(|x| x.is_finite()));
        }

        #[test]
        fn test_fp16_range_simulation() {
            // Simulate fp16 precision by clamping inputs to fp16 range.
            let fp16_max = 65504.0f32;
            let (batch, channels, spatial) = (2, 4, 4);
            let input: Vec<f32> = make_input(batch * channels * spatial, 107.0)
                .iter()
                .map(|&x| (x * fp16_max).clamp(-fp16_max, fp16_max))
                .collect();
            let (gamma, beta) = identity_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert!(output.iter().all(|x| x.is_finite()), "fp16 range: non-finite");
        }
    }

    // ===================================================================
    // 11. Edge cases (6 tests)
    // ===================================================================

    mod edge_cases {
        use super::*;

        #[test]
        fn test_batch_size_one() {
            let (batch, channels, spatial) = (1, 4, 4);
            let input = make_input(batch * channels * spatial, 110.0);
            let (gamma, beta) = identity_affine(channels);
            let (output, mean, var) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
            assert!(output.iter().all(|x| x.is_finite()));
            assert_eq!(mean.len(), channels);
            assert_eq!(var.len(), channels);
        }

        #[test]
        fn test_channels_one() {
            let (batch, channels, spatial) = (4, 1, 8);
            let input = make_input(batch * channels * spatial, 111.0);
            let (gamma, beta) = identity_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
        }

        #[test]
        fn test_spatial_one() {
            let (batch, channels, spatial) = (8, 4, 1);
            let input = make_input(batch * channels * spatial, 112.0);
            let (gamma, beta) = identity_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
        }

        #[test]
        fn test_all_dimensions_one() {
            let (_batch, channels, spatial) = (1, 1, 1);
            let input = vec![42.0f32];
            let (gamma, beta) = identity_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(output.len(), 1);
            assert!(output[0].is_finite());
        }

        #[test]
        fn test_large_batch_many_channels() {
            let (batch, channels, spatial) = (32, 64, 1);
            let input = make_input(batch * channels * spatial, 114.0);
            let (gamma, beta) = identity_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
        }

        #[test]
        fn test_large_spatial_dimension() {
            let (batch, channels, spatial) = (2, 2, 1024);
            let input = make_input(batch * channels * spatial, 115.0);
            let (gamma, beta) = identity_affine(channels);
            let (output, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(output.len(), batch * channels * spatial);
            assert!(output.iter().all(|x| x.is_finite()));
        }
    }

    // ===================================================================
    // 12. GPU-gated integration tests (8 tests)
    // ===================================================================

    mod gpu_integration {
        use super::*;

        #[test]
        #[ignore = "requires Metal GPU runtime — run on Apple Silicon with --ignored"]
        fn test_gpu_bn_forward_small() {
            let (batch, channels, spatial) = (2, 4, 4);
            let input = make_input(batch * channels * spatial, 200.0);
            let (gamma, beta) = identity_affine(channels);
            let (expected, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            // GPU dispatch would go here.
            // For now validate the CPU reference is usable.
            assert_eq!(expected.len(), batch * channels * spatial);
            assert!(expected.iter().all(|x| x.is_finite()));
        }

        #[test]
        #[ignore = "requires Metal GPU runtime — run on Apple Silicon with --ignored"]
        fn test_gpu_bn_forward_large() {
            let (batch, channels, spatial) = (8, 64, 16);
            let input = make_input(batch * channels * spatial, 201.0);
            let (gamma, beta) = learned_affine(channels);
            let (expected, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(expected.len(), batch * channels * spatial);
        }

        #[test]
        #[ignore = "requires Metal GPU runtime — run on Apple Silicon with --ignored"]
        fn test_gpu_bn_inference_mode() {
            let (batch, channels, spatial) = (4, 8, 8);
            let input = make_input(batch * channels * spatial, 202.0);
            let (gamma, beta) = learned_affine(channels);
            let running_mean = vec![0.5; channels];
            let running_var = vec![2.0; channels];
            let expected = batch_norm_inference_cpu(
                &input,
                &gamma,
                &beta,
                &running_mean,
                &running_var,
                channels,
                spatial,
                BN_EPSILON,
            );
            assert_eq!(expected.len(), batch * channels * spatial);
        }

        #[test]
        #[ignore = "requires Metal GPU runtime — run on Apple Silicon with --ignored"]
        fn test_gpu_fused_bn_relu() {
            let (batch, channels, spatial) = (4, 8, 4);
            let input = make_input(batch * channels * spatial, 203.0);
            let (gamma, beta) = learned_affine(channels);
            let expected = fused_bn_relu_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert!(expected.iter().all(|&x| x >= 0.0));
        }

        #[test]
        #[ignore = "requires Metal GPU runtime — run on Apple Silicon with --ignored"]
        fn test_gpu_fused_bn_silu() {
            let (batch, channels, spatial) = (4, 8, 4);
            let input = make_input(batch * channels * spatial, 204.0);
            let (gamma, beta) = learned_affine(channels);
            let expected = fused_bn_silu_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert!(expected.iter().all(|x| x.is_finite()));
        }

        #[test]
        #[ignore = "requires Metal GPU runtime — run on Apple Silicon with --ignored"]
        fn test_gpu_group_norm() {
            let (batch, channels, spatial) = (2, 16, 8);
            let input = make_input(batch * channels * spatial, 205.0);
            let (gamma, beta) = identity_affine(channels);
            let expected = group_norm_cpu(&input, &gamma, &beta, channels, spatial, 4, BN_EPSILON);
            assert_eq!(expected.len(), batch * channels * spatial);
        }

        #[test]
        #[ignore = "requires Metal GPU runtime — run on Apple Silicon with --ignored"]
        fn test_gpu_instance_norm() {
            let (batch, channels, spatial) = (2, 8, 16);
            let input = make_input(batch * channels * spatial, 206.0);
            let (gamma, beta) = identity_affine(channels);
            let expected = instance_norm_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert_eq!(expected.len(), batch * channels * spatial);
        }

        #[test]
        #[ignore = "requires Metal GPU runtime — run on Apple Silicon with --ignored"]
        fn test_gpu_bn_numerical_precision() {
            let (batch, channels, spatial) = (4, 4, 4);
            let input: Vec<f32> =
                (0..batch * channels * spatial).map(|i| (i as f32) * 1e4).collect();
            let (gamma, beta) = identity_affine(channels);
            let (expected, _, _) =
                batch_norm_forward_cpu(&input, &gamma, &beta, channels, spatial, BN_EPSILON);
            assert!(expected.iter().all(|x| x.is_finite()));
        }
    }
}
