# Simulation: `i2s_quantization_stability_test` in `property_tests.rs` uses a simplified `test_data` generation

The `i2s_quantization_stability_test` in `crates/bitnet-quantization/src/property_tests.rs` uses a simplified `test_data` generation. It uses `((i as f32).sin() * scale_factor).clamp(-2.0, 2.0)` to generate test data. This might not cover all possible input distributions. This is a form of simulation.

**File:** `crates/bitnet-quantization/src/property_tests.rs`

**Function:** `i2s_quantization_stability_test`

**Code:**
```rust
            #[test]
            fn i2s_quantization_stability_test(
                block_count in 1usize..100,
                scale_factor in 0.1f32..10.0f32
            ) {
                let layout = I2SLayout::default();
                let total_elems = block_count * layout.block_size;

                // Create test data with known patterns
                let mut test_data = vec![0.0f32; total_elems];
                for (i, elem) in test_data.iter_mut().enumerate().take(total_elems) {
                    *elem = ((i as f32).sin() * scale_factor).clamp(-2.0, 2.0);
                }

                // ...
            }
```

## Proposed Fix

The `i2s_quantization_stability_test` should use a more comprehensive `test_data` generation that covers a wider range of input distributions. This would involve using different statistical distributions (e.g., uniform, normal, exponential, bimodal) and adversarial patterns (e.g., high-frequency, checkerboard, step function) to generate test data.

### Example Implementation

```rust
            #[test]
            fn i2s_quantization_stability_test(
                block_count in 1usize..100,
                scale_factor in 0.1f32..10.0f32,
                distribution_type in prop_oneof![
                    Just("uniform"),
                    Just("normal"),
                    Just("exponential"),
                    Just("bimodal"),
                    Just("sparse"),
                    Just("neural_weights"),
                    Just("adversarial"),
                ]
            ) {
                let layout = I2SLayout::default();
                let total_elems = block_count * layout.block_size;

                let test_data = match distribution_type {
                    "uniform" => generate_uniform_data(total_elems, -1.0, 1.0),
                    "normal" => generate_normal_data(total_elems, 0.0, 0.3),
                    "exponential" => generate_exponential_data(total_elems, 1.0),
                    "bimodal" => generate_bimodal_data(total_elems),
                    "sparse" => generate_sparse_data(total_elems, 0.1),
                    "neural_weights" => generate_neural_weight_distribution(total_elems),
                    "adversarial" => generate_adversarial_pattern(total_elems),
                    _ => unreachable!(),
                };

                // ...
            }
```
