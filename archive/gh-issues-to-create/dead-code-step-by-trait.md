# Dead code: `StepBy` trait and implementation in `property_based_tests.rs` are never used

The `StepBy` trait and its implementation for `std::ops::RangeInclusive<f32>` in `crates/bitnet-quantization/src/property_based_tests.rs` are defined but not used. This is a form of dead code.

**File:** `crates/bitnet-quantization/src/property_based_tests.rs`

**Trait:** `StepBy`

**Code:**
```rust
    // Trait for step_by iterator (simple implementation)
    trait StepBy {
        fn step_by(self, step: usize) -> Vec<f32>;
    }

    impl StepBy for std::ops::RangeInclusive<f32> {
        fn step_by(self, step: usize) -> Vec<f32> {
            let mut result = Vec::new();
            let start = *self.start();
            let end = *self.end();
            let num_steps = step;

            for i in 0..num_steps {
                let t = i as f32 / (num_steps - 1) as f32;
                let value = start + t * (end - start);
                if value <= end {
                    result.push(value);
                }
            }

            result
        }
    }
```

## Proposed Fix

If the `StepBy` trait and its implementation are not intended to be used, they should be removed to reduce the size of the codebase and improve maintainability. If they are intended to be used, they should be integrated into the property-based tests.

### Example Integration

```rust
        let test_sequences = vec![
            (-1.0..=1.0).step_by(16),
            (-0.5..=0.5).step_by(8),
            (0.0..=1.0).step_by(10),
        ];
```
