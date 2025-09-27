# Cross-Language Validation Framework

This directory contains the enhanced cross-language validation framework for the BitNet Python to Rust migration. The framework provides comprehensive testing capabilities to ensure functional parity and performance validation between implementations.

## Features

### 1. Python Subprocess Runner
- Executes original BitNet.cpp inference through subprocess calls
- Captures and parses output for comparison
- Handles timeouts and error conditions gracefully
- Supports all original CLI parameters and configurations

### 2. Token-Level Comparison Utilities
- **Configurable Tolerance**: Adjustable thresholds for token sequence matching
- **Detailed Analysis**: Position-specific difference reporting
- **Logits Comparison**: Numerical comparison of model outputs with configurable precision
- **Match Ratio Calculation**: Percentage-based similarity metrics

### 3. Performance Comparison Tools
- **Regression Detection**: Automatically detects performance regressions exceeding 5% threshold
- **Speedup Analysis**: Calculates and reports performance improvements
- **Baseline Tracking**: Maintains historical performance baselines
- **Statistical Analysis**: Mean, median, min, max performance metrics

### 4. Test Data Generators
- **Edge Cases**: Empty inputs, boundary values, extreme sizes
- **Stress Tests**: High-load scenarios, batch processing, long sequences
- **Numerical Edge Cases**: Zero weights, sparse matrices, extreme values
- **Model Configurations**: Minimal, large, and unusual dimension models

## Usage

### Basic Usage

```bash
# Run basic validation tests
python run_validation.py --test-type basic

# Run all tests including edge cases and stress tests
python run_validation.py --test-type all

# Run with custom Rust binary
python run_validation.py --rust-binary /path/to/bitnet-cli
```

### Advanced Configuration

```bash
# Custom numerical tolerances
python run_validation.py \
    --rtol 1e-5 \
    --atol 1e-6 \
    --token-match-ratio 0.98 \
    --max-token-diff 2

# Performance regression testing
python run_validation.py \
    --regression-threshold 3.0 \
    --fail-on-regression

# Filter specific test categories
python run_validation.py \
    --categories edge_case stress_test \
    --patterns quantization
```

### Programmatic Usage

```python
from framework import (
    CrossLanguageValidator,
    TestCaseGenerator,
    create_default_config
)

# Create validator
config = create_default_config()
validator = CrossLanguageValidator(config)

# Generate and run tests
test_cases = TestCaseGenerator.generate_all_tests()
results = validator.validate_test_suite(test_cases)

# Generate report
validator.generate_report(results, Path("report.json"))
```

## Configuration Options

### Numerical Tolerances

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rtol` | 1e-4 | Relative tolerance for numerical comparisons |
| `atol` | 1e-5 | Absolute tolerance for numerical comparisons |
| `min_token_match_ratio` | 0.95 | Minimum ratio of matching tokens |
| `max_token_differences` | 5 | Maximum number of token differences allowed |
| `logits_rtol` | 1e-3 | Relative tolerance for logits comparison |
| `logits_atol` | 1e-4 | Absolute tolerance for logits comparison |
| `min_logits_close_ratio` | 0.99 | Minimum ratio of close logits values |
| `max_logits_abs_diff` | 0.1 | Maximum absolute difference in logits |

### Performance Tolerances

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_regression_percent` | 5.0 | Maximum acceptable performance regression (%) |

## Test Categories

### Basic Tests
- **Model Forward**: Basic transformer forward pass validation
- **Quantization**: Weight quantization accuracy testing
- **Inference**: End-to-end inference validation

### Edge Case Tests
- **Empty Inputs**: Zero-length sequences and empty prompts
- **Boundary Values**: Min/max token values and extreme dimensions
- **Large Inputs**: Very long sequences and large model configurations
- **Special Characters**: Unicode, newlines, and control characters

### Stress Tests
- **High Load**: Long inference runs with large token counts
- **Batch Processing**: Multiple concurrent inference requests
- **Memory Pressure**: Large model configurations and long contexts

## Output Format

The framework generates comprehensive JSON reports with:

```json
{
  "summary": {
    "total_tests": 45,
    "passed_tests": 42,
    "failed_tests": 3,
    "python_failures": 1,
    "rust_failures": 1,
    "output_mismatches": 1
  },
  "performance_summary": {
    "average_speedup": 2.3,
    "max_speedup": 4.1,
    "min_speedup": 1.2
  },
  "detailed_results": [...],
  "environment_info": {...}
}
```

## Error Handling

The framework provides robust error handling for:
- **Subprocess Failures**: Timeout, crash, or invalid output
- **Numerical Issues**: NaN, infinity, or precision loss
- **Memory Constraints**: Large model or input size limitations
- **Environment Issues**: Missing dependencies or invalid paths

## Integration with CI/CD

The framework is designed for CI/CD integration:

```yaml
# Example GitHub Actions step
- name: Run Cross-Language Validation
  run: |
    python tests/cross_validation/run_validation.py \
      --test-type all \
      --fail-on-regression \
      --output validation_report.json
```

## Testing the Framework

Run the framework self-tests:

```bash
python tests/cross_validation/test_framework.py
```

This validates all framework components and ensures proper functionality.

## Files

- `framework.py`: Core validation framework implementation
- `run_validation.py`: Command-line interface for running validations
- `test_framework.py`: Self-tests for the framework components
- `README.md`: This documentation file

## Requirements

- Python 3.8+
- NumPy
- PyTorch (for Python baseline tests)
- Access to original BitNet.cpp implementation
- Rust BitNet binary (for cross-validation)

## Contributing

When adding new test cases or validation logic:

1. Add test cases to the appropriate generator method
2. Update tolerance configurations if needed
3. Add documentation for new parameters
4. Run self-tests to ensure framework integrity
5. Update this README with new features

## Troubleshooting

### Common Issues

1. **"Python environment validation failed"**
   - Ensure PyTorch and NumPy are installed
   - Check that BitNet Python modules are accessible

2. **"Rust binary not found"**
   - Build the Rust project first: `cargo build --release`
   - Specify binary path with `--rust-binary`

3. **"Subprocess timeout"**
   - Increase timeout with `--timeout`
   - Check for infinite loops in test cases

4. **"Numerical comparison failed"**
   - Adjust tolerances with `--rtol` and `--atol`
   - Check for NaN or infinity values in outputs

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
python run_validation.py --verbose --test-type basic
```

This provides detailed information about test execution, comparisons, and any failures.