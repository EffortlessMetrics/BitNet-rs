# BitNet Python Baseline Test Suite

This comprehensive test suite validates the Python implementation of BitNet.cpp to establish a baseline for the Rust migration. The tests ensure functional correctness, numerical precision, and performance characteristics that must be preserved during the migration.

## Overview

The test suite is designed to:

1. **Validate Core Functionality**: Ensure all BitNet components work correctly
2. **Establish Numerical Baselines**: Create reference outputs for cross-validation
3. **Performance Benchmarking**: Measure performance characteristics for regression detection
4. **Property-Based Testing**: Verify mathematical properties and edge cases
5. **Cross-Platform Validation**: Test across different hardware configurations

## Test Structure

```
tests/python_baseline/
├── conftest.py                    # Pytest configuration and fixtures
├── test_model_loading.py          # Model initialization and loading tests
├── test_quantization.py           # Quantization algorithm tests
├── test_inference.py              # Inference engine tests
├── test_performance_benchmarks.py # Performance and regression tests
├── test_property_based.py         # Property-based tests with Hypothesis
├── test_fixtures.py               # Test fixtures and known-good outputs
├── run_tests.py                   # Test runner script
├── requirements.txt               # Python dependencies
├── pytest.ini                     # Pytest configuration
└── README.md                      # This file
```

## Test Categories

### 1. Unit Tests (`test_model_loading.py`, `test_quantization.py`, `test_inference.py`)

**Purpose**: Validate individual components and their interactions.

**Coverage**:
- Model configuration and initialization
- Layer structure and parameter counts
- Forward pass correctness
- Quantization algorithms (I2_S, TL1, TL2)
- Inference engines (CPU/GPU)
- Memory management and caching

**Key Features**:
- Deterministic testing with fixed seeds
- Cross-platform compatibility
- Memory usage validation
- Numerical stability checks

### 2. Property-Based Tests (`test_property_based.py`)

**Purpose**: Verify mathematical properties and handle edge cases using Hypothesis.

**Properties Tested**:
- Quantization idempotency
- Sign preservation
- Range constraints
- Batch consistency
- Numerical stability

**Benefits**:
- Discovers edge cases automatically
- Tests with random but reproducible inputs
- Validates mathematical invariants
- Stress tests with extreme values

### 3. Performance Benchmarks (`test_performance_benchmarks.py`)

**Purpose**: Establish performance baselines and detect regressions.

**Metrics Tracked**:
- Forward pass throughput (tokens/second)
- Single token generation latency
- Memory bandwidth utilization
- Quantization speed
- Model loading time

**Features**:
- Statistical analysis with multiple runs
- Performance regression detection
- Memory usage profiling
- Cross-device comparisons

### 4. Test Fixtures (`test_fixtures.py`)

**Purpose**: Generate and validate known-good outputs for cross-validation.

**Components**:
- Model output fixtures
- Quantization reference data
- Inference test cases
- Numerical precision requirements

**Usage**:
- Cross-language validation
- Regression testing
- Reference implementations

## Running Tests

### Prerequisites

```bash
# Install dependencies
pip install -r tests/python_baseline/requirements.txt

# Ensure BitNet modules are in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/gpu:$(pwd)/utils"
```

### Quick Start

```bash
# Run all tests
python tests/python_baseline/run_tests.py --all

# Run specific test suites
python tests/python_baseline/run_tests.py --unit --property

# Run with coverage
python tests/python_baseline/run_tests.py --unit --coverage

# Run performance tests
python tests/python_baseline/run_tests.py --performance

# Run GPU tests (requires CUDA)
python tests/python_baseline/run_tests.py --gpu
```

### Advanced Usage

```bash
# Run tests in parallel
python tests/python_baseline/run_tests.py --all --parallel 4

# Generate detailed report
python tests/python_baseline/run_tests.py --all --report results.json

# Run only fast tests
pytest tests/python_baseline/ -m "not slow"

# Run specific test file
pytest tests/python_baseline/test_quantization.py -v

# Run with specific markers
pytest tests/python_baseline/ -m "quantization and not gpu"
```

## Test Markers

Tests are organized using pytest markers:

- `slow`: Long-running tests (performance, large models)
- `gpu`: Tests requiring CUDA
- `quantization`: Quantization-related tests
- `inference`: Inference engine tests
- `conversion`: Model conversion tests
- `property`: Property-based tests
- `benchmark`: Performance benchmarks
- `integration`: Integration tests
- `unit`: Unit tests

## Configuration

### Numerical Tolerances

Default tolerances for numerical comparisons:

```python
NUMERICAL_TOLERANCE = {
    "rtol": 1e-4,  # Relative tolerance
    "atol": 1e-5,  # Absolute tolerance
}
```

### Performance Thresholds

Performance regression detection:

```python
PERFORMANCE_TOLERANCE = {
    "max_regression_percent": 5.0,  # Maximum allowed regression
}
```

### Test Data

Standard test configurations:

```python
MODEL_SHAPES = [
    (2560, 2560),   # Square matrix
    (3840, 2560),   # Rectangular
    (13824, 2560),  # Large input
    (2560, 6912),   # Large output
]

BATCH_SIZES = [1, 4, 8]
SEQUENCE_LENGTHS = [64, 128, 512]
```

## Cross-Validation Framework

The test suite includes a framework for cross-validation between Python and Rust implementations:

### 1. Baseline Generation

```python
# Generate baseline outputs
python tests/python_baseline/run_tests.py --fixtures

# This creates reference data in tests/fixtures/
```

### 2. Cross-Language Validation

```python
# Compare Rust implementation against Python baseline
from tests.python_baseline.test_fixtures import TestFixtureGenerator

generator = TestFixtureGenerator("tests/fixtures")
baseline = generator.load_fixture("model_baseline")

# Run Rust implementation with same inputs
# Compare outputs within tolerance
```

### 3. Regression Detection

```python
# Detect performance regressions
python tests/python_baseline/run_tests.py --performance --report current.json

# Compare against baseline
python scripts/compare_performance.py baseline.json current.json
```

## Continuous Integration

### GitHub Actions Integration

```yaml
name: Python Baseline Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r tests/python_baseline/requirements.txt
    
    - name: Run tests
      run: |
        python tests/python_baseline/run_tests.py --all --coverage
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: tests/coverage.xml
```

### Performance Monitoring

```yaml
- name: Performance Regression Check
  run: |
    python tests/python_baseline/run_tests.py --performance --report current.json
    python scripts/check_regression.py baseline.json current.json
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure Python path includes BitNet modules
   export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/gpu:$(pwd)/utils"
   ```

2. **CUDA Tests Failing**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Skip GPU tests if CUDA unavailable
   pytest tests/python_baseline/ -m "not gpu"
   ```

3. **Memory Issues**
   ```bash
   # Reduce test parallelism
   python tests/python_baseline/run_tests.py --all --parallel 1
   
   # Skip memory-intensive tests
   pytest tests/python_baseline/ -m "not slow"
   ```

4. **Numerical Precision Issues**
   ```python
   # Adjust tolerances in conftest.py
   NUMERICAL_TOLERANCE = {
       "rtol": 1e-3,  # Looser tolerance
       "atol": 1e-4,
   }
   ```

### Debug Mode

```bash
# Run with verbose output and no capture
pytest tests/python_baseline/ -v -s --tb=long

# Run single test with debugging
pytest tests/python_baseline/test_quantization.py::TestBitLinearQuantization::test_quant_input_basic -v -s
```

### Performance Profiling

```bash
# Profile memory usage
python -m memory_profiler tests/python_baseline/test_performance_benchmarks.py

# Profile CPU usage
python -m cProfile -o profile.stats tests/python_baseline/run_tests.py --performance
```

## Contributing

### Adding New Tests

1. **Follow naming conventions**: `test_*.py` files, `Test*` classes, `test_*` methods
2. **Use appropriate markers**: Add `@pytest.mark.slow` for long tests
3. **Include docstrings**: Describe test purpose and expected behavior
4. **Handle edge cases**: Test with empty inputs, extreme values, etc.
5. **Validate outputs**: Check shapes, types, and numerical properties

### Test Data Guidelines

1. **Use fixed seeds**: Ensure reproducible results
2. **Test multiple scales**: Small and large inputs
3. **Include edge cases**: Zero, infinity, NaN values
4. **Document expectations**: Clear success criteria

### Performance Test Guidelines

1. **Warmup runs**: Exclude from timing measurements
2. **Multiple iterations**: Statistical significance
3. **Resource cleanup**: Prevent memory leaks
4. **Platform awareness**: Account for hardware differences

## Integration with Rust Migration

This test suite serves as the foundation for validating the Rust migration:

### Phase 1: Baseline Establishment
- Run complete test suite on Python implementation
- Generate reference outputs and performance metrics
- Document numerical precision requirements

### Phase 2: Cross-Validation
- Implement equivalent tests in Rust
- Compare outputs against Python baseline
- Validate numerical accuracy within tolerances

### Phase 3: Performance Validation
- Compare Rust performance against Python baseline
- Ensure performance improvements meet targets
- Validate memory usage and efficiency gains

### Phase 4: Regression Prevention
- Continuous testing of both implementations
- Automated performance monitoring
- Alert on regressions or compatibility issues

## Numerical Precision Requirements

### Quantization Accuracy
- **Sign preservation**: 100% for non-zero values
- **Zero preservation**: 100% for exact zeros
- **Range validation**: All outputs in {-1, 0, 1}

### Inference Accuracy
- **Logits comparison**: rtol=1e-4, atol=1e-5
- **Token generation**: Identical for greedy sampling
- **Probability distributions**: Sum to 1.0 within 1e-6

### Performance Targets
- **Throughput**: Within 5% of baseline
- **Latency**: Within 5% of baseline
- **Memory usage**: No significant increase

## Documentation and Reporting

### Test Reports

Generated reports include:
- Test execution summary
- Performance metrics
- Coverage statistics
- Failure analysis
- Regression detection

### Baseline Documentation

Reference documentation includes:
- Model configurations tested
- Input/output specifications
- Numerical precision requirements
- Performance characteristics
- Known limitations and edge cases

This comprehensive test suite ensures that the Rust migration maintains full compatibility with the Python implementation while achieving the performance and safety benefits of Rust.