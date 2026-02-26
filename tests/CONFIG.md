# BitNet-rs Testing Framework Configuration

This document describes the configuration system for the BitNet-rs testing framework.

## Overview

The testing framework uses a hierarchical configuration system that supports:

- **File-based configuration**: TOML configuration files
- **Environment variable overrides**: Environment variables take precedence
- **Programmatic configuration**: Create configurations in code
- **Validation**: Comprehensive validation with detailed error reporting

## Configuration Sources

The configuration system loads settings from multiple sources in order of precedence:

1. **Environment variables** (highest precedence)
2. **Configuration files** (in order):
   - File specified by `BITNET_TEST_CONFIG` environment variable
   - `bitnet-test.toml` in current directory
   - `tests/config.toml` in current directory
   - `.bitnet/test-config.toml` in current directory
3. **Default values** (lowest precedence)

## Configuration File Format

Configuration files use TOML format. See `tests/config.example.toml` for a complete example.

### Core Settings

```toml
# Maximum number of tests to run in parallel
max_parallel_tests = 4

# Timeout for individual tests (in seconds or duration format)
test_timeout = "300s"

# Directory for test cache and temporary files
cache_dir = "tests/cache"

# Logging level: trace, debug, info, warn, error
log_level = "info"

# Minimum code coverage threshold (0.0 to 1.0)
coverage_threshold = 0.9
```

### Fixture Configuration

```toml
[fixtures]
# Automatically download missing fixtures
auto_download = true

# Maximum cache size in bytes
max_cache_size = 10737418240  # 10 GB

# How often to clean up old fixtures
cleanup_interval = "86400s"  # 24 hours

# Timeout for fixture downloads
download_timeout = "300s"  # 5 minutes

# Base URL for fixture downloads (optional)
base_url = "https://huggingface.co/bitnet-models"

# Custom fixture definitions
[[fixtures.custom_fixtures]]
name = "test-model-small"
url = "https://example.com/models/test-small.gguf"
checksum = "abc123def456789"
description = "Small test model for basic validation"
```

### Cross-Validation Configuration

```toml
[crossval]
# Enable cross-validation testing against C++ implementation
enabled = false

# Path to C++ BitNet binary (required if enabled)
cpp_binary_path = "/path/to/bitnet-cpp"

# Test cases to run for cross-validation
test_cases = [
    "basic_inference",
    "tokenization",
    "model_loading"
]

# Enable performance and accuracy comparisons
performance_comparison = true
accuracy_comparison = true

[crossval.tolerance]
# Minimum token accuracy (0.0 to 1.0)
min_token_accuracy = 0.999999

# Maximum probability divergence
max_probability_divergence = 0.000001

# Maximum acceptable performance regression (ratio)
max_performance_regression = 0.1

# Numerical tolerance for floating-point comparisons
numerical_tolerance = 0.000001
```

### Reporting Configuration

```toml
[reporting]
# Output directory for test reports
output_dir = "test-reports"

# Report formats to generate
formats = ["Html", "Json", "Junit"]

# Include test artifacts in reports
include_artifacts = true

# Generate code coverage reports
generate_coverage = true

# Generate performance reports
generate_performance = true

# Upload reports to external services
upload_reports = false
```

## Environment Variables

All configuration options can be overridden using environment variables:

### Core Settings
- `BITNET_TEST_CONFIG`: Path to configuration file
- `BITNET_TEST_PARALLEL`: Maximum parallel tests
- `BITNET_TEST_TIMEOUT`: Test timeout in seconds
- `BITNET_TEST_CACHE_DIR`: Cache directory path
- `BITNET_TEST_LOG_LEVEL`: Logging level
- `BITNET_TEST_COVERAGE_THRESHOLD`: Coverage threshold (0.0-1.0)

### Fixture Settings
- `BITNET_TEST_AUTO_DOWNLOAD`: Enable auto-download (true/false)
- `BITNET_TEST_MAX_CACHE_SIZE`: Maximum cache size in bytes
- `BITNET_TEST_FIXTURE_BASE_URL`: Base URL for fixture downloads

### Cross-Validation Settings
- `BITNET_TEST_CROSSVAL_ENABLED`: Enable cross-validation (true/false)
- `BITNET_TEST_CPP_BINARY`: Path to C++ binary

### Reporting Settings
- `BITNET_TEST_REPORT_DIR`: Report output directory
- `BITNET_TEST_REPORT_FORMATS`: Comma-separated list of formats
- `BITNET_TEST_GENERATE_COVERAGE`: Generate coverage reports (true/false)

## Predefined Configurations

The framework provides several predefined configurations:

### Default Configuration
```rust
use bitnet_tests::TestConfig;
let config = TestConfig::default();
```

### CI Configuration
Optimized for continuous integration environments:
```rust
use bitnet_tests::ci_config;
let config = ci_config();
```

Features:
- Reduced parallel tests to avoid resource contention
- Shorter timeouts for faster feedback
- Verbose logging for debugging
- Always generates coverage reports
- Includes JUnit format for CI integration

### Development Configuration
Optimized for local development:
```rust
use bitnet_tests::dev_config;
let config = dev_config();
```

Features:
- Uses all available CPU cores
- Longer timeouts for debugging
- Less verbose logging
- Skips coverage by default for speed
- Only generates HTML reports

### Minimal Configuration
Optimized for minimal resource usage:
```rust
use bitnet_tests::minimal_config;
let config = minimal_config();
```

Features:
- Sequential test execution
- Short timeouts
- Minimal logging
- No coverage or performance reporting
- Disables cross-validation and auto-download

## Configuration Validation

The framework includes comprehensive configuration validation:

### Using the CLI Validator
```bash
# Validate default configuration
cargo run --bin validate_config

# Validate specific configuration file
cargo run --bin validate_config -- path/to/config.toml
```

### Programmatic Validation
```rust
use bitnet_tests::{ConfigValidator, load_test_config};

// Validate current configuration
let validator = ConfigValidator::new()?;
let result = validator.validate();

if !result.is_valid() {
    for error in &result.errors {
        eprintln!("Error in {}: {}", error.field, error.message);
    }
}
```

### Validation Checks

The validator performs these checks:

**Errors (prevent execution):**
- Invalid parallel test count (must be > 0, <= 100)
- Invalid timeout values
- Invalid coverage threshold (must be 0.0-1.0)
- Invalid log level
- Missing cache/report directory parents
- Invalid custom fixture URLs or checksums
- Missing C++ binary when cross-validation enabled
- Invalid tolerance values

**Warnings (may cause issues):**
- Parallel tests much higher than CPU cores
- Very short or long timeouts
- Cache size exceeds available disk space
- Auto-download enabled without internet connectivity
- Insecure HTTP URLs for fixtures
- Missing coverage tools when coverage enabled

**Info (optimization suggestions):**
- Sequential execution when parallel possible
- Very strict tolerance settings
- Many report formats selected

## Advanced Usage

### Merging Configurations
```rust
use bitnet_tests::{TestConfig, merge_configs};

let base = TestConfig::default();
let override_config = /* load from somewhere */;
let merged = merge_configs(base, override_config);
```

### Saving Configurations
```rust
use bitnet_tests::{TestConfig, save_config_to_file};
use std::path::PathBuf;

let config = TestConfig::default();
save_config_to_file(&config, &PathBuf::from("my-config.toml"))?;
```

### Getting Configuration Values
```rust
use bitnet_tests::{TestConfig, get_config_value};

let config = TestConfig::default();
let parallel_tests = get_config_value(&config, "max_parallel_tests");
let auto_download = get_config_value(&config, "fixtures.auto_download");
```

## Best Practices

1. **Use environment variables for CI/CD**: Override sensitive or environment-specific settings
2. **Validate configurations**: Always validate before using in production
3. **Use predefined configs**: Start with `ci_config()` or `dev_config()` and customize
4. **Monitor resource usage**: Adjust parallel tests based on available resources
5. **Secure fixture URLs**: Use HTTPS URLs and verify checksums
6. **Regular cleanup**: Configure appropriate cleanup intervals for fixtures
7. **Appropriate timeouts**: Balance between catching hangs and allowing slow tests

## Troubleshooting

### Common Issues

**Configuration file not found:**
- Check file path and permissions
- Use `BITNET_TEST_CONFIG` environment variable
- Verify TOML syntax

**Validation errors:**
- Run `cargo run --bin validate_config` for detailed errors
- Check file permissions for cache and report directories
- Verify C++ binary path and permissions

**Performance issues:**
- Reduce `max_parallel_tests` if system is overloaded
- Increase cache size if fixtures are re-downloaded frequently
- Disable coverage generation for faster local testing

**Cross-validation failures:**
- Verify C++ binary is compatible version
- Check tolerance settings aren't too strict
- Ensure test models are available

### Debug Configuration Loading
```rust
use bitnet_tests::{load_test_config, TestError};

match load_test_config() {
    Ok(config) => println!("Config loaded: {:?}", config),
    Err(TestError::ConfigError { message }) => {
        eprintln!("Config error: {}", message);
    }
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Migration Guide

When upgrading the testing framework, configuration changes may be required:

1. **Check validation**: Run the validator on existing configurations
2. **Review new options**: Check for new configuration options in examples
3. **Update environment variables**: Verify environment variable names haven't changed
4. **Test thoroughly**: Validate that tests still run as expected

For specific migration instructions, see the framework's CHANGELOG.md.
