# BitNet.rs Testing Framework Documentation

This directory contains comprehensive documentation for the BitNet.rs testing framework, covering everything from basic usage to advanced testing scenarios.

## Documentation Structure

- **[Framework Overview](framework-overview.md)** - Architecture and core concepts
- **[Test Authoring Guide](test-authoring-guide.md)** - Guidelines and best practices for writing tests
- **[Cross-Validation Guide](cross-validation-guide.md)** - Setup and usage of cross-implementation comparison
- **[Troubleshooting Guide](troubleshooting-guide.md)** - Common issues and debugging techniques
- **[Performance Testing Guide](performance-testing-guide.md)** - Benchmarking and performance validation

## Quick Start

For developers new to the testing framework:

1. Read the [Framework Overview](framework-overview.md) to understand the architecture
2. Follow the [Test Authoring Guide](test-authoring-guide.md) to write your first tests
3. Use the [Cross-Validation Guide](cross-validation-guide.md) to validate against C++ implementation
4. Refer to the [Troubleshooting Guide](troubleshooting-guide.md) when issues arise

## Running Tests

```bash
# Run all tests
cargo test

# Run unit tests only
cargo test --lib

# Run integration tests
cargo test --test integration_tests

# Run cross-validation tests
cargo test --test crossval_tests

# Generate coverage report
cargo tarpaulin --out html --output-dir target/coverage
```

## Contributing

When adding new tests or modifying the testing framework, please:

1. Follow the guidelines in the [Test Authoring Guide](test-authoring-guide.md)
2. Update documentation as needed
3. Ensure all tests pass before submitting changes
4. Add appropriate test coverage for new functionality

## Support

If you encounter issues not covered in the troubleshooting guide, please:

1. Check existing GitHub issues
2. Create a new issue with detailed reproduction steps
3. Include relevant logs and system information