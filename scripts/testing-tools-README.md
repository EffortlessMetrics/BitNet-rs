# Testing Tools

This directory contains helpful scripts and tools for working with the BitNet.rs testing framework.

## Test Template Generator

Quickly create test files from templates with proper structure and examples.

### Usage

**Linux/macOS:**
```bash
# Create a unit test
./scripts/create-test-template.sh my_feature_test

# Create an integration test
./scripts/create-test-template.sh -t integration workflow_test

# Create a unit test for a specific module
./scripts/create-test-template.sh -t unit -m bitnet_models model_test

# Create a performance test
./scripts/create-test-template.sh -t performance benchmark_test
```

**Windows (PowerShell):**
```powershell
# Create a unit test
.\scripts\create-test-template.ps1 my_feature_test

# Create an integration test
.\scripts\create-test-template.ps1 workflow_test -Type integration

# Create a unit test for a specific module
.\scripts\create-test-template.ps1 model_test -Type unit -Module bitnet_models

# Create a performance test
.\scripts\create-test-template.ps1 benchmark_test -Type performance
```

### What It Creates

The script generates complete test files with:

- **Unit Tests**: Basic test structure with success/error cases, multiple test scenarios, and TODO comments for guidance
- **Integration Tests**: Workflow testing with file I/O, component interaction, and error handling examples
- **Performance Tests**: Benchmarking templates with timing, memory usage, and throughput testing

### Generated File Locations

- **Unit tests**: `tests/unit/test_<name>.rs` or `tests/unit/<module>/test_<name>.rs`
- **Integration tests**: `tests/integration/<name>_integration_test.rs`
- **Performance tests**: `tests/performance/<name>_performance_test.rs`

## Quick Start Workflow

1. **Generate a test template**:
   ```bash
   ./scripts/create-test-template.sh my_new_feature
   ```

2. **Edit the generated file** to replace example code with your actual test logic

3. **Run your test**:
   ```bash
   cargo test --test test_my_new_feature
   ```

4. **Iterate and improve** your tests based on the patterns shown in the template

## Additional Resources

- [Quick Start Guide](../docs/testing/quick-start-guide.md) - Get started with testing in 5 minutes
- [Test Templates](../docs/testing/test-templates.md) - Copy-paste examples for common scenarios
- [Test Authoring Guide](../docs/testing/test-authoring-guide.md) - Comprehensive testing best practices
- [Example Tests](../tests/examples/) - Working examples you can run and study

## Tips

- Use descriptive test names that explain what you're testing
- Start with the generated template and customize it for your specific needs
- Look at existing tests in the codebase for inspiration
- Run `cargo test --help` to see all available testing options

Happy testing! 🧪