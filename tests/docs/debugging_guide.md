# BitNet.rs Test Debugging Guide

This guide explains how to use the comprehensive debugging support in the BitNet.rs testing framework to identify and resolve test issues quickly.

## Overview

The debugging system provides:
- **Detailed test execution tracing** - Track every phase of test execution
- **Resource monitoring** - Monitor memory, CPU, and system resources
- **Error capture and analysis** - Comprehensive error reporting with context
- **Performance profiling** - Identify slow tests and bottlenecks
- **Interactive debugging tools** - CLI tools for analyzing debug reports
- **Automated troubleshooting guides** - Generated guides for common issues

## Quick Start

### 1. Enable Debugging

Set environment variables to enable debugging:

```bash
export BITNET_DEBUG_ENABLED=true
export BITNET_DEBUG_VERBOSE=true
export BITNET_DEBUG_OUTPUT_DIR=tests/debug
```

### 2. Run Tests with Debugging

```bash
# Run all tests with debugging
BITNET_DEBUG_ENABLED=true cargo test

# Run specific test with full debugging
BITNET_DEBUG_ENABLED=true BITNET_DEBUG_VERBOSE=true cargo test my_failing_test

# Run with custom debug output directory
BITNET_DEBUG_OUTPUT_DIR=/tmp/bitnet_debug cargo test
```

### 3. Analyze Results

```bash
# Run the debug CLI
cargo run --example debug_cli_example

# Or analyze a specific report
cargo run --example debug_cli_example analyze tests/debug/debug_session_123/debug_report.json
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BITNET_DEBUG_ENABLED` | `false` | Enable/disable debugging |
| `BITNET_DEBUG_VERBOSE` | `false` | Enable verbose logging |
| `BITNET_DEBUG_STACK_TRACES` | `true` | Capture stack traces on errors |
| `BITNET_DEBUG_ENVIRONMENT` | `true` | Capture environment information |
| `BITNET_DEBUG_SYSTEM_INFO` | `true` | Capture system information |
| `BITNET_DEBUG_ARTIFACTS` | `true` | Save debug artifacts to files |
| `BITNET_DEBUG_MAX_FILES` | `100` | Maximum number of debug files to keep |
| `BITNET_DEBUG_OUTPUT_DIR` | `tests/debug` | Directory for debug output |

### Programmatic Configuration

```rust
use tests::common::debugging::DebugConfig;

let debug_config = DebugConfig {
    enabled: true,
    capture_stack_traces: true,
    capture_environment: true,
    capture_system_info: true,
    verbose_logging: true,
    save_debug_artifacts: true,
    max_debug_files: 100,
    debug_output_dir: PathBuf::from("tests/debug"),
};
```

## Debug Output Structure

When debugging is enabled, the following files are generated:

```
tests/debug/
├── debug_session_1234567890/
│   ├── debug_report.json          # Main debug report
│   ├── debug_summary.md           # Human-readable summary
│   ├── troubleshooting_guide.md   # Generated troubleshooting guide
│   ├── test_my_test/
│   │   ├── trace.json             # Detailed test trace
│   │   └── debug_messages.txt     # Debug messages
│   └── error_my_test_1234567890.json  # Error reports
```

## Using the Debug CLI

### Interactive Mode

```bash
cargo run --example debug_cli_example interactive
```

Available commands in interactive mode:
- `analyze <path>` - Analyze a debug report file
- `guide <test>` - Generate troubleshooting guide for a test
- `patterns` - Find common patterns across all reports
- `list` - List all available debug reports
- `help` - Show help message
- `quit` - Exit

### Command Line Mode

```bash
# Analyze a specific report
cargo run --example debug_cli_example analyze tests/debug/session_123/debug_report.json

# Find patterns across all reports
cargo run --example debug_cli_example patterns

# Generate troubleshooting guide for a specific test
cargo run --example debug_cli_example guide my_failing_test
```

## Debug Report Contents

### Main Report (`debug_report.json`)

```json
{
  "session_id": "debug_session_1234567890",
  "start_time": "2024-01-01T12:00:00Z",
  "end_time": "2024-01-01T12:05:00Z",
  "total_tests": 10,
  "failed_tests": 2,
  "error_count": 3,
  "test_summaries": [...],
  "performance_summary": {
    "total_duration": "5m",
    "peak_memory": 1073741824,
    "average_test_duration": "30s",
    "slowest_tests": [...]
  },
  "system_summary": {
    "cpu_cores": 8,
    "available_memory": 16777216000,
    "disk_space": 1000000000000
  },
  "recommendations": [...],
  "artifacts": [...]
}
```

### Test Trace (`test_name/trace.json`)

```json
{
  "test_name": "my_test",
  "start_time": "2024-01-01T12:00:00Z",
  "end_time": "2024-01-01T12:00:30Z",
  "phases": [
    {
      "name": "setup",
      "start_time": "2024-01-01T12:00:00Z",
      "end_time": "2024-01-01T12:00:05Z",
      "status": "Completed",
      "details": {}
    },
    {
      "name": "execute",
      "start_time": "2024-01-01T12:00:05Z",
      "end_time": "2024-01-01T12:00:25Z",
      "status": "Failed",
      "details": {
        "error": "Assertion failed"
      }
    }
  ],
  "resource_usage": [...],
  "debug_messages": [...],
  "stack_traces": [...],
  "artifacts": [...]
}
```

## Common Debugging Scenarios

### 1. Test Failures

When a test fails, the debugging system automatically:
- Captures the error with full context
- Records the exact phase where failure occurred
- Saves stack traces (if enabled)
- Generates troubleshooting suggestions

**Example workflow:**
```bash
# Run failing test with debugging
BITNET_DEBUG_ENABLED=true cargo test my_failing_test

# Analyze the results
cargo run --example debug_cli_example guide my_failing_test
```

### 2. Performance Issues

For slow tests, the system tracks:
- Execution time per phase
- Memory usage over time
- Resource consumption patterns
- Performance comparisons

**Example workflow:**
```bash
# Run with performance monitoring
BITNET_DEBUG_ENABLED=true cargo test slow_test

# Find performance patterns
cargo run --example debug_cli_example patterns
```

### 3. Memory Leaks

Memory issues are tracked through:
- Peak memory usage per test
- Memory usage over time
- Memory allocation patterns
- Resource cleanup verification

**Example workflow:**
```bash
# Run with memory monitoring
BITNET_DEBUG_ENABLED=true cargo test memory_intensive_test

# Analyze memory usage
cargo run --example debug_cli_example analyze tests/debug/latest/debug_report.json
```

### 4. Flaky Tests

For intermittent failures:
- Run multiple times with debugging
- Analyze patterns across runs
- Identify environmental factors
- Generate stability reports

**Example workflow:**
```bash
# Run flaky test multiple times
for i in {1..10}; do
  BITNET_DEBUG_ENABLED=true cargo test flaky_test
done

# Find patterns
cargo run --example debug_cli_example patterns
```

## Integration with Test Code

### Basic Integration

```rust
use tests::common::debug_integration::{create_debug_harness, debug_config_from_env};

#[tokio::test]
async fn my_test_with_debugging() {
    let test_config = TestConfig::default();
    let debug_config = debug_config_from_env();
    
    let debug_harness = create_debug_harness(test_config, Some(debug_config)).await?;
    let debugger = debug_harness.debugger();
    
    // Your test code here
    
    // Generate debug report
    let report = debugger.generate_debug_report().await?;
    debugger.save_debug_report(&report).await?;
}
```

### Custom Debug Messages

```rust
// Add debug messages during test execution
debugger.add_debug_message("my_test", "Starting critical section").await?;

// Start/end phases manually
debugger.start_phase("my_test", "data_loading").await?;
// ... data loading code ...
debugger.end_phase("my_test", "data_loading", true, None).await?;
```

### Error Capture

```rust
// Capture errors with context
if let Err(e) = risky_operation() {
    debugger.capture_error(Some("my_test"), &e).await?;
    return Err(e);
}
```

## Best Practices

### 1. Selective Debugging

Don't enable debugging for all tests all the time:
- Use for failing tests and performance investigations
- Enable in CI only for specific scenarios
- Use environment variables for easy control

### 2. Debug Message Guidelines

- Use clear, descriptive messages
- Include relevant context (values, states)
- Don't spam with too many messages
- Use appropriate log levels

### 3. Resource Management

- Clean up debug files regularly
- Set appropriate limits on file count
- Monitor debug output disk usage
- Use compression for long-term storage

### 4. CI Integration

```yaml
# GitHub Actions example
- name: Run tests with debugging on failure
  run: |
    if ! cargo test; then
      echo "Tests failed, running with debugging..."
      BITNET_DEBUG_ENABLED=true cargo test --no-fail-fast
    fi

- name: Upload debug artifacts
  if: failure()
  uses: actions/upload-artifact@v3
  with:
    name: debug-reports
    path: tests/debug/
    retention-days: 30
```

## Troubleshooting the Debugger

### Common Issues

1. **Debug files not generated**
   - Check `BITNET_DEBUG_ENABLED=true`
   - Verify write permissions to output directory
   - Check disk space

2. **Large debug files**
   - Reduce `BITNET_DEBUG_MAX_FILES`
   - Disable verbose logging for routine tests
   - Clean up old debug sessions

3. **Performance impact**
   - Debugging adds overhead, especially with verbose logging
   - Use selectively for problem investigation
   - Consider disabling stack trace capture for performance tests

### Debug the Debugger

```bash
# Enable debug logging for the debugger itself
RUST_LOG=debug BITNET_DEBUG_VERBOSE=true cargo test

# Check debug configuration
cargo run --example debug_cli_example interactive
> help
```

## Advanced Features

### Custom Metrics

```rust
// Add custom metrics to test results
let mut custom_metrics = HashMap::new();
custom_metrics.insert("api_calls".to_string(), 42.0);
custom_metrics.insert("cache_hits".to_string(), 0.85);

// Include in test metrics
TestMetrics {
    custom_metrics,
    ..Default::default()
}
```

### Pattern Analysis

The system can identify patterns across multiple test runs:
- Common failure modes
- Performance regressions
- Environmental dependencies
- Resource usage trends

### Automated Recommendations

Based on collected data, the system generates:
- Performance optimization suggestions
- Stability improvement recommendations
- Resource usage optimizations
- Test structure improvements

## Examples

See the following example files:
- `tests/examples/debugging_example.rs` - Basic debugging usage
- `tests/examples/debug_cli_example.rs` - CLI tool usage

Run examples:
```bash
# Run debugging example
cargo run --example debugging_example

# Run CLI example
cargo run --example debug_cli_example
```

## API Reference

### Core Types

- `TestDebugger` - Main debugging interface
- `DebugConfig` - Configuration for debugging features
- `DebugReport` - Comprehensive debug report
- `TestTrace` - Detailed trace of test execution
- `DebugCli` - Command-line interface for analysis

### Key Methods

- `TestDebugger::new(config)` - Create debugger instance
- `start_test_debug(name)` - Begin debugging a test
- `end_test_debug(name, result)` - End debugging a test
- `capture_error(test, error)` - Capture error with context
- `generate_debug_report()` - Create comprehensive report
- `generate_troubleshooting_guide()` - Create troubleshooting guide

For complete API documentation, see the inline documentation in the source code.