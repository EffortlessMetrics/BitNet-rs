# Troubleshooting and Debugging Guide

## Introduction

This guide helps you diagnose and resolve common issues encountered when working with the BitNet.rs testing framework. It covers debugging techniques, common error patterns, and systematic approaches to problem resolution.

## General Debugging Approach

### 1. Enable Detailed Logging

```bash
# Enable debug logging for all components
export RUST_LOG="debug"

# Enable specific component logging
export RUST_LOG="bitnet=debug,crossval=trace,test_harness=info"

# Run tests with logging
cargo test --no-default-features --features cpu -- --nocapture

# Save logs to file
cargo test --no-default-features --features cpu -- --nocapture 2>&1 | tee test_debug.log
```

### 2. Use Rust Debugging Tools

```bash
# Run with backtrace on panic
export RUST_BACKTRACE=1
cargo test --no-default-features --features cpu

# Full backtrace
export RUST_BACKTRACE=full
cargo test --no-default-features --features cpu

# Run specific test with debugging
cargo test test_name --no-default-features --features cpu -- --exact --nocapture
```

### 3. Isolate the Problem

```bash
# Run single test
cargo test specific_test_name --no-default-features --features cpu -- --exact

# Run tests in single thread
cargo test --no-default-features --features cpu -- --test-threads=1

# Run with timeout
timeout 60s cargo test --no-default-features --features cpu
```

## Common Issues and Solutions

### Test Execution Issues

#### Issue: Tests Hang or Timeout

**Symptoms:**
- Tests never complete
- High CPU usage with no progress
- Timeout errors

**Debugging Steps:**
1. Check for deadlocks:
```bash
# Run with timeout and get stack trace
timeout 30s cargo test --no-default-features --features cpu &
sleep 20
kill -QUIT $!  # Send SIGQUIT for stack trace
```

2. Enable async debugging:
```rust
// Add to test
tokio::time::timeout(Duration::from_secs(10), async_operation).await
    .expect("Operation should complete within 10 seconds");
```

3. Check resource usage:
```bash
# Monitor during test execution
htop
# or
ps aux | grep cargo
```

**Solutions:**
- Add timeouts to async operations
- Check for infinite loops in test logic
- Reduce parallelism: `cargo test --no-default-features --features cpu -- --test-threads=1`
- Increase system timeout limits

#### Issue: Tests Fail Intermittently

**Symptoms:**
- Tests pass sometimes, fail other times
- Different results on different runs
- Race conditions

**Debugging Steps:**
1. Run tests multiple times:
```bash
# Run test 10 times
for i in {1..10}; do
    echo "Run $i"
    cargo test --no-default-features --features cpu test_name -- --exact || break
done
```

2. Check for shared state:
```rust
// Look for static variables or shared resources
static mut SHARED_STATE: Option<SomeType> = None;
```

3. Enable thread sanitizer (if available):
```bash
export RUSTFLAGS="-Z sanitizer=thread"
cargo test --no-default-features --features cpu
```

**Solutions:**
- Remove shared mutable state
- Use proper synchronization primitives
- Make tests independent
- Use fresh instances for each test

#### Issue: Out of Memory Errors

**Symptoms:**
- `std::alloc::handle_alloc_error`
- System becomes unresponsive
- Tests killed by OOM killer

**Debugging Steps:**
1. Monitor memory usage:
```bash
# Run with memory monitoring
valgrind --tool=massif cargo test --no-default-features --features cpu
# or
/usr/bin/time -v cargo test --no-default-features --features cpu
```

2. Check for memory leaks:
```rust
// Add memory tracking to tests
let memory_before = get_memory_usage();
// ... test code ...
let memory_after = get_memory_usage();
assert!(memory_after - memory_before < MAX_MEMORY_INCREASE);
```

**Solutions:**
- Reduce test data size
- Implement proper cleanup
- Use smaller models for testing
- Increase system memory or swap

### Fixture Management Issues

#### Issue: Fixture Download Failures

**Symptoms:**
- `FixtureError: Download failed`
- Network timeout errors
- Checksum mismatch errors

**Debugging Steps:**
1. Test network connectivity:
```bash
curl -I https://example.com/model.gguf
wget --spider https://example.com/model.gguf
```

2. Check cache directory:
```bash
ls -la tests/cache/
df -h tests/cache/
```

3. Verify checksums manually:
```bash
sha256sum tests/cache/model.gguf
```

**Solutions:**
- Check internet connection
- Clear cache: `rm -rf tests/cache`
- Use local fixtures for offline testing
- Update fixture URLs and checksums
- Increase download timeout

#### Issue: Corrupted Fixtures

**Symptoms:**
- `FixtureError: Checksum mismatch`
- Model loading failures
- Unexpected test results

**Debugging Steps:**
1. Verify file integrity:
```bash
file tests/cache/model.gguf
hexdump -C tests/cache/model.gguf | head
```

2. Re-download fixture:
```bash
rm tests/cache/model.gguf
cargo test --no-default-features --features cpu  # Will trigger re-download
```

**Solutions:**
- Clear and re-download fixtures
- Check disk space and permissions
- Verify source file integrity
- Update checksums in configuration

### Cross-Validation Issues

#### Issue: C++ Implementation Not Found

**Symptoms:**
- `CrossValidationError: C++ binary not found`
- Permission denied errors
- Command not found errors

**Debugging Steps:**
1. Check binary existence and permissions:
```bash
ls -la legacy/bitnet.cpp/build/bitnet_cpp
file legacy/bitnet.cpp/build/bitnet_cpp
ldd legacy/bitnet.cpp/build/bitnet_cpp  # Check dependencies
```

2. Test C++ binary directly:
```bash
./legacy/bitnet.cpp/build/bitnet_cpp --help
```

**Solutions:**
- Build C++ implementation: `cd legacy/bitnet.cpp && mkdir build && cd build && cmake .. && make`
- Set correct path in configuration
- Check file permissions: `chmod +x legacy/bitnet.cpp/build/bitnet_cpp`
- Install missing dependencies

#### Issue: Accuracy Comparison Failures

**Symptoms:**
- `Accuracy below threshold`
- Token mismatches
- Probability divergence

**Debugging Steps:**
1. Analyze first mismatch:
```rust
if let Some(mismatch) = &result.accuracy_result.first_mismatch {
    println!("Mismatch at position {}", mismatch.position);
    println!("Rust token: {} ({})", mismatch.rust_token, 
             tokenizer.decode(&[mismatch.rust_token]));
    println!("C++ token: {} ({})", mismatch.cpp_token,
             tokenizer.decode(&[mismatch.cpp_token]));
    println!("Context: {:?}", mismatch.context);
}
```

2. Compare tokenization:
```rust
let rust_tokens = rust_impl.tokenize(input).await?;
let cpp_tokens = cpp_impl.tokenize(input).await?;
println!("Rust tokens: {:?}", rust_tokens);
println!("C++ tokens: {:?}", cpp_tokens);
```

3. Check model compatibility:
```bash
# Compare model metadata
strings model.gguf | grep -i version
```

**Solutions:**
- Verify model compatibility between implementations
- Check for floating-point precision differences
- Adjust tolerance settings if differences are acceptable
- Update implementation to match reference behavior

### Performance Issues

#### Issue: Slow Test Execution

**Symptoms:**
- Tests take much longer than expected
- High CPU usage
- System becomes unresponsive

**Debugging Steps:**
1. Profile test execution:
```bash
# Use perf (Linux)
perf record cargo test --no-default-features --features cpu
perf report

# Use Instruments (macOS)
instruments -t "Time Profiler" cargo test --no-default-features --features cpu
```

2. Identify bottlenecks:
```rust
// Add timing to test sections
let start = Instant::now();
// ... test code ...
println!("Section took: {:?}", start.elapsed());
```

**Solutions:**
- Reduce test data size
- Use smaller models for unit tests
- Optimize critical paths
- Increase parallelism where safe
- Use release builds for performance tests

#### Issue: Memory Usage Too High

**Symptoms:**
- High memory usage during tests
- System swapping
- OOM killer activation

**Debugging Steps:**
1. Monitor memory usage:
```bash
# Monitor memory during test
watch -n 1 'ps aux | grep cargo'
```

2. Use memory profilers:
```bash
# Valgrind (Linux)
valgrind --tool=massif cargo test --no-default-features --features cpu

# Heaptrack (Linux)
heaptrack cargo test --no-default-features --features cpu
```

**Solutions:**
- Use smaller test models
- Implement proper resource cleanup
- Reduce parallel test execution
- Add memory limits to tests

### Build and Dependency Issues

#### Issue: Compilation Failures

**Symptoms:**
- Rust compilation errors
- Missing dependencies
- Version conflicts

**Debugging Steps:**
1. Check Rust version:
```bash
rustc --version
cargo --version
```

2. Update dependencies:
```bash
cargo update
cargo clean && cargo build --no-default-features --features cpu
```

3. Check for conflicts:
```bash
cargo tree
cargo tree --duplicates
```

**Solutions:**
- Update Rust toolchain
- Resolve dependency conflicts
- Clean build artifacts
- Check feature flag compatibility

#### Issue: FFI/C++ Integration Problems

**Symptoms:**
- Linking errors
- Segmentation faults
- ABI compatibility issues

**Debugging Steps:**
1. Check C++ build:
```bash
cd legacy/bitnet.cpp/build
make clean && make VERBOSE=1
```

2. Test FFI bindings:
```bash
# Run with debug symbols
cargo test --no-default-features --features debug-symbols
```

3. Use debugging tools:
```bash
# GDB for segfaults
gdb --args cargo test --no-default-features --features cpu
# or
lldb -- cargo test --no-default-features --features cpu
```

**Solutions:**
- Rebuild C++ implementation
- Check ABI compatibility
- Update FFI bindings
- Use proper memory management

## Systematic Debugging Process

### Step 1: Reproduce the Issue

1. **Isolate the failing test:**
```bash
cargo test failing_test_name --no-default-features --features cpu -- --exact --nocapture
```

2. **Create minimal reproduction:**
```rust
#[tokio::test]
async fn minimal_repro() {
    // Minimal code that reproduces the issue
}
```

3. **Document the environment:**
```bash
# Save environment info
rustc --version > debug_info.txt
cargo --version >> debug_info.txt
uname -a >> debug_info.txt
```

### Step 2: Gather Information

1. **Enable comprehensive logging:**
```bash
export RUST_LOG="trace"
export RUST_BACKTRACE="full"
cargo test failing_test --no-default-features --features cpu -- --nocapture 2>&1 | tee debug.log
```

2. **Collect system information:**
```bash
# Memory usage
free -h
# Disk usage
df -h
# Process information
ps aux | grep cargo
```

3. **Check configuration:**
```bash
cat tests/config.toml
env | grep BITNET
env | grep CROSSVAL
```

### Step 3: Analyze the Problem

1. **Review error messages:**
   - Look for specific error types
   - Check error context and stack traces
   - Identify the failing component

2. **Check recent changes:**
```bash
git log --oneline -10
git diff HEAD~1
```

3. **Compare with working state:**
```bash
git bisect start
git bisect bad HEAD
git bisect good last_known_good_commit
```

### Step 4: Test Hypotheses

1. **Test individual components:**
```rust
#[tokio::test]
async fn test_component_isolation() {
    // Test each component separately
}
```

2. **Vary test conditions:**
   - Different input sizes
   - Different configurations
   - Different environments

3. **Use debugging tools:**
```bash
# Memory debugging
valgrind cargo test --no-default-features --features cpu

# Thread debugging
cargo test --no-default-features --features cpu -- --test-threads=1
```

### Step 5: Implement and Verify Fix

1. **Make targeted changes:**
   - Fix the root cause, not symptoms
   - Make minimal changes
   - Document the fix

2. **Test the fix:**
```bash
# Test the specific issue
cargo test failing_test --no-default-features --features cpu

# Run related tests
cargo test related_module --no-default-features --features cpu

# Run full test suite
cargo test --no-default-features --features cpu
```

3. **Verify no regressions:**
```bash
# Run tests multiple times
for i in {1..5}; do cargo test --no-default-features --features cpu || break; done
```

## Debugging Tools and Techniques

### Rust-Specific Tools

#### 1. Cargo Tools
```bash
# Check for common issues
cargo clippy

# Format code
cargo fmt

# Security audit
cargo audit

# Dependency analysis
cargo tree
cargo outdated
```

#### 2. Testing Tools
```bash
# Test with different features
cargo test --no-default-features --features "cpu,feature1,feature2"
cargo test --no-default-features --features cpu

# Test documentation
cargo test --no-default-features --doc --no-default-features --features cpu

# Test examples
cargo test --no-default-features --examples --no-default-features --features cpu
```

#### 3. Profiling Tools
```bash
# CPU profiling
cargo install flamegraph
cargo flamegraph --test test_name

# Memory profiling
cargo install cargo-profdata
cargo profdata -- test test_name
```

### System-Level Debugging

#### 1. Process Monitoring
```bash
# Monitor system resources
htop
iotop
nethogs

# Process tracing
strace -p $(pgrep cargo)
ltrace -p $(pgrep cargo)
```

#### 2. Memory Analysis
```bash
# Memory usage
pmap $(pgrep cargo)
cat /proc/$(pgrep cargo)/status

# Memory leaks
valgrind --leak-check=full cargo test --no-default-features --features cpu
```

#### 3. Network Debugging
```bash
# Network activity
netstat -tulpn
ss -tulpn

# Packet capture
tcpdump -i any host example.com
```

## Prevention Strategies

### 1. Robust Test Design
- Add comprehensive error handling
- Use timeouts for all async operations
- Implement proper resource cleanup
- Make tests deterministic

### 2. Continuous Monitoring
- Set up automated test runs
- Monitor test execution times
- Track memory usage trends
- Alert on test failures

### 3. Documentation
- Document known issues and workarounds
- Maintain troubleshooting runbooks
- Keep debugging guides updated
- Share knowledge with team

### 4. Environment Management
- Use consistent test environments
- Pin dependency versions
- Automate environment setup
- Test on multiple platforms

## Getting Help

### 1. Internal Resources
- Check existing documentation
- Search previous issues and solutions
- Consult team knowledge base
- Review code comments and commit messages

### 2. Community Resources
- Rust community forums
- GitHub issues and discussions
- Stack Overflow
- Discord/IRC channels

### 3. Reporting Issues
When reporting issues, include:
- Minimal reproduction case
- Complete error messages and stack traces
- Environment information (OS, Rust version, etc.)
- Steps taken to debug
- Expected vs. actual behavior

### 4. Creating Debug Reports
```bash
# Generate comprehensive debug report
cat > debug_report.md << EOF
# Debug Report

## Issue Description
[Describe the issue]

## Environment
- OS: $(uname -a)
- Rust: $(rustc --version)
- Cargo: $(cargo --version)

## Reproduction Steps
1. [Step 1]
2. [Step 2]

## Error Output
\`\`\`
$(cargo test --no-default-features --features cpu failing_test 2>&1)
\`\`\`

## Additional Information
[Any other relevant information]
EOF
```

This troubleshooting guide provides systematic approaches to identifying and resolving issues in the BitNet.rs testing framework. Regular use of these debugging techniques will help maintain a robust and reliable testing environment.