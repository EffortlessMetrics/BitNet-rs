# BitNet.rs TODO Implementation Quick Start Guide

**Estimated Total Effort:** 270-360 hours (6-9 person-weeks)  
**Report Date:** 2025-10-19

---

## Start Here: Top 5 Priorities

### 1. AC05 Health Checks (CRITICAL) - 40-60 hours
**Files:** `crates/bitnet-server/tests/ac05_health_checks.rs`  
**Blocked By:** Nothing (start here!)  
**Blocks:** Production deployment certification

**Quick Start:**
```bash
# 1. Create health response struct in bitnet-server/src/monitoring/
# 2. Implement /health endpoint
# 3. Implement Kubernetes probe endpoints
# 4. Add performance indicator calculations
# 5. Create health monitoring system

# Test with:
cargo test -p bitnet-server --test ac05_health_checks
```

**Key Tasks:**
- [ ] HealthResponse struct + JSON serialization
- [ ] Component status tracking (model_manager, execution_router, etc)
- [ ] Kubernetes liveness probe handler (<100ms response)
- [ ] Kubernetes readiness probe handler (checks model_loaded, device_available)
- [ ] CPU/GPU utilization metrics
- [ ] Memory usage tracking
- [ ] Performance indicator calculation
- [ ] Error handling and degraded state support

---

### 2. Server Infrastructure Gaps (MEDIUM) - 20-25 hours
**Files:** `crates/bitnet-server/src/{lib.rs, execution_router.rs, batch_engine.rs}`  
**Blocked By:** Nothing  
**Blocks:** GPU support, health checks

**Quick Start:**
```bash
# Priority fixes:
# 1. Make device selection configurable (currently hardcoded CPU)
# 2. Implement GPU memory reporting
# 3. Implement graceful shutdown
# 4. Fix CUDA device detection

# Check existing TODOs:
grep -n "TODO" crates/bitnet-server/src/*.rs
```

**Key Tasks:**
- [ ] Device configuration system (CPU/GPU/Metal selection)
- [ ] GPU memory reporting implementation
- [ ] CUDA device detection
- [ ] Graceful shutdown with timeout
- [ ] Metal device support stub

---

### 3. AC04 Receipt Generation (HIGH) - 20-30 hours
**Files:** `crates/bitnet-inference/tests/issue_254_ac4_receipt_generation.rs`  
**Blocked By:** Inference engine kernel tracking  
**Blocks:** Compute integrity validation

**Quick Start:**
```bash
# 1. Create InferenceReceipt struct
# 2. Implement kernel ID tracking in inference engine
# 3. Add environment variable capture
# 4. Implement file I/O to ci/inference.json
# 5. Create receipt validation logic

# Enable test:
cargo test -p bitnet-inference --test issue_254_ac4_receipt_generation -- --ignored
```

**Key Tasks:**
- [ ] InferenceReceipt struct definition
- [ ] Kernel ID tracking system
- [ ] Environment variable capture
- [ ] Receipt file I/O
- [ ] Mock kernel detection
- [ ] Performance baseline measurement

---

### 4. Real Model Loading Tests (MEDIUM) - 25-35 hours
**Files:** `crates/bitnet-models/tests/real_model_loading.rs`  
**Blocked By:** Model loading infrastructure  
**Blocks:** Real model validation testing

**Quick Start:**
```bash
# 1. Implement ProductionModelLoader
# 2. Add model structure validation
# 3. Implement tensor alignment checks (32-byte)
# 4. Add device compatibility checking

# Enable and test:
# First remove #![cfg(false)] from test file
cargo test -p bitnet-models --test real_model_loading
```

**Key Tasks:**
- [ ] ProductionModelLoader implementation
- [ ] Model structure validation (layers, dimensions)
- [ ] Tensor alignment verification (32-byte requirement)
- [ ] Quantization format detection
- [ ] Device compatibility checking
- [ ] Memory allocation safety
- [ ] Loading performance benchmarks

---

### 5. Cross-Validation Infrastructure (HIGH) - 30-40 hours
**Files:** `crates/bitnet-inference/tests/ac4_cross_validation_accuracy.rs`  
**Blocked By:** C++ reference implementation, GGML FFI  
**Blocks:** Numerical accuracy validation

**Quick Start:**
```bash
# 1. Implement model loading for cross-validation
# 2. Create BitNet.rs inference wrapper
# 3. Implement C++ reference runner
# 4. Create comparison metrics system
# 5. Implement aggregation functions

# Enable test:
# First remove #![cfg(any())] from top of file
cargo test -p bitnet-inference --test ac4_cross_validation_accuracy -- --ignored
```

**Key Tasks:**
- [ ] Model loading wrapper
- [ ] BitNet.rs inference execution
- [ ] C++ reference implementation runner
- [ ] Output comparison metrics
- [ ] Accuracy aggregation (>99% target)
- [ ] Correlation calculation (>99.9% target)
- [ ] MSE measurement (≤1e-6 target)

---

## Implementation Order (Phase Timeline)

### Week 1-2: Production Readiness
1. AC05 Health Checks (40-60h)
2. Server Infrastructure (20-25h)

### Week 3-4: Validation Infrastructure  
1. AC04 Receipt Generation (20-30h)
2. Cross-Validation (30-40h)
3. Real Model Loading (25-35h)

### Week 5-6: Feature Completeness
1. TL1/TL2 Quantization (20-30h)
2. Tokenizer Tests (25-35h)
3. Autoregressive Generation (15-20h)

### Week 7-8: Quality & Polish
1. Property Tests (15-20h)
2. Mock Elimination (10-15h)
3. Documentation (5h)

---

## Common Implementation Patterns

### Pattern 1: Test Infrastructure (Tokenizer, Model Loading)

```rust
// Step 1: Remove #![cfg(false)] or #![cfg(any())]
// Step 2: Create helper implementation functions
// Step 3: Implement unimplemented!() stubs one by one
// Step 4: Run tests and fix compilation errors
// Step 5: Implement actual functionality

// From this:
#[test]
fn test_feature() {
    let result = helper_function(); // unimplemented!()
    assert!(result.is_ok());
}

// To this:
#[test]
fn test_feature() {
    let result = helper_function();
    assert!(result.is_ok());
}

fn helper_function() -> Result<Data> {
    // Actual implementation
}
```

### Pattern 2: API Endpoints (Health Checks)

```rust
// Step 1: Create response struct with JSON serialization
#[derive(Serialize)]
struct HealthResponse {
    status: String,
    timestamp: String,
    // ... fields
}

// Step 2: Implement handler
async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    // Implementation
}

// Step 3: Register route
let app = Router::new()
    .route("/health", get(health_handler))
    .with_state(state);

// Step 4: Test endpoint
// Uses tokio::test with HTTP client
```

### Pattern 3: Feature Implementation (Receipt System)

```rust
// Step 1: Define data structures
#[derive(Serialize, Deserialize)]
struct InferenceReceipt {
    // All fields from spec
}

// Step 2: Implement generation during inference
impl InferenceEngine {
    fn track_kernel_id(&mut self, kernel: &str) {
        // Add to receipt.kernels
    }
}

// Step 3: Implement validation
impl InferenceReceipt {
    fn validate(&self) -> Result<()> {
        // Verify compute_path != "mock"
        // Verify required fields
    }
}

// Step 4: Implement file I/O
impl InferenceReceipt {
    fn save_to_file(&self, path: &Path) -> Result<()> {
        serde_json::to_writer(File::create(path)?, self)?;
        Ok(())
    }
}
```

---

## Testing Strategy

### Health Checks
```bash
# Manual testing
curl http://localhost:8000/health
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready

# Automated tests
cargo test -p bitnet-server --test ac05_health_checks -- --test-threads=1

# Load testing
ab -n 1000 -c 10 http://localhost:8000/health
```

### Receipt Generation
```bash
# Verify receipt file created
test -f ci/inference.json && echo "Receipt found"

# Validate schema
jq . ci/inference.json

# Verify compute_path
jq .compute_path ci/inference.json  # Should be "real"
```

### Model Loading
```bash
# Test with real model
export BITNET_GGUF=models/model.gguf
cargo test -p bitnet-models --test real_model_loading

# Test with timeout
timeout 120 cargo test -p bitnet-models --test real_model_loading
```

---

## Performance Targets

| Component | Target | Priority |
|-----------|--------|----------|
| Health check response | <50ms avg, <200ms P99 | CRITICAL |
| Liveness probe | <100ms | CRITICAL |
| Tokenization | ≥10K tokens/sec | HIGH |
| Lookup operations | ≤10ns per access | MEDIUM |
| Model loading | <60s for 2B models | MEDIUM |
| Quantization accuracy | ≥99% tokens | HIGH |

---

## Debugging Tips

### "unimplemented!() called"
```bash
# Find all unimplemented stubs
grep -r "unimplemented" crates/

# Enable debug builds to get better stack traces
RUST_BACKTRACE=1 cargo test test_name
```

### "TODO comment encountered"
```bash
# Find all TODOs in specific test file
grep -n "TODO" crates/bitnet-server/tests/ac05_health_checks.rs

# Count TODOs by category
grep "TODO" crates/bitnet-server/tests/ac05_health_checks.rs | wc -l
```

### Test Failures
```bash
# Run with verbose output
cargo test test_name -- --nocapture

# Run single test with logging
RUST_LOG=debug cargo test test_name -- --nocapture

# Profile tests
cargo test test_name --release
```

---

## Key Files Reference

| Task | Primary File | Secondary Files |
|------|-------------|-----------------|
| Health checks | `crates/bitnet-server/tests/ac05_health_checks.rs` | `crates/bitnet-server/src/monitoring/` |
| Receipt system | `crates/bitnet-inference/tests/issue_254_ac4_receipt_generation.rs` | `crates/bitnet-inference/src/engine.rs` |
| Server config | `crates/bitnet-server/src/lib.rs` | `crates/bitnet-server/src/{batch_engine,execution_router}.rs` |
| Model loading | `crates/bitnet-models/tests/real_model_loading.rs` | `crates/bitnet-models/src/lib.rs` |
| Cross-validation | `crates/bitnet-inference/tests/ac4_cross_validation_accuracy.rs` | `crates/bitnet-models/src/lib.rs` |

---

## Getting Help

### Understanding Requirements
1. Read the specific test file to see what's expected
2. Check CLAUDE.md for architecture overview
3. Look at related passing tests for patterns
4. Review doc files in `docs/` directory

### Finding Similar Code
```bash
# Find similar implementations
grep -r "impl.*Response" crates/bitnet-server/

# Look at working tests
ls crates/bitnet-inference/tests/ | head -10
cat crates/bitnet-inference/tests/decode_smoke.rs
```

### Common Patterns
- Response types: See `bitnet_common/src/response.rs` 
- Error handling: See `bitnet_common/src/error.rs`
- Testing patterns: See `tests/common/` directory
- Async code: Look at tokio test examples

---

## Checklist for First Implementation

- [ ] Pick highest priority item from Phase 1
- [ ] Read related test file completely
- [ ] Identify all unimplemented!() stubs
- [ ] Create implementation in appropriate source file
- [ ] Run tests: `cargo test --test <name>`
- [ ] Fix compiler errors
- [ ] Test passes locally
- [ ] Create git commit
- [ ] Create PR with description
- [ ] Request review
- [ ] Merge

---

## Next Steps

1. **NOW:** Choose one high-priority task above
2. **TODAY:** Create skeleton implementation (stub functions)
3. **THIS WEEK:** First complete implementation done
4. **NEXT WEEK:** Start second high-priority task
5. **MONTH:** Complete all Phase 1 items

Start with AC05 Health Checks or Server Infrastructure - they have no dependencies and unblock other work!

