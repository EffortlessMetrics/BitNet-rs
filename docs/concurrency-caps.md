# Concurrency Caps Implementation

BitNet-rs implements a comprehensive concurrency capping strategy to prevent resource exhaustion during parallel test execution.

## Core Components

### 1. Preflight Script (`scripts/preflight.sh`)
- Monitors system resource usage (PIDs, file descriptors, load)
- Sets conservative concurrency defaults
- Auto-degrades to single-threaded mode under high system load
- Configures BLAS thread limits for Python validation scripts

### 2. E2E Gate (`scripts/e2e-gate.sh`)
- Limits concurrent heavy test suites (cross-validation, integration tests)
- Uses file locking to queue test runs when system is busy
- Integrates with preflight caps for resource management

### 3. Cargo Aliases (`.cargo/config.toml`)
- `t2`: Run tests with 2-thread cap
- `t1`: Run tests with 1-thread (deterministic mode)
- `crossval-capped`: Cross-validation with thread caps
- `gpu-capped`: GPU tests with concurrency limits

### 4. Test Infrastructure (`tests/common/concurrency_caps.rs`)
- Rayon thread pool initialization with caps
- Async task concurrency limits
- Deterministic execution helpers

### 5. Container Limits (`docker-compose.test.yml`)
- Hard resource limits (CPU, memory, PIDs, file descriptors)
- Isolated test environments for different workloads
- Volume caching for faster subsequent builds

## Environment Variables

```bash
# Thread control
RUST_TEST_THREADS=2      # Rust test parallelism
RAYON_NUM_THREADS=2      # Rayon thread pool size
CROSSVAL_WORKERS=2       # Cross-validation test workers

# BLAS thread limits (Python scripts)
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1

# BitNet deterministic execution
BITNET_DETERMINISTIC=1
BITNET_SEED=42
```

## Usage Examples

```bash
# Standard capped test execution
scripts/preflight.sh && cargo t2

# Gate heavy E2E tests (max 2 concurrent)
scripts/e2e-gate.sh ./scripts/crossval.sh

# Container-isolated testing
docker-compose -f docker-compose.test.yml up rust-cpu-tests

# CI/deterministic mode
RUST_TEST_THREADS=1 RAYON_NUM_THREADS=1 cargo t1
```