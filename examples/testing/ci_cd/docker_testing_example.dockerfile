# Example Dockerfile for BitNet.rs testing in containerized environments
# This demonstrates how to set up a complete testing environment in Docker

# Multi-stage build for efficient testing container
FROM rust:1.75-bullseye as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libssl-dev \
    curl \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Rust testing tools
RUN cargo install cargo-tarpaulin cargo-nextest cargo-criterion --locked

# Set up working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY Cargo.toml Cargo.lock ./
COPY crates/*/Cargo.toml ./crates/*/

# Create dummy source files to build dependencies
RUN mkdir -p src crates/bitnet-common/src crates/bitnet-models/src \
    crates/bitnet-quantization/src crates/bitnet-kernels/src \
    crates/bitnet-inference/src crates/bitnet-tokenizers/src \
    crates/bitnet-cli/src crates/bitnet-server/src \
    crates/bitnet-py/src crates/bitnet-wasm/src crates/bitnet-ffi/src \
    crates/bitnet-sys/src

# Create dummy lib.rs files
RUN echo "fn main() {}" > src/lib.rs && \
    find crates -name "src" -type d -exec sh -c 'echo "fn main() {}" > "$1/lib.rs"' _ {} \;

# Build dependencies
RUN cargo build --all-features --workspace

# Remove dummy files
RUN rm -rf src crates/*/src

# Testing stage
FROM builder as tester

# Copy actual source code
COPY . .

# Set up test environment
ENV BITNET_TEST_CACHE=/app/test-cache
ENV BITNET_LOG_LEVEL=info
ENV RUST_BACKTRACE=1
ENV CARGO_TERM_COLOR=always

# Create test directories
RUN mkdir -p $BITNET_TEST_CACHE test-results

# Install C++ BitNet for cross-validation (optional)
RUN git clone https://github.com/microsoft/BitNet.git bitnet-cpp && \
    cd bitnet-cpp && \
    make -j$(nproc) && \
    cp bitnet /usr/local/bin/ || echo "C++ BitNet build failed, skipping"

# Install Python dependencies for test reporting
RUN pip3 install jinja2 matplotlib seaborn pandas

# Copy test configuration
COPY examples/testing/ci_cd/test_configuration_examples.toml /app/test-config.toml

# Build the project
RUN cargo build --all-features --workspace --release

# Run tests by default
CMD ["cargo", "test", "--all-features", "--workspace", "--verbose"]

# Production testing image
FROM debian:bullseye-slim as test-runner

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for reporting
RUN pip3 install jinja2 matplotlib seaborn pandas

# Create app user
RUN useradd -m -u 1000 testuser

# Set up directories
WORKDIR /app
RUN mkdir -p test-cache test-results && \
    chown -R testuser:testuser /app

# Copy built binaries and test files
COPY --from=tester /app/target/release/ ./target/release/
COPY --from=tester /app/tests/ ./tests/
COPY --from=tester /app/examples/testing/ ./examples/testing/
COPY --from=tester /usr/local/bin/bitnet /usr/local/bin/ 2>/dev/null || true

# Copy test scripts
COPY scripts/generate_test_report.py ./scripts/
COPY examples/testing/ci_cd/test_configuration_examples.toml ./test-config.toml

# Set environment variables
ENV BITNET_TEST_CACHE=/app/test-cache
ENV BITNET_LOG_LEVEL=info
ENV RUST_BACKTRACE=1

# Switch to test user
USER testuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command runs comprehensive tests
CMD ["./scripts/run_comprehensive_tests.sh"]

# Example usage commands (as comments for documentation)
# 
# Build the testing image:
# docker build -f examples/testing/ci_cd/docker_testing_example.dockerfile -t bitnet-rs-test .
#
# Run unit tests:
# docker run --rm -v $(pwd)/test-results:/app/test-results bitnet-rs-test cargo test --lib
#
# Run integration tests:
# docker run --rm -v $(pwd)/test-results:/app/test-results bitnet-rs-test cargo test --test integration_tests
#
# Run cross-validation tests:
# docker run --rm -v $(pwd)/test-results:/app/test-results bitnet-rs-test cargo test --test cross_validation_tests
#
# Run performance benchmarks:
# docker run --rm -v $(pwd)/test-results:/app/test-results bitnet-rs-test cargo criterion
#
# Generate test report:
# docker run --rm -v $(pwd)/test-results:/app/test-results bitnet-rs-test python3 scripts/generate_test_report.py
#
# Run with custom configuration:
# docker run --rm -v $(pwd)/custom-config.toml:/app/test-config.toml -v $(pwd)/test-results:/app/test-results bitnet-rs-test
#
# Run in interactive mode for debugging:
# docker run --rm -it -v $(pwd)/test-results:/app/test-results bitnet-rs-test bash
#
# Run with GPU support (requires nvidia-docker):
# docker run --rm --gpus all -v $(pwd)/test-results:/app/test-results bitnet-rs-test cargo test --features gpu
#
# Run memory-constrained tests:
# docker run --rm -m 1g -v $(pwd)/test-results:/app/test-results bitnet-rs-test cargo test --features memory-constrained