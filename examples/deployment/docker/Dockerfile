# Multi-stage Dockerfile for BitNet.rs
# Optimized for production deployment with minimal image size

# Build stage
FROM rust:1.75-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency files first for better caching
COPY Cargo.toml Cargo.lock ./
COPY crates/*/Cargo.toml ./crates/

# Create dummy source files to build dependencies
RUN mkdir -p crates/bitnet-common/src \
    crates/bitnet-models/src \
    crates/bitnet-quantization/src \
    crates/bitnet-kernels/src \
    crates/bitnet-inference/src \
    crates/bitnet-tokenizers/src \
    crates/bitnet-server/src \
    crates/bitnet-cli/src \
    crates/bitnet-ffi/src \
    && echo "fn main() {}" > crates/bitnet-cli/src/main.rs \
    && echo "// dummy" > crates/bitnet-common/src/lib.rs \
    && echo "// dummy" > crates/bitnet-models/src/lib.rs \
    && echo "// dummy" > crates/bitnet-quantization/src/lib.rs \
    && echo "// dummy" > crates/bitnet-kernels/src/lib.rs \
    && echo "// dummy" > crates/bitnet-inference/src/lib.rs \
    && echo "// dummy" > crates/bitnet-tokenizers/src/lib.rs \
    && echo "// dummy" > crates/bitnet-server/src/lib.rs \
    && echo "// dummy" > crates/bitnet-ffi/src/lib.rs

# Build dependencies
RUN cargo build --release --bin bitnet-server
RUN rm -rf crates/*/src

# Copy actual source code
COPY crates/ ./crates/
COPY src/ ./src/
COPY examples/ ./examples/
COPY benches/ ./benches/
COPY tests/ ./tests/

# Build the application
RUN cargo build --release --bin bitnet-server

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r bitnet && useradd -r -g bitnet bitnet

# Create directories
RUN mkdir -p /app/models /app/config /app/logs \
    && chown -R bitnet:bitnet /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/bitnet-server /app/bitnet-server
RUN chmod +x /app/bitnet-server

# Copy configuration files
COPY examples/deployment/docker/config/ /app/config/

# Switch to non-root user
USER bitnet
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Expose port
EXPOSE 3000

# Environment variables
ENV RUST_LOG=info
ENV BITNET_CONFIG_PATH=/app/config/bitnet.toml
ENV BITNET_MODEL_PATH=/app/models/bitnet-1.58b.gguf

# Run the application
CMD ["./bitnet-server"]
