# Multi-stage Docker build for BitNet-rs
# Supports both CPU and GPU backends

# Build stage
FROM rust:1.89-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/
COPY crossval/ ./crossval/
COPY tests/ ./tests/
COPY xtask/ ./xtask/
COPY src/ ./src/
COPY build.rs ./

# Build with CPU features by default
ARG FEATURES=cpu
RUN cargo build --release --no-default-features --features ${FEATURES}

# Runtime stage - minimal image
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash bitnet

# Copy binary from builder
COPY --from=builder /app/target/release/bitnet /usr/local/bin/bitnet
COPY --from=builder /app/target/release/bitnet-server /usr/local/bin/bitnet-server

# Create directories for models and data
RUN mkdir -p /data /models && \
    chown -R bitnet:bitnet /data /models

# Switch to non-root user
USER bitnet
WORKDIR /home/bitnet

# Environment variables
ENV RUST_LOG=info
ENV BITNET_MODEL_PATH=/models
ENV BITNET_DATA_PATH=/data

# Expose server port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD bitnet --version || exit 1

# Default command
CMD ["bitnet", "--help"]

# GPU-enabled runtime stage
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04 AS runtime-gpu

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash bitnet

# Copy binary from builder
COPY --from=builder /app/target/release/bitnet /usr/local/bin/bitnet
COPY --from=builder /app/target/release/bitnet-server /usr/local/bin/bitnet-server

# Create directories for models and data
RUN mkdir -p /data /models && \
    chown -R bitnet:bitnet /data /models

# Switch to non-root user
USER bitnet
WORKDIR /home/bitnet

# Environment variables
ENV RUST_LOG=info
ENV BITNET_MODEL_PATH=/models
ENV BITNET_DATA_PATH=/data
ENV CUDA_VISIBLE_DEVICES=0

# Expose server port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD bitnet --version || exit 1

# Default command
CMD ["bitnet", "--help"]