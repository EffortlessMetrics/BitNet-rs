# syntax=docker/dockerfile:1.6
# Multi-stage Docker build for BitNet-rs (CPU + CUDA)
# Supports both CPU and GPU backends

# CPU builder
FROM rust:1.89-bookworm AS builder-cpu

# Install sccache for faster rebuilds
RUN cargo install sccache
ENV RUSTC_WRAPPER=sccache

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

# Git metadata build args (injected at build time)
ARG VCS_REF
ARG VCS_BRANCH
ARG VCS_DESCRIBE
ENV VERGEN_GIT_SHA=${VCS_REF:-unknown} \
    VERGEN_GIT_BRANCH=${VCS_BRANCH:-unknown} \
    VERGEN_GIT_DESCRIBE=${VCS_DESCRIBE:-unknown} \
    VERGEN_IDEMPOTENT=1

# Leverage BuildKit caches for faster rebuilds
ARG FEATURES=cpu
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --locked --release --no-default-features --features ${FEATURES}

# CUDA builder (CUDA toolchain + Rust)
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 AS builder-gpu
RUN apt-get update && apt-get install -y curl build-essential pkg-config libssl-dev ca-certificates git && \
    rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.89.0
ENV PATH=/root/.cargo/bin:$PATH
# Install sccache for faster rebuilds
RUN /root/.cargo/bin/cargo install sccache
ENV RUSTC_WRAPPER=sccache
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/
COPY crossval/ ./crossval/
COPY tests/ ./tests/
COPY xtask/ ./xtask/
COPY src/ ./src/
COPY build.rs ./

# Git metadata build args (injected at build time)
ARG VCS_REF
ARG VCS_BRANCH
ARG VCS_DESCRIBE
ENV VERGEN_GIT_SHA=${VCS_REF:-unknown} \
    VERGEN_GIT_BRANCH=${VCS_BRANCH:-unknown} \
    VERGEN_GIT_DESCRIBE=${VCS_DESCRIBE:-unknown} \
    VERGEN_IDEMPOTENT=1

ARG FEATURES=gpu
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --locked --release --no-default-features --features ${FEATURES}

# Runtime stage - minimal image
FROM debian:bookworm-slim AS runtime

# Git metadata for container labels
ARG VCS_REF
ARG VCS_BRANCH
ARG VCS_DESCRIBE

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with specific UID/GID for security
RUN groupadd -g 10001 bitnet && \
    useradd -m -u 10001 -g 10001 -s /bin/bash bitnet

# Copy binary from CPU builder
COPY --from=builder-cpu /app/target/release/bitnet /usr/local/bin/bitnet
COPY --from=builder-cpu /app/target/release/bitnet-server /usr/local/bin/bitnet-server

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

# OCI standard labels for container registries
LABEL org.opencontainers.image.revision=${VCS_REF:-unknown} \
      org.opencontainers.image.version=${VCS_DESCRIBE:-unknown} \
      org.opencontainers.image.ref.name=${VCS_BRANCH:-unknown} \
      org.opencontainers.image.source="https://github.com/bitnet-io/bitnet-rs" \
      org.opencontainers.image.title="BitNet.rs" \
      org.opencontainers.image.description="High-performance 1-bit LLM inference engine"

# Default command
CMD ["bitnet", "--help"]

# GPU-enabled runtime stage
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04 AS runtime-gpu

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with specific UID/GID for security
RUN groupadd -g 10001 bitnet && \
    useradd -m -u 10001 -g 10001 -s /bin/bash bitnet

# Copy binary from GPU builder
COPY --from=builder-gpu /app/target/release/bitnet /usr/local/bin/bitnet
COPY --from=builder-gpu /app/target/release/bitnet-server /usr/local/bin/bitnet-server

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