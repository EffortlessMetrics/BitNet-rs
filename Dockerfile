# syntax=docker/dockerfile:1.6
# Multi-stage Docker build for BitNet-rs (CPU + CUDA)
#
# Build targets:
#   docker build --target runtime     -t bitnet-rs:cpu .   # CPU-only (default)
#   docker build --target runtime-gpu -t bitnet-rs:gpu .   # CUDA-enabled
#
# Uses cargo-chef for dependency caching: source changes only rebuild the
# application, not all ~400 transitive dependencies.

# ── Stage 1: Install cargo-chef ──────────────────────────────────
FROM rust:1.92-bookworm AS chef
RUN cargo install cargo-chef --locked
WORKDIR /app

# ── Stage 2: Capture dependency recipe ───────────────────────────
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ── Stage 3a: CPU builder ────────────────────────────────────────
FROM chef AS builder-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Cook dependencies first (cached until Cargo.toml/Cargo.lock change)
COPY --from=planner /app/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo chef cook --release --no-default-features --features cpu --recipe-path recipe.json

# Copy full source and build application
COPY . .
ARG VCS_REF
ARG VCS_BRANCH
ARG VCS_DESCRIBE
ENV VERGEN_GIT_SHA=${VCS_REF:-unknown} \
    VERGEN_GIT_BRANCH=${VCS_BRANCH:-unknown} \
    VERGEN_GIT_DESCRIBE=${VCS_DESCRIBE:-unknown} \
    VERGEN_IDEMPOTENT=1
RUN cargo build --locked --release --no-default-features --features cpu && \
    cp target/release/bitnet target/release/bitnet-server /usr/local/bin/

# ── Stage 3b: GPU builder (CUDA toolchain + Rust) ───────────────
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 AS builder-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential cmake pkg-config libssl-dev ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.92.0
ENV PATH=/root/.cargo/bin:$PATH
RUN cargo install cargo-chef --locked
WORKDIR /app

# Cook dependencies first (cached until Cargo.toml/Cargo.lock change)
COPY --from=planner /app/recipe.json recipe.json
RUN --mount=type=cache,target=/root/.cargo/registry \
    cargo chef cook --release --no-default-features --features gpu --recipe-path recipe.json

# Copy full source and build application
COPY . .
ARG VCS_REF
ARG VCS_BRANCH
ARG VCS_DESCRIBE
ENV VERGEN_GIT_SHA=${VCS_REF:-unknown} \
    VERGEN_GIT_BRANCH=${VCS_BRANCH:-unknown} \
    VERGEN_GIT_DESCRIBE=${VCS_DESCRIBE:-unknown} \
    VERGEN_IDEMPOTENT=1
RUN cargo build --locked --release --no-default-features --features gpu && \
    cp target/release/bitnet target/release/bitnet-server /usr/local/bin/

# ── Stage 4a: CPU runtime (minimal) ─────────────────────────────
FROM debian:bookworm-slim AS runtime

ARG VCS_REF
ARG VCS_BRANCH
ARG VCS_DESCRIBE

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 10001 bitnet && \
    useradd -m -u 10001 -g 10001 -s /bin/false bitnet

COPY --from=builder-cpu /usr/local/bin/bitnet /usr/local/bin/bitnet
COPY --from=builder-cpu /usr/local/bin/bitnet-server /usr/local/bin/bitnet-server

RUN mkdir -p /data /models && \
    chown -R bitnet:bitnet /data /models

USER bitnet
WORKDIR /home/bitnet

# RUST_LOG            - log verbosity (default: info)
# BITNET_MODEL_PATH   - directory for GGUF model files
# BITNET_DATA_PATH    - writable data directory
ENV RUST_LOG=info \
    BITNET_MODEL_PATH=/models \
    BITNET_DATA_PATH=/data

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

LABEL org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${VCS_DESCRIBE}" \
      org.opencontainers.image.ref.name="${VCS_BRANCH}" \
      org.opencontainers.image.source="https://github.com/bitnet-io/bitnet-rs" \
      org.opencontainers.image.title="bitnet-rs" \
      org.opencontainers.image.description="High-performance 1-bit LLM inference engine"

CMD ["bitnet-server"]

# ── Stage 4b: GPU runtime (CUDA runtime libs) ───────────────────
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04 AS runtime-gpu

ARG VCS_REF
ARG VCS_BRANCH
ARG VCS_DESCRIBE

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 10001 bitnet && \
    useradd -m -u 10001 -g 10001 -s /bin/false bitnet

COPY --from=builder-gpu /usr/local/bin/bitnet /usr/local/bin/bitnet
COPY --from=builder-gpu /usr/local/bin/bitnet-server /usr/local/bin/bitnet-server

RUN mkdir -p /data /models && \
    chown -R bitnet:bitnet /data /models

USER bitnet
WORKDIR /home/bitnet

# CUDA_VISIBLE_DEVICES - restrict visible GPUs (default: all)
# RUST_LOG             - log verbosity (default: info)
# BITNET_MODEL_PATH    - directory for GGUF model files
# BITNET_DATA_PATH     - writable data directory
ENV RUST_LOG=info \
    BITNET_MODEL_PATH=/models \
    BITNET_DATA_PATH=/data \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

LABEL org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${VCS_DESCRIBE}" \
      org.opencontainers.image.ref.name="${VCS_BRANCH}" \
      org.opencontainers.image.source="https://github.com/bitnet-io/bitnet-rs" \
      org.opencontainers.image.title="bitnet-rs" \
      org.opencontainers.image.description="High-performance 1-bit LLM inference engine (CUDA)"

CMD ["bitnet-server"]
