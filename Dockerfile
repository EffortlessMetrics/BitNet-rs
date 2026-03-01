# syntax=docker/dockerfile:1.6
# Multi-stage Docker build for BitNet-rs (CPU + CUDA)
# Uses cargo-chef for dependency caching.
#
# Build targets:
#   docker build --target runtime     . # CPU (default)
#   docker build --target runtime-gpu . # CUDA GPU

# ── Stage 1: cargo-chef planner (shared) ────────────────────────────
FROM rust:1.92-bookworm AS planner
RUN cargo install cargo-chef --locked
WORKDIR /app
COPY Cargo.toml Cargo.lock build.rs ./
COPY src/ ./src/
COPY crates/ ./crates/
COPY crossval/ ./crossval/
COPY tests/ ./tests/
COPY xtask/ ./xtask/
COPY xtask-build-helper/ ./xtask-build-helper/
COPY benches/ ./benches/
COPY fuzz/ ./fuzz/
RUN cargo chef prepare --recipe-path recipe.json

# ── Stage 2a: CPU dependency cook ───────────────────────────────────
FROM rust:1.92-bookworm AS deps-cpu
RUN cargo install cargo-chef --locked
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=planner /app/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,id=cpu-target,target=/app/target \
    cargo chef cook --release --no-default-features --features cpu --recipe-path recipe.json

# ── Stage 2b: CUDA dependency cook ─────────────────────────────────
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 AS deps-gpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential cmake pkg-config libssl-dev ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.92.0
ENV PATH=/root/.cargo/bin:$PATH
RUN cargo install cargo-chef --locked
WORKDIR /app
COPY --from=planner /app/recipe.json recipe.json
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,id=gpu-target,target=/app/target \
    cargo chef cook --release --no-default-features --features gpu --recipe-path recipe.json

# ── Stage 3a: CPU builder ──────────────────────────────────────────
FROM deps-cpu AS builder-cpu

ARG VCS_REF
ARG VCS_BRANCH
ARG VCS_DESCRIBE
ENV VERGEN_GIT_SHA=${VCS_REF:-unknown} \
    VERGEN_GIT_BRANCH=${VCS_BRANCH:-unknown} \
    VERGEN_GIT_DESCRIBE=${VCS_DESCRIBE:-unknown} \
    VERGEN_IDEMPOTENT=1

COPY Cargo.toml Cargo.lock build.rs ./
COPY src/ ./src/
COPY crates/ ./crates/
COPY crossval/ ./crossval/
COPY tests/ ./tests/
COPY xtask/ ./xtask/
COPY xtask-build-helper/ ./xtask-build-helper/
COPY benches/ ./benches/
COPY fuzz/ ./fuzz/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,id=cpu-target,target=/app/target \
    cargo build --locked --release --no-default-features --features cpu \
    && cp target/release/bitnet target/release/server /usr/local/bin/

# ── Stage 3b: CUDA builder ─────────────────────────────────────────
FROM deps-gpu AS builder-gpu

ARG VCS_REF
ARG VCS_BRANCH
ARG VCS_DESCRIBE
ENV VERGEN_GIT_SHA=${VCS_REF:-unknown} \
    VERGEN_GIT_BRANCH=${VCS_BRANCH:-unknown} \
    VERGEN_GIT_DESCRIBE=${VCS_DESCRIBE:-unknown} \
    VERGEN_IDEMPOTENT=1

COPY Cargo.toml Cargo.lock build.rs ./
COPY src/ ./src/
COPY crates/ ./crates/
COPY crossval/ ./crossval/
COPY tests/ ./tests/
COPY xtask/ ./xtask/
COPY xtask-build-helper/ ./xtask-build-helper/
COPY benches/ ./benches/
COPY fuzz/ ./fuzz/
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,id=gpu-target,target=/app/target \
    cargo build --locked --release --no-default-features --features gpu \
    && cp target/release/bitnet target/release/server /usr/local/bin/

# ── Stage 4a: CPU runtime (minimal) ────────────────────────────────
FROM debian:bookworm-slim AS runtime

ARG VCS_REF
ARG VCS_BRANCH
ARG VCS_DESCRIBE

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 10001 bitnet && \
    useradd -m -u 10001 -g 10001 -s /bin/false bitnet

COPY --from=builder-cpu /usr/local/bin/bitnet  /usr/local/bin/bitnet
COPY --from=builder-cpu /usr/local/bin/server  /usr/local/bin/server

RUN mkdir -p /data /models && chown bitnet:bitnet /data /models

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
      org.opencontainers.image.source="https://github.com/EffortlessMetrics/BitNet-rs" \
      org.opencontainers.image.title="bitnet-rs" \
      org.opencontainers.image.description="High-performance 1-bit LLM inference engine"

CMD ["server"]

# ── Stage 4b: GPU runtime (CUDA libs included) ─────────────────────
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04 AS runtime-gpu

ARG VCS_REF
ARG VCS_BRANCH
ARG VCS_DESCRIBE

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 10001 bitnet && \
    useradd -m -u 10001 -g 10001 -s /bin/false bitnet

COPY --from=builder-gpu /usr/local/bin/bitnet  /usr/local/bin/bitnet
COPY --from=builder-gpu /usr/local/bin/server  /usr/local/bin/server

RUN mkdir -p /data /models && chown bitnet:bitnet /data /models

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
      org.opencontainers.image.source="https://github.com/EffortlessMetrics/BitNet-rs" \
      org.opencontainers.image.title="bitnet-rs-gpu" \
      org.opencontainers.image.description="High-performance 1-bit LLM inference engine (CUDA)"

CMD ["server"]
