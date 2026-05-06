FROM rust:1.92-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    g++ \
    jq \
    libssl-dev \
    make \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN rustup component add llvm-tools-preview

RUN cargo install cargo-llvm-cov --locked && \
    cargo install cargo-nextest --locked
