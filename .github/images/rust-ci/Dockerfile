FROM rust:1.95-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    g++ \
    git \
    jq \
    libssl-dev \
    make \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN rustup component add llvm-tools-preview

RUN cargo install cargo-llvm-cov --version 0.8.4 --locked && \
    cargo install cargo-nextest --version 0.9.131 --locked
