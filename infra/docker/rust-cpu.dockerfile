# BitNet-rs CPU Test Environment
FROM rust:1.89-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    bc \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set up Rust environment
RUN rustup component add clippy rustfmt

# Create workspace
WORKDIR /workspace

# Pre-build common dependencies (speeds up subsequent builds)
COPY Cargo.toml Cargo.lock ./
COPY crates/*/Cargo.toml ./crates/
RUN mkdir -p src && echo "fn main() {}" > src/main.rs && \
    find crates -name "Cargo.toml" -exec dirname {} \; | \
    while read dir; do mkdir -p "$dir/src" && echo "fn main() {}" > "$dir/src/lib.rs"; done && \
    cargo build --workspace --no-default-features --features cpu && \
    rm -rf src crates/*/src

# Set concurrency environment defaults
ENV RUST_TEST_THREADS=2
ENV RAYON_NUM_THREADS=2
ENV BITNET_DETERMINISTIC=1
ENV BITNET_SEED=42

CMD ["bash"]