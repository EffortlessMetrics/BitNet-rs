# BitNet-rs GPU Test Environment
FROM nvidia/cuda:12.2-devel-ubuntu22.04

# Install Rust
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    bc \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.89
ENV PATH="/root/.cargo/bin:${PATH}"

# Add Rust components
RUN rustup component add clippy rustfmt

# Set up CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Create workspace
WORKDIR /workspace

# Pre-build dependencies for GPU features
COPY Cargo.toml Cargo.lock ./
COPY crates/*/Cargo.toml ./crates/
RUN mkdir -p src && echo "fn main() {}" > src/main.rs && \
    find crates -name "Cargo.toml" -exec dirname {} \; | \
    while read dir; do mkdir -p "$dir/src" && echo "fn main() {}" > "$dir/src/lib.rs"; done && \
    cargo build --workspace --no-default-features --features cuda && \
    rm -rf src crates/*/src

# Set concurrency environment defaults
ENV RUST_TEST_THREADS=2
ENV RAYON_NUM_THREADS=2
ENV BITNET_DETERMINISTIC=1
ENV BITNET_SEED=42

CMD ["bash"]
