# BitNet-rs Cross-Validation Test Environment
FROM rust:1.89-slim

# Install system dependencies including C++ build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    git \
    bc \
    procps \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for validation scripts
RUN pip3 install --no-cache-dir numpy scipy

# Set up Rust environment
RUN rustup component add clippy rustfmt

# Create workspace
WORKDIR /workspace

# Pre-build dependencies including FFI
COPY Cargo.toml Cargo.lock ./
COPY crates/*/Cargo.toml ./crates/
COPY crossval/Cargo.toml ./crossval/
COPY xtask/Cargo.toml ./xtask/
RUN mkdir -p src crossval/src xtask/src && \
    echo "fn main() {}" > src/main.rs && \
    echo "fn main() {}" > crossval/src/lib.rs && \
    echo "fn main() {}" > xtask/src/main.rs && \
    find crates -name "Cargo.toml" -exec dirname {} \; | \
    while read dir; do mkdir -p "$dir/src" && echo "fn main() {}" > "$dir/src/lib.rs"; done && \
    cargo build --workspace --no-default-features --features "cpu,ffi,crossval" && \
    rm -rf src crossval/src xtask/src crates/*/src

# Set concurrency environment defaults
ENV RUST_TEST_THREADS=2
ENV RAYON_NUM_THREADS=2
ENV CROSSVAL_WORKERS=2
ENV BITNET_DETERMINISTIC=1
ENV BITNET_SEED=42
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

CMD ["bash"]