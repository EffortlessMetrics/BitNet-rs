# Multi-stage Dockerfile for BitNet.rs
# Optimized for production deployment with minimal attack surface

# Build stage
FROM rust:1.70-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency manifests
COPY Cargo.toml Cargo.lock ./
COPY crates/*/Cargo.toml ./crates/

# Create dummy source files to cache dependencies
RUN mkdir -p src crates/bitnet-{common,models,quantization,kernels,inference,tokenizers,server,cli,ffi}/src && \
    echo "fn main() {}" > src/main.rs && \
    find crates -name Cargo.toml -exec dirname {} \; | xargs -I {} touch {}/src/lib.rs

# Build dependencies (cached layer)
RUN cargo build --release --features server,cli && \
    rm -rf src crates/*/src

# Copy actual source code
COPY . .

# Build the application
RUN cargo build --release --features server,cli

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r bitnet \
    && useradd -r -g bitnet -s /bin/false bitnet

# Copy binaries from builder
COPY --from=builder /app/target/release/bitnet /usr/local/bin/
COPY --from=builder /app/target/release/server /usr/local/bin/bitnet-server

# Copy configuration and documentation
COPY --from=builder /app/README.md /app/CHANGELOG.md /usr/share/doc/bitnet/
COPY --from=builder /app/LICENSE* /usr/share/doc/bitnet/

# Create directories for data and configuration
RUN mkdir -p /etc/bitnet /var/lib/bitnet /var/log/bitnet && \
    chown -R bitnet:bitnet /var/lib/bitnet /var/log/bitnet

# Create default configuration
RUN cat > /etc/bitnet/server.toml << 'EOF'
[server]
host = "0.0.0.0"
port = 8080
workers = 4

[model]
# Model path should be mounted as volume
path = "/var/lib/bitnet/model.gguf"

[logging]
level = "info"
format = "json"

[metrics]
enabled = true
endpoint = "/metrics"
EOF

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Switch to non-root user
USER bitnet

# Expose port
EXPOSE 8080

# Set environment variables
ENV RUST_LOG=info
ENV BITNET_CONFIG=/etc/bitnet/server.toml

# Default command
CMD ["bitnet-server", "--config", "/etc/bitnet/server.toml"]

# Labels for metadata
LABEL org.opencontainers.image.title="BitNet.rs"
LABEL org.opencontainers.image.description="High-performance Rust implementation of BitNet 1-bit LLM inference"
LABEL org.opencontainers.image.vendor="BitNet Contributors"
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/microsoft/BitNet"
LABEL org.opencontainers.image.documentation="https://docs.rs/bitnet"