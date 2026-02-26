# Web Integration Examples

This directory contains examples for building web services and APIs with bitnet-rs.

## Examples

- **`axum_server.rs`** - REST API server using Axum framework
- **`warp_server.rs`** - High-performance server using Warp framework
- **`actix_server.rs`** - Web service using Actix-web framework

## Running Examples

```bash
# Axum server
cargo run --example axum_server --no-default-features --features cpu,server

# Warp server
cargo run --example warp_server --no-default-features --features cpu,server

# Actix server
cargo run --example actix_server --no-default-features --features cpu,server
```

## API Endpoints

All servers typically provide:
- `POST /inference` - Run model inference
- `GET /health` - Health check endpoint
- `GET /metrics` - Performance metrics
- `POST /batch` - Batch inference

## Prerequisites

- Basic example prerequisites
- Web framework dependencies (added automatically)
- Network access for server binding
- Optional: Load testing tools for performance evaluation
