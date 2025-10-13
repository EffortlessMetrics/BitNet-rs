# WebAssembly Examples

This directory contains examples for running BitNet.rs in web browsers and WebAssembly environments.

## Examples

- **`browser/`** - Browser-based neural network inference
- **`pwa/`** - Progressive Web App with offline inference

## Running Examples

### Browser Example
```bash
# Build for browser target
cargo build --target wasm32-unknown-unknown -p bitnet-wasm --no-default-features --features browser

# Serve locally (requires basic HTTP server)
cd examples/wasm/browser
python -m http.server 8000
```

### PWA Example
```bash
# Build PWA bundle
cargo build --target wasm32-unknown-unknown -p bitnet-wasm --no-default-features --features browser,inference

# Serve PWA
cd examples/wasm/pwa
npm install && npm start
```

## Prerequisites

- Rust with `wasm32-unknown-unknown` target installed
- `wasm-pack` for building WebAssembly packages
- Basic HTTP server for local development
- Modern web browser with WebAssembly support
- For PWA: Node.js and npm for dependency management

## Features

- Client-side neural network inference
- Offline model loading and caching
- WebWorker support for non-blocking inference
- Progressive enhancement with fallbacks
