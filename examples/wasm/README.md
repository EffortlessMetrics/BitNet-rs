# WebAssembly Examples

This directory contains WebAssembly browser/PWA integration shells for BitNet-rs. Real WASM inference is still a scaffolded lane; see [`docs/wasm/WASM_INFERENCE_LANE.md`](../../docs/wasm/WASM_INFERENCE_LANE.md) for the proof contract and milestone path.

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
# Build PWA bundle. The inference feature is opt-in and currently represents
# scaffolded plumbing, not a completed browser inference proof.
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

## Current capabilities and boundaries

- Browser and PWA integration shells
- Offline model loading/caching UI scaffolding
- WebWorker support scaffolding for non-blocking execution
- Progressive enhancement with fallbacks
- Real inference claims require the receipt-backed milestones in [`docs/wasm/WASM_INFERENCE_LANE.md`](../../docs/wasm/WASM_INFERENCE_LANE.md)
