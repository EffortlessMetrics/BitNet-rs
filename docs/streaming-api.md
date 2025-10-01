# Streaming Generation with Token IDs

BitNet.rs provides comprehensive streaming generation capabilities with real-time token ID access, supporting both library API and Server-Sent Events (SSE) for web applications.

## Core Features

1. **StreamResponse Structure**: Enhanced streaming response with both text and token IDs (added in v0.1.0)
2. **Real-time Generation**: Token-by-token streaming with configurable buffering
3. **Server-Sent Events**: HTTP/1.1 SSE endpoint with JSON token metadata
4. **Error Resilience**: Comprehensive error handling with recovery suggestions
5. **Performance Optimization**: Configurable buffer sizes and flush intervals

## Library API Usage

```rust
use bitnet_inference::{InferenceEngine, GenerationConfig, StreamingConfig};
use futures::StreamExt;

// Create streaming configuration
let streaming_config = StreamingConfig {
    buffer_size: 10,
    flush_interval_ms: 50,
    max_retries: 3,
    token_timeout_ms: 5000,
    cancellable: true,
};

// Generate with streaming
let mut stream = engine.generate_stream_with_config("Explain quantum computing", &config);

while let Some(result) = stream.next().await {
    match result {
        Ok(stream_response) => {
            // Access generated text
            print!("{}", stream_response.text);
            
            // Access token IDs (new in v0.1.0)
            for &token_id in &stream_response.token_ids {
                eprintln!("[DEBUG] Token ID: {}", token_id);
            }
        }
        Err(e) => eprintln!("Stream error: {}", e),
    }
}
```

## Server-Sent Events (SSE) API

Start the BitNet server with streaming support:

```bash
# Start server with model
cargo run -p bitnet-server -- --port 8080 --model model.gguf

# Test streaming endpoint
curl -X POST http://localhost:8080/v1/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing", 
    "max_tokens": 100, 
    "temperature": 0.7,
    "detailed_errors": true
  }' \
  --no-buffer
```

## SSE Response Format

The streaming endpoint returns Server-Sent Events with the following structure:

```javascript
// Token event (generated text + token ID)
event: token
data: {
  "token": "quantum",
  "token_id": 24975,
  "cumulative_time_ms": 150,
  "position": 1
}

// Completion event (final statistics)
event: complete
data: {
  "total_tokens": 95,
  "total_time_ms": 3200,
  "tokens_per_second": 29.69,
  "completed_normally": true,
  "completion_reason": "Generation completed successfully"
}

// Error event (if issues occur)
event: error
data: {
  "error_type": "generation",
  "message": "Token generation failed",
  "recovery_hints": ["Try reducing max_tokens", "Check model compatibility"],
  "tokens_before_error": 15
}
```

## Testing Streaming Functionality

```bash
# Test streaming generation
cargo run --example streaming_generation --no-default-features --features cpu

# Test server streaming
cargo test --no-default-features -p bitnet-server --no-default-features --features cpu streaming

# Test token ID accuracy
cargo test --no-default-features -p bitnet-inference --no-default-features --features cpu test_token_id_streaming

# Test concurrent streaming
cargo test --no-default-features -p bitnet-inference --no-default-features --features cpu test_concurrent_streams

# Validate SSE format
cargo test --no-default-features -p bitnet-server --no-default-features --features cpu sse_token_ids_match_model_outputs
```

## Troubleshooting Streaming Issues

```bash
# Debug streaming with detailed logging
RUST_LOG=bitnet_inference::streaming=debug cargo run --example streaming_generation

# Test with different buffer configurations
cargo run --example streaming_generation -- --buffer-size 5 --flush-interval 25ms

# Validate token ID consistency
cargo test --no-default-features --features cpu -p bitnet-inference test_streaming_token_id_consistency

# Check server streaming health
curl -X GET http://localhost:8080/health
```

## Security Note

Starting with v0.1.0, BitNet.rs uses PyO3 v0.25.1 to resolve CVE-2024-9979 security vulnerability. This affects Python bindings and server components:

```bash
# Verify PyO3 version
cargo tree -p bitnet-py | grep pyo3

# Expected: pyo3 v0.25.1 or later
```