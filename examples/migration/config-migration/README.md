# Configuration Migration Example

This example shows how to migrate configuration files from legacy BitNet implementations to BitNet.rs.

## Overview

This migration demonstrates:
- Converting C++ configuration files to Rust TOML format
- Updating Python configuration to work with BitNet.rs
- Migrating server configurations and deployment settings
- Handling environment variables and runtime configuration

## Before: C++ Configuration

### Legacy CMake Configuration
```cmake
# before/CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(bitnet_cpp)

# Model configuration
set(BITNET_MODEL_PATH "/models/bitnet_b1_58-3B.gguf")
set(BITNET_MAX_TOKENS 2048)
set(BITNET_BATCH_SIZE 512)
set(BITNET_THREADS 8)

# Performance settings
set(BITNET_USE_GPU ON)
set(BITNET_GPU_LAYERS 32)
set(BITNET_MEMORY_POOL_SIZE "4GB")

# Server configuration
set(BITNET_SERVER_PORT 8080)
set(BITNET_SERVER_HOST "0.0.0.0")
set(BITNET_LOG_LEVEL "INFO")

# Build configuration
add_executable(bitnet_server src/server.cpp)
target_compile_definitions(bitnet_server PRIVATE
    BITNET_MODEL_PATH="${BITNET_MODEL_PATH}"
    BITNET_MAX_TOKENS=${BITNET_MAX_TOKENS}
    BITNET_BATCH_SIZE=${BITNET_BATCH_SIZE}
)
```

### Legacy JSON Configuration
```json
// before/config.json
{
  "model": {
    "path": "/models/bitnet_b1_58-3B.gguf",
    "max_tokens": 2048,
    "batch_size": 512,
    "context_length": 4096
  },
  "inference": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1
  },
  "performance": {
    "threads": 8,
    "use_gpu": true,
    "gpu_layers": 32,
    "memory_pool": "4GB"
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "max_connections": 100,
    "timeout": 30
  },
  "logging": {
    "level": "INFO",
    "file": "/var/log/bitnet.log",
    "max_size": "100MB"
  }
}
```

## After: BitNet.rs Configuration

### Rust TOML Configuration
```toml
# after/bitnet.toml
[model]
path = "/models/bitnet_b1_58-3B.gguf"
max_tokens = 2048
batch_size = 512
context_length = 4096

[inference]
temperature = 0.7
top_p = 0.9
top_k = 40
repeat_penalty = 1.1
# New Rust-specific optimizations
use_flash_attention = true
rope_scaling = "linear"
quantization = "q4_0"

[performance]
threads = 8
use_gpu = true
gpu_layers = 32
memory_pool = "4GB"
# Rust-specific performance settings
async_inference = true
batch_processing = true
memory_mapping = true

[server]
host = "0.0.0.0"
port = 8080
max_connections = 100
timeout = 30
# Enhanced server features
cors_enabled = true
rate_limiting = true
metrics_enabled = true
health_check_path = "/health"

[logging]
level = "info"
format = "json"
file = "/var/log/bitnet.log"
max_size = "100MB"
rotation = "daily"
# Structured logging with tracing
spans = true
events = true
targets = ["bitnet_server", "bitnet_inference"]

[security]
# New security features
api_key_required = false
tls_enabled = false
cors_origins = ["*"]
max_request_size = "10MB"

[monitoring]
# Built-in monitoring
prometheus_enabled = true
prometheus_port = 9090
jaeger_enabled = false
jaeger_endpoint = "http://localhost:14268"
```

### Environment Configuration
```bash
# after/.env
# Model configuration
BITNET_MODEL_PATH="/models/bitnet_b1_58-3B.gguf"
BITNET_CONFIG_PATH="./bitnet.toml"

# Server configuration
BITNET_SERVER_HOST="0.0.0.0"
BITNET_SERVER_PORT="8080"

# Performance tuning
BITNET_THREADS="8"
BITNET_GPU_LAYERS="32"
BITNET_BATCH_SIZE="512"

# Logging
RUST_LOG="bitnet_server=info,bitnet_inference=debug"
RUST_BACKTRACE="1"

# Monitoring
BITNET_METRICS_ENABLED="true"
BITNET_PROMETHEUS_PORT="9090"
```

### Cargo.toml Configuration
```toml
# after/Cargo.toml
[package]
name = "bitnet-server"
version = "0.1.0"
edition = "2021"

[dependencies]
bitnet-inference = { path = "../../crates/bitnet-inference" }
bitnet-server = { path = "../../crates/bitnet-server" }
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json"] }

[features]
default = ["gpu"]
gpu = ["bitnet-inference/gpu"]
metrics = ["bitnet-server/prometheus"]
tracing = ["bitnet-server/jaeger"]

[[bin]]
name = "bitnet-server"
path = "src/main.rs"

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
```

## Migration Tools

### Configuration Converter Script
```rust
// migration_tool/src/main.rs
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Deserialize)]
struct LegacyConfig {
    model: LegacyModelConfig,
    inference: LegacyInferenceConfig,
    performance: LegacyPerformanceConfig,
    server: LegacyServerConfig,
    logging: LegacyLoggingConfig,
}

#[derive(Serialize)]
struct BitNetConfig {
    model: ModelConfig,
    inference: InferenceConfig,
    performance: PerformanceConfig,
    server: ServerConfig,
    logging: LoggingConfig,
    security: SecurityConfig,
    monitoring: MonitoringConfig,
}

fn convert_config(legacy_path: &Path, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Read legacy JSON config
    let legacy_content = fs::read_to_string(legacy_path)?;
    let legacy_config: LegacyConfig = serde_json::from_str(&legacy_content)?;
    
    // Convert to new format with enhancements
    let bitnet_config = BitNetConfig {
        model: ModelConfig {
            path: legacy_config.model.path,
            max_tokens: legacy_config.model.max_tokens,
            batch_size: legacy_config.model.batch_size,
            context_length: legacy_config.model.context_length,
        },
        inference: InferenceConfig {
            temperature: legacy_config.inference.temperature,
            top_p: legacy_config.inference.top_p,
            top_k: legacy_config.inference.top_k,
            repeat_penalty: legacy_config.inference.repeat_penalty,
            // Add new Rust-specific features
            use_flash_attention: true,
            rope_scaling: "linear".to_string(),
            quantization: "q4_0".to_string(),
        },
        performance: PerformanceConfig {
            threads: legacy_config.performance.threads,
            use_gpu: legacy_config.performance.use_gpu,
            gpu_layers: legacy_config.performance.gpu_layers,
            memory_pool: legacy_config.performance.memory_pool,
            // Add Rust optimizations
            async_inference: true,
            batch_processing: true,
            memory_mapping: true,
        },
        server: ServerConfig {
            host: legacy_config.server.host,
            port: legacy_config.server.port,
            max_connections: legacy_config.server.max_connections,
            timeout: legacy_config.server.timeout,
            // Add new server features
            cors_enabled: true,
            rate_limiting: true,
            metrics_enabled: true,
            health_check_path: "/health".to_string(),
        },
        logging: LoggingConfig {
            level: legacy_config.logging.level.to_lowercase(),
            format: "json".to_string(),
            file: legacy_config.logging.file,
            max_size: legacy_config.logging.max_size,
            rotation: "daily".to_string(),
            // Add structured logging
            spans: true,
            events: true,
            targets: vec!["bitnet_server".to_string(), "bitnet_inference".to_string()],
        },
        security: SecurityConfig {
            api_key_required: false,
            tls_enabled: false,
            cors_origins: vec!["*".to_string()],
            max_request_size: "10MB".to_string(),
        },
        monitoring: MonitoringConfig {
            prometheus_enabled: true,
            prometheus_port: 9090,
            jaeger_enabled: false,
            jaeger_endpoint: "http://localhost:14268".to_string(),
        },
    };
    
    // Write new TOML config
    let toml_content = toml::to_string_pretty(&bitnet_config)?;
    fs::write(output_path, toml_content)?;
    
    println!("Configuration migrated successfully!");
    println!("Legacy config: {}", legacy_path.display());
    println!("New config: {}", output_path.display());
    
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <legacy_config.json> <output_config.toml>", args[0]);
        std::process::exit(1);
    }
    
    let legacy_path = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);
    
    if let Err(e) = convert_config(legacy_path, output_path) {
        eprintln!("Error converting config: {}", e);
        std::process::exit(1);
    }
}
```

### Usage Script
```bash
#!/bin/bash
# migrate_config.sh

set -e

echo "BitNet Configuration Migration Tool"
echo "=================================="

# Check if legacy config exists
if [ ! -f "config.json" ]; then
    echo "Error: Legacy config.json not found"
    exit 1
fi

# Backup original config
cp config.json config.json.backup
echo "✓ Backed up original config to config.json.backup"

# Convert configuration
cargo run --bin config-converter config.json bitnet.toml
echo "✓ Converted config.json to bitnet.toml"

# Generate environment file
echo "Generating .env file..."
cat > .env << EOF
# Generated from legacy configuration
BITNET_CONFIG_PATH="./bitnet.toml"
RUST_LOG="bitnet_server=info,bitnet_inference=debug"
RUST_BACKTRACE="1"
BITNET_METRICS_ENABLED="true"
EOF
echo "✓ Generated .env file"

# Update Cargo.toml if needed
if [ ! -f "Cargo.toml" ]; then
    echo "Creating Cargo.toml..."
    cat > Cargo.toml << 'EOF'
[package]
name = "bitnet-app"
version = "0.1.0"
edition = "2021"

[dependencies]
bitnet-inference = { path = "../../crates/bitnet-inference" }
bitnet-server = { path = "../../crates/bitnet-server" }
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"

[features]
default = ["gpu"]
gpu = ["bitnet-inference/gpu"]
EOF
    echo "✓ Created Cargo.toml"
fi

echo ""
echo "Migration completed successfully!"
echo ""
echo "Next steps:"
echo "1. Review bitnet.toml and adjust settings as needed"
echo "2. Update your application code to use the new config format"
echo "3. Test the new configuration: cargo run"
echo "4. Remove legacy files when satisfied: rm config.json.backup"
```

## Key Configuration Changes

### 1. Format Change
- **Before**: JSON configuration files
- **After**: TOML configuration (more readable, better for Rust)

### 2. Enhanced Features
- **Security settings**: API keys, CORS, TLS configuration
- **Monitoring**: Built-in Prometheus and Jaeger support
- **Structured logging**: JSON format with tracing spans
- **Performance optimizations**: Async inference, memory mapping

### 3. Environment Integration
- **Environment variables**: Better 12-factor app compliance
- **Runtime configuration**: Dynamic configuration updates
- **Feature flags**: Conditional compilation support

### 4. Deployment Improvements
- **Docker-friendly**: Environment-based configuration
- **Kubernetes-ready**: ConfigMap and Secret integration
- **Health checks**: Built-in health and readiness endpoints

## Migration Checklist

- [ ] **Backup original configuration files**
- [ ] **Run configuration converter tool**
- [ ] **Review generated TOML configuration**
- [ ] **Update environment variables**
- [ ] **Test new configuration format**
- [ ] **Update deployment scripts**
- [ ] **Verify monitoring and logging**
- [ ] **Update documentation**

## Common Migration Issues

### Issue: Missing Configuration Keys
```toml
# Add missing required keys
[model]
path = "/path/to/model.gguf"  # Required
```

### Issue: Invalid TOML Syntax
```toml
# Wrong: JSON-style arrays
cors_origins = ["*"]

# Correct: TOML arrays
cors_origins = ["*"]
```

### Issue: Environment Variable Conflicts
```bash
# Check for conflicts
env | grep BITNET

# Clear old variables
unset OLD_BITNET_VAR
```

---

**Configuration migrated!** Your BitNet.rs application now uses modern, type-safe configuration with enhanced features.