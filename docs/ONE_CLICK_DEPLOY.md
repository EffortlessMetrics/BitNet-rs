# One-Click Deploy Guide

BitNet-rs is designed for **true one-click deployment**. Every operation is optimized for simplicity.

## 🚀 Quick Start (Literally One Command)

```bash
# Clone and deploy in one line
git clone https://github.com/microsoft/BitNet-rs.git && cd BitNet-rs && make

# Or if you already have the repo
make
```

That's it! This will:
- ✅ Detect your environment (CPU/GPU)
- ✅ Install dependencies
- ✅ Build optimized binaries
- ✅ Download models
- ✅ Run tests
- ✅ Start the service

## 📋 One-Click Commands

### Using Make (Simplest)

```bash
make              # Quick start - builds and tests everything
make run          # Run the CLI
make serve        # Start the server
make test         # Run all tests
make bench        # Run benchmarks
make gpu          # Check GPU availability
make docker       # Build and run in Docker
make deploy       # Deploy to production
```

### Even Shorter Shortcuts

```bash
make q            # Quick start
make r            # Run
make t            # Test
make b            # Build
make c            # Clean
make f            # Format
make l            # Lint
make d            # Docs
```

### Using the Deploy Script

```bash
./deploy.sh       # Interactive menu
./deploy.sh quick # Quick start
./deploy.sh full  # Full installation
./deploy.sh prod  # Production deployment
```

## 🐳 Docker One-Click

### CPU Version
```bash
docker-compose up
```

### GPU Version
```bash
docker-compose --profile gpu up
```

### Production with Monitoring
```bash
docker-compose --profile production --profile monitoring up
```

## ☁️ Cloud Deployment

### AWS (One-Click with CloudFormation)
```bash
aws cloudformation create-stack \
  --stack-name bitnet-rs \
  --template-body file://aws/cloudformation.yaml
```

### Google Cloud (One-Click with gcloud)
```bash
gcloud run deploy bitnet-rs \
  --source . \
  --platform managed \
  --allow-unauthenticated
```

### Azure (One-Click with Azure CLI)
```bash
az webapp up \
  --name bitnet-rs \
  --runtime "DOCKER|bitnet-rs:latest"
```

### Kubernetes (One-Click with Helm)
```bash
helm install bitnet-rs ./charts/bitnet-rs
```

## 🎯 Common Tasks - All One-Click

### Development Setup
```bash
make dev
```
Sets up:
- Git hooks
- VS Code settings
- Rust toolchain
- Development dependencies

### Run Tests
```bash
make test        # All tests
make test-quick  # Quick tests only
make test-gpu    # GPU tests
```

### Benchmarking
```bash
make bench       # Run benchmarks
make flame       # Generate flamegraph
make profile     # Profile with perf
```

### Code Quality
```bash
make check       # Format, lint, and test
make fix         # Auto-fix all issues
```

### Documentation
```bash
make docs        # Generate and open docs
```

### Cleanup
```bash
make clean       # Clean build artifacts
make update      # Update all dependencies
```

### Docker Builds
For faster Docker builds with BuildKit caching:
```bash
# Enable BuildKit for faster builds with cache mounts
export DOCKER_BUILDKIT=1

# Build CPU version
docker build --target runtime -t bitnet:cpu .

# Build GPU version  
docker build --target runtime-gpu -t bitnet:gpu .

# Use docker-compose (BuildKit enabled automatically)
docker compose up --build bitnet-cpu
docker compose --profile gpu up --build bitnet-gpu

# Optional: Control sccache cache size (default: 10G)
export SCCACHE_CACHE_SIZE=20G  # Adjust based on available disk
docker compose up --build bitnet-cpu
```

## 🔧 Environment Detection

The system automatically detects:
- **OS**: Linux, macOS, Windows (WSL)
- **Architecture**: x86_64, ARM64
- **GPU**: NVIDIA (CUDA), AMD (ROCm), Apple (Metal)
- **CPU**: Core count, SIMD support
- **Memory**: Available RAM
- **Dependencies**: Missing packages

## 📦 Package Managers

### Homebrew (macOS/Linux)
```bash
brew tap microsoft/bitnet
brew install bitnet-rs
```

### Cargo (Rust)
```bash
cargo install bitnet-rs
```

### NPM (Node.js bindings)
```bash
npm install @microsoft/bitnet-rs
```

### Pip (Python bindings)
```bash
pip install bitnet-rs
```

## 🎮 Interactive Mode

Just run `make` or `./deploy.sh` without arguments for an interactive menu:

```
🚀 BitNet-rs One-Click Deploy 🚀
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1) Quick Start (Recommended)
2) Full Installation
3) Development Setup
4) Production Deploy
5) Run Tests
6) Run Benchmarks
7) Clean & Rebuild
8) GPU Check
9) Update Everything
0) Exit

Select option: _
```

## 🔄 CI/CD Integration

### GitHub Actions
```yaml
- name: Deploy BitNet-rs
  run: make deploy
```

### GitLab CI
```yaml
deploy:
  script:
    - make deploy
```

### Jenkins
```groovy
stage('Deploy') {
    steps {
        sh 'make deploy'
    }
}
```

## 📊 Monitoring

One-click monitoring setup:
```bash
make monitoring
```

This starts:
- Prometheus (metrics collection)
- Grafana (visualization)
- Health checks
- Performance dashboards

Access at:
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

## 🆘 Troubleshooting

### If Something Goes Wrong
```bash
make fix         # Auto-fix common issues
make clean       # Clean and restart
make update      # Update everything
```

### Debug Mode
```bash
make verbose     # Run with debug output
RUST_LOG=debug make run
```

### Check System
```bash
make check       # Run all checks
cargo xtask gpu-preflight  # Check GPU
```

## 🎨 Customization

### Custom Features
```bash
FEATURES=gpu make build      # Build with GPU
FEATURES=cpu make build      # Build CPU-only
```

### Custom Flags
```bash
RUSTFLAGS="-C target-cpu=native" make release
```

### Environment Variables
```bash
BITNET_THREADS=8 make run
CUDA_VISIBLE_DEVICES=0,1 make run
```

## 📱 Platform-Specific

### macOS (Apple Silicon)
```bash
make              # Automatically uses Metal
```

### Linux (NVIDIA GPU)
```bash
make              # Automatically uses CUDA if available
```

### Windows (WSL2)
```bash
make              # Works out of the box
```

### Docker (Any Platform)
```bash
docker run -it bitnet-rs/bitnet-rs
```

## 🌐 Web UI

Start the web interface:
```bash
make serve
# Open http://localhost:8080
```

## 📈 Performance

Optimized build:
```bash
make release      # Native CPU optimizations
```

Benchmark:
```bash
make bench        # Run performance tests
```

## 🔐 Security

Security audit:
```bash
make audit        # Check for vulnerabilities
```

## 📝 License

MIT OR Apache-2.0

---

**Remember**: Everything is designed to be one-click. If you need more than one command, we've failed. Please open an issue!