# BitNet.rs Package Distribution

This directory contains packaging configurations for distributing BitNet.rs through various package managers and distribution channels.

## 🦀 Primary Distribution: crates.io

BitNet.rs is primarily distributed through [crates.io](https://crates.io), the official Rust package registry:

```bash
# Install CLI and server tools
cargo install bitnet-cli bitnet-server

# Add library to your project
cargo add bitnet
```

### Published Crates

| Crate | Description | crates.io |
|-------|-------------|-----------|
| `bitnet` | Main library crate | [![Crates.io](https://img.shields.io/crates/v/bitnet.svg)](https://crates.io/crates/bitnet) |
| `bitnet-cli` | Command-line interface | [![Crates.io](https://img.shields.io/crates/v/bitnet-cli.svg)](https://crates.io/crates/bitnet-cli) |
| `bitnet-server` | HTTP inference server | [![Crates.io](https://img.shields.io/crates/v/bitnet-server.svg)](https://crates.io/crates/bitnet-server) |
| `bitnet-common` | Common utilities | [![Crates.io](https://img.shields.io/crates/v/bitnet-common.svg)](https://crates.io/crates/bitnet-common) |
| `bitnet-models` | Model definitions | [![Crates.io](https://img.shields.io/crates/v/bitnet-models.svg)](https://crates.io/crates/bitnet-models) |
| `bitnet-quantization` | Quantization algorithms | [![Crates.io](https://img.shields.io/crates/v/bitnet-quantization.svg)](https://crates.io/crates/bitnet-quantization) |
| `bitnet-kernels` | Compute kernels | [![Crates.io](https://img.shields.io/crates/v/bitnet-kernels.svg)](https://crates.io/crates/bitnet-kernels) |
| `bitnet-inference` | Inference engine | [![Crates.io](https://img.shields.io/crates/v/bitnet-inference.svg)](https://crates.io/crates/bitnet-inference) |
| `bitnet-tokenizers` | Text tokenization | [![Crates.io](https://img.shields.io/crates/v/bitnet-tokenizers.svg)](https://crates.io/crates/bitnet-tokenizers) |
| `bitnet-ffi` | C FFI bindings | [![Crates.io](https://img.shields.io/crates/v/bitnet-ffi.svg)](https://crates.io/crates/bitnet-ffi) |
| `bitnet-py` | Python bindings | [![Crates.io](https://img.shields.io/crates/v/bitnet-py.svg)](https://crates.io/crates/bitnet-py) |
| `bitnet-wasm` | WebAssembly bindings | [![Crates.io](https://img.shields.io/crates/v/bitnet-wasm.svg)](https://crates.io/crates/bitnet-wasm) |

## 📦 Binary Releases

Pre-compiled binaries are available for major platforms through GitHub Releases:

### Supported Platforms

| Platform | Architecture | Features | Status |
|----------|-------------|----------|--------|
| Linux | x86_64 (glibc) | CPU, AVX2 | ✅ Available |
| Linux | x86_64 (musl) | CPU, AVX2 | ✅ Available |
| Linux | ARM64 | CPU, NEON | ✅ Available |
| macOS | x86_64 | CPU, AVX2 | ✅ Available |
| macOS | ARM64 (M1/M2) | CPU, NEON | ✅ Available |
| Windows | x86_64 | CPU, AVX2 | ✅ Available |
| Windows | ARM64 | CPU, NEON | ✅ Available |

### Installation Scripts

**Unix/Linux/macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/microsoft/BitNet/main/scripts/install.sh | bash
```

**Windows (PowerShell):**
```powershell
iwr -useb https://raw.githubusercontent.com/microsoft/BitNet/main/scripts/install.ps1 | iex
```

## 🍺 Package Managers

### Homebrew (macOS/Linux)

**Status:** 🚧 In Development

```bash
# Coming soon
brew install bitnet-rs
```

**Configuration:** [`homebrew/bitnet-rs.rb`](homebrew/bitnet-rs.rb)

### Chocolatey (Windows)

**Status:** 🚧 In Development

```powershell
# Coming soon
choco install bitnet-rs
```

**Configuration:** [`chocolatey/bitnet-rs.nuspec`](chocolatey/bitnet-rs.nuspec)

### Snap (Linux)

**Status:** 🚧 In Development

```bash
# Coming soon
snap install bitnet-rs
```

**Configuration:** [`snap/snapcraft.yaml`](snap/snapcraft.yaml)

### APT (Debian/Ubuntu)

**Status:** 📋 Planned

```bash
# Planned
apt install bitnet-rs
```

### DNF (Fedora/RHEL)

**Status:** 📋 Planned

```bash
# Planned
dnf install bitnet-rs
```

### Arch User Repository (AUR)

**Status:** 📋 Planned

```bash
# Planned
yay -S bitnet-rs
```

## 🐳 Container Images

### Docker Hub

**Status:** ✅ Available

```bash
# Pull official image
docker pull bitnet/bitnet-rust:latest

# Run CLI
docker run --rm bitnet/bitnet-rust:latest bitnet-cli --version

# Run server
docker run -p 8080:8080 bitnet/bitnet-rust:latest bitnet-server --port 8080
```

### GitHub Container Registry

**Status:** ✅ Available

```bash
# Pull from GHCR
docker pull ghcr.io/microsoft/bitnet-rust:latest
```

## 📊 Distribution Strategy

### Primary Distribution (Rust Ecosystem)

1. **crates.io** - Main distribution channel for Rust users
2. **docs.rs** - Automatic documentation hosting
3. **GitHub Releases** - Binary releases and source archives

### Secondary Distribution (System Package Managers)

1. **Homebrew** - macOS and Linux users
2. **Chocolatey** - Windows users
3. **Snap** - Universal Linux packages
4. **APT/DNF** - Native Linux distribution packages

### Specialized Distribution

1. **Docker Hub** - Container deployments
2. **Kubernetes Helm Charts** - Orchestrated deployments
3. **Language-specific packages** - PyPI (Python), npm (Node.js)

## 🔄 Release Process

### Automated Release Pipeline

1. **Version Bump** - Update all crate versions consistently
2. **Testing** - Comprehensive test suite across platforms
3. **Binary Building** - Cross-compilation for all targets
4. **crates.io Publishing** - Publish crates in dependency order
5. **GitHub Release** - Create release with binaries and notes
6. **Package Manager Updates** - Update package configurations
7. **Container Images** - Build and push Docker images

### Release Checklist

- [ ] Version consistency across all crates
- [ ] Comprehensive testing (unit, integration, cross-validation)
- [ ] Documentation updates
- [ ] Changelog updates
- [ ] Binary compilation for all platforms
- [ ] Checksum generation and verification
- [ ] crates.io publication
- [ ] GitHub release creation
- [ ] Package manager notifications
- [ ] Container image updates

## 🛠️ Packaging Guidelines

### Version Management

- All crates must have consistent versions
- Use semantic versioning (SemVer)
- Coordinate releases across all packages

### Platform Support

- **Tier 1**: Linux x86_64, macOS x86_64/ARM64, Windows x86_64
- **Tier 2**: Linux ARM64, Windows ARM64
- **Tier 3**: Other architectures on request

### Feature Flags

- **Default**: CPU-only with basic optimizations
- **GPU**: CUDA support for NVIDIA GPUs
- **Full**: All features enabled
- **Minimal**: Bare minimum for embedded use

### Dependencies

- Minimize external dependencies
- Use static linking where possible
- Provide both glibc and musl variants for Linux

## 🔐 Security and Verification

### Checksums

All binary releases include SHA256 checksums for verification:

```bash
# Download and verify
curl -L -O https://github.com/microsoft/BitNet/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS
```

### Code Signing

- **Windows**: Binaries signed with Microsoft certificate
- **macOS**: Binaries signed and notarized
- **Linux**: GPG signatures for package repositories

### Supply Chain Security

- Reproducible builds where possible
- Dependency auditing with `cargo audit`
- Regular security updates
- Vulnerability disclosure process

## 📈 Distribution Metrics

### Success Metrics

- **Download counts** from GitHub Releases
- **Installation counts** from package managers
- **crates.io download statistics**
- **Docker image pull counts**
- **User feedback and adoption**

### Quality Metrics

- **Installation success rate**
- **Platform compatibility**
- **Performance benchmarks**
- **User satisfaction surveys**

## 🤝 Contributing to Packaging

### Adding New Package Managers

1. Create configuration files in appropriate subdirectory
2. Add installation instructions to main documentation
3. Update CI/CD pipeline for automated updates
4. Test installation process thoroughly
5. Submit PR with changes

### Improving Existing Packages

1. Test current package installation
2. Identify issues or improvements
3. Update configuration files
4. Test changes across platforms
5. Submit PR with improvements

### Package Maintenance

- Monitor package manager feedback
- Update configurations for new releases
- Respond to user issues promptly
- Coordinate with package manager maintainers

## 📞 Support

### Package-Specific Issues

- **crates.io**: Use GitHub issues
- **Homebrew**: Check Homebrew documentation
- **Chocolatey**: Use Chocolatey community
- **Snap**: Use Snapcraft forum
- **Docker**: Use GitHub issues

### General Support

- **GitHub Issues**: https://github.com/microsoft/BitNet/issues
- **Discussions**: https://github.com/microsoft/BitNet/discussions
- **Documentation**: https://docs.rs/bitnet

---

**Note**: This packaging strategy prioritizes the Rust implementation as the primary distribution method, with package managers providing convenient access for users who prefer system-level installation.