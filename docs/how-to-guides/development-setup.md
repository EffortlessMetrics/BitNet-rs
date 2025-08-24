# How to Set Up Your Development Environment

This guide explains how to set up your development environment for BitNet.rs.

## System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Architecture**: x86_64 or aarch64

## 1. Install Rust

If you don't have Rust installed, you can install it with the following command:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

If you already have Rust installed, make sure it's up to date:

```bash
rustup update stable
```

## 2. Install Rust Components

Install the following Rust components:

```bash
rustup component add rustfmt clippy llvm-tools-preview
```

## 3. Install Development Tools

Install the following development tools using `cargo`:

```bash
cargo install cargo-audit
cargo install cargo-deny
cargo install cargo-machete
cargo install cargo-outdated
cargo install cargo-llvm-cov
cargo install cargo-criterion
cargo install cargo-expand
cargo install cargo-watch
cargo install cargo-edit
```

## 4. Install System Dependencies

### Linux (apt)

```bash
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev
```

### Linux (yum)

```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y openssl-devel pkg-config
```

### Linux (pacman)

```bash
sudo pacman -S --needed base-devel openssl pkg-config
```

### macOS

```bash
brew install pkg-config openssl
```

If you don't have Homebrew installed, you can install it from [brew.sh](https://brew.sh).

### Windows

On Windows, you'll need to install:

- Visual Studio Build Tools or Visual Studio with C++ support
- Git for Windows

## 5. Set Up Git Hooks

This step is optional, but recommended. To set up Git hooks, run the following command from the root of the repository:

```bash
./scripts/dev-setup.sh --skip-rust --skip-tools --skip-system --skip-ide --skip-verify
```

This will create a pre-commit hook that runs formatting, clippy, and quick tests before each commit.

## 6. Verify Your Setup

To verify your setup, run the following command from the root of the repository:

```bash
cargo check --workspace --features cpu
```

If the command runs successfully, your development environment is ready!
