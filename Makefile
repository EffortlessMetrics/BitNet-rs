# BitNet-rs Makefile - One-click everything
# Usage: make [target]

.PHONY: help all quick install dev test bench clean gpu docker run serve repl release deploy update fmt lint check fix docs ci setup

# Default target
.DEFAULT_GOAL := quick

# Colors
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Detect OS and features
OS := $(shell uname -s | tr '[:upper:]' '[:lower:]')
ARCH := $(shell uname -m)
FEATURES := cpu

# Check for GPU
ifeq ($(shell which nvidia-smi 2>/dev/null),)
	ifeq ($(OS),darwin)
		ifeq ($(ARCH),arm64)
			FEATURES := gpu
		endif
	endif
else
	FEATURES := gpu
endif

# Number of parallel jobs
JOBS := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

#############################################################################
# PRIMARY TARGETS - THE ONES YOU'LL USE MOST
#############################################################################

## help: Show this help message
help:
	@echo "$(BLUE)BitNet-rs One-Click Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Primary Commands:$(NC)"
	@grep -E '^## ' Makefile | sed 's/## /  make /' | column -t -s ':'
	@echo ""
	@echo "$(YELLOW)Quick Examples:$(NC)"
	@echo "  make              # Quick start (builds and tests)"
	@echo "  make run          # Run the CLI"
	@echo "  make test         # Run all tests"
	@echo "  make bench        # Run benchmarks"
	@echo ""

## quick: One-click quick start (default)
quick:
	@echo "$(GREEN)🚀 Quick Start$(NC)"
	@./deploy.sh quick

## all: Build everything with all features
all:
	@echo "$(GREEN)Building everything...$(NC)"
	@cargo build --workspace --all-features --release
	@echo "$(GREEN)✓ Build complete$(NC)"

## install: Full installation with all dependencies
install:
	@echo "$(GREEN)Full installation...$(NC)"
	@./deploy.sh full

## dev: Setup development environment
dev:
	@echo "$(GREEN)Setting up development environment...$(NC)"
	@./deploy.sh dev

#############################################################################
# BUILD TARGETS
#############################################################################

## build: Build with detected features
build:
	@echo "$(GREEN)Building with $(FEATURES) features...$(NC)"
	@cargo build --release --no-default-features --features $(FEATURES)

## release: Build optimized release
release:
	@echo "$(GREEN)Building optimized release...$(NC)"
	@RUSTFLAGS="-C target-cpu=native -C lto=fat -C embed-bitcode=yes" \
		cargo build --release --no-default-features --features $(FEATURES)

## docker: Build Docker image
docker:
	@echo "$(GREEN)Building Docker image...$(NC)"
	@docker build -t bitnet-rs:latest .

#############################################################################
# TEST TARGETS
#############################################################################

## test: Run all tests
test:
	@echo "$(GREEN)Running tests...$(NC)"
	@cargo test --workspace --no-default-features --features $(FEATURES)

## test-quick: Run quick tests only
test-quick:
	@cargo test --workspace --lib --no-default-features --features $(FEATURES)

## test-gpu: Run GPU tests (if available)
test-gpu:
	@echo "$(GREEN)Running GPU tests...$(NC)"
	@cargo run -p xtask -- gpu-smoke

## test-integration: Run integration tests
test-integration:
	@cargo test --workspace --test '*' --no-default-features --features $(FEATURES)

## bench: Run benchmarks
bench:
	@echo "$(GREEN)Running benchmarks...$(NC)"
	@cargo bench --workspace --no-default-features --features $(FEATURES)

#############################################################################
# RUN TARGETS
#############################################################################

## run: Run the CLI
run:
	@cargo run --release --no-default-features --features $(FEATURES) -- $(ARGS)

## serve: Start the server
serve:
	@cargo run --release -p bitnet-server --no-default-features --features $(FEATURES)

## repl: Start interactive REPL
repl:
	@cargo run --release --no-default-features --features $(FEATURES) -- repl

## demo: Run demos
demo:
	@cargo run -p xtask -- demo --which all

#############################################################################
# DEVELOPMENT TARGETS
#############################################################################

## fmt: Format all code
fmt:
	@echo "$(GREEN)Formatting code...$(NC)"
	@cargo fmt --all
	@echo "$(GREEN)✓ Code formatted$(NC)"

## lint: Run clippy lints
lint:
	@echo "$(GREEN)Running clippy...$(NC)"
	@cargo clippy --workspace --all-targets --all-features -- -D warnings

## check: Run all checks (fmt, lint, test)
check: fmt lint test
	@echo "$(GREEN)✓ All checks passed$(NC)"

## fix: Auto-fix issues
fix:
	@echo "$(GREEN)Auto-fixing issues...$(NC)"
	@cargo fix --workspace --allow-dirty --allow-staged
	@cargo fmt --all
	@cargo clippy --workspace --fix --allow-dirty --allow-staged -- -D warnings

## docs: Generate and open documentation
docs:
	@echo "$(GREEN)Generating documentation...$(NC)"
	@cargo doc --workspace --no-deps --open

#############################################################################
# GPU TARGETS
#############################################################################

## gpu: Check GPU availability
gpu:
	@cargo run -p xtask -- gpu-preflight

## gpu-smoke: Run GPU smoke tests
gpu-smoke:
	@cargo run -p xtask -- gpu-smoke

#############################################################################
# MAINTENANCE TARGETS
#############################################################################

## clean: Clean all build artifacts
clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	@cargo clean
	@rm -rf target/
	@echo "$(GREEN)✓ Clean complete$(NC)"

## update: Update all dependencies
update:
	@echo "$(GREEN)Updating dependencies...$(NC)"
	@cargo update
	@rustup update
	@echo "$(GREEN)✓ Updates complete$(NC)"

## setup: Initial setup
setup:
	@echo "$(GREEN)Running initial setup...$(NC)"
	@./deploy.sh full

#############################################################################
# CI/CD TARGETS
#############################################################################

## ci: Run CI checks
ci:
	@echo "$(GREEN)Running CI checks...$(NC)"
	@cargo fmt --all -- --check
	@cargo clippy --workspace --all-targets --all-features -- -D warnings
	@cargo test --workspace --no-default-features --features cpu
	@echo "$(GREEN)✓ CI checks passed$(NC)"

## deploy: Deploy to production
deploy:
	@echo "$(GREEN)Deploying to production...$(NC)"
	@./deploy.sh prod

#############################################################################
# UTILITY TARGETS
#############################################################################

## download-model: Download BitNet model
download-model:
	@cargo run -p xtask -- download-model

## crossval: Run cross-validation tests
crossval:
	@cargo run -p xtask -- full-crossval

## tree: Show project structure
tree:
	@tree -I 'target|venv|__pycache__|.git|node_modules' -L 3

## loc: Count lines of code
loc:
	@tokei --exclude target --exclude venv

## size: Show binary sizes
size:
	@du -h target/release/bitnet* 2>/dev/null | sort -h || echo "No release binaries built yet"

#############################################################################
# SHORTCUTS
#############################################################################

# Single letter shortcuts for common commands
b: build
t: test
r: run
c: clean
f: fmt
l: lint
d: docs
g: gpu

# Quick compound commands
bt: build test
bf: build fmt
cf: clean fmt
cb: clean build
ct: clean test
fr: fmt run
ft: fmt test

# Even quicker
q: quick
a: all
i: install

#############################################################################
# SPECIAL TARGETS
#############################################################################

## watch: Watch for changes and rebuild
watch:
	@cargo watch -x 'build --release --no-default-features --features $(FEATURES)'

## flame: Generate flamegraph (requires cargo-flamegraph)
flame:
	@cargo flamegraph --release --no-default-features --features $(FEATURES)

## audit: Security audit
audit:
	@cargo audit

## outdated: Check for outdated dependencies
outdated:
	@cargo outdated

## bloat: Analyze binary bloat
bloat:
	@cargo bloat --release --no-default-features --features $(FEATURES)

#############################################################################
# DOCKER SHORTCUTS
#############################################################################

## docker-run: Run in Docker
docker-run: docker
	@docker run -it --rm bitnet-rs:latest

## docker-gpu: Run with GPU support in Docker
docker-gpu: docker
	@docker run -it --rm --gpus all bitnet-rs:latest

#############################################################################
# ADVANCED TARGETS
#############################################################################

## profile: Profile with perf (Linux only)
profile:
	@cargo build --release --no-default-features --features $(FEATURES)
	@perf record -g target/release/bitnet
	@perf report

## valgrind: Run with valgrind (Linux only)
valgrind:
	@cargo build --release --no-default-features --features $(FEATURES)
	@valgrind --leak-check=full --show-leak-kinds=all target/release/bitnet

## heaptrack: Memory profiling (requires heaptrack)
heaptrack:
	@cargo build --release --no-default-features --features $(FEATURES)
	@heaptrack target/release/bitnet

#############################################################################
# EXPERIMENTAL
#############################################################################

## wasm: Build WebAssembly target
wasm:
	@cargo build --target wasm32-unknown-unknown -p bitnet-wasm

## python: Build Python bindings
python:
	@cd crates/bitnet-py && maturin develop

#############################################################################
# HELP UTILITIES
#############################################################################

## list: List all targets
list:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | grep -v Makefile

## verbose: Run with verbose output
verbose:
	@RUST_LOG=debug cargo run --release --no-default-features --features $(FEATURES)

.SILENT: help list