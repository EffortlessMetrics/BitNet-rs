# BitNet-rs Makefile - One-click everything
# Usage: make [target]

# Default goal & common vars
.DEFAULT_GOAL := quick
CARGO ?= cargo

# Pretty colors (safe fallbacks if not a TTY)
GREEN := $(shell tput setaf 2 2>/dev/null || echo "")
YELLOW := $(shell tput setaf 3 2>/dev/null || echo "")
BLUE := $(shell tput setaf 4 2>/dev/null || echo "")
RED := $(shell tput setaf 1 2>/dev/null || echo "")
NC := $(shell tput sgr0 2>/dev/null || echo "")

# Declare phony targets
.PHONY: help all quick install dev test bench clean gpu docker run serve repl release deploy update fmt lint check fix docs ci setup \
        build test-quick test-gpu test-integration gpu-smoke download-model crossval tree loc size \
        watch flame audit outdated bloat docker-run docker-gpu profile valgrind heaptrack wasm python list verbose \
        b t r c f l d g bt bf cf cb ct fr ft q a i

# Detect OS and features
OS := $(shell uname -s | tr '[:upper:]' '[:lower:]')
ARCH := $(shell uname -m)

# Check for GPU availability
GPU_AVAILABLE := $(shell command -v nvidia-smi 2> /dev/null)
ifeq ($(GPU_AVAILABLE),)
  ifeq ($(OS),darwin)
    ifeq ($(ARCH),arm64)
      FEATURES ?= gpu
    else
      FEATURES ?= cpu
    endif
  else
    FEATURES ?= cpu
  endif
else
  FEATURES ?= gpu
endif

# Number of parallel jobs
JOBS := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

#############################################################################
# PRIMARY TARGETS - THE ONES YOU'LL USE MOST
#############################################################################

## help: Show annotated targets
help:
	@echo "$(BLUE)BitNet-rs One-Click Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Primary Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z0-9_ -]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort
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
	@scripts/deploy.sh quick

## all: Build everything with all features
all:
	@echo "$(GREEN)Building everything...$(NC)"
	@$(CARGO) build --workspace --all-features --release
	@echo "$(GREEN)✓ Build complete$(NC)"

## install: Full installation with all dependencies
install:
	@echo "$(GREEN)Full installation...$(NC)"
	@scripts/deploy.sh full

## dev: Setup development environment
dev:
	@echo "$(GREEN)Setting up development environment...$(NC)"
	@scripts/deploy.sh dev

#############################################################################
# BUILD TARGETS
#############################################################################

## build: Build with detected features
build:
	@echo "$(GREEN)Building with $(FEATURES) features...$(NC)"
	@$(CARGO) build --locked --release --no-default-features --features $(FEATURES)

## release: Build optimized release
release:
	@echo "$(GREEN)Building optimized release...$(NC)"
	@RUSTFLAGS="-C target-cpu=native -C lto=fat" \
		$(CARGO) build --locked --release --no-default-features --features $(FEATURES)

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
	@$(CARGO) test --locked --workspace --no-default-features --features $(FEATURES)

## test-quick: Run quick tests only
test-quick:
	@$(CARGO) test --locked --workspace --lib --no-default-features --features $(FEATURES)

## test-gpu: Run GPU tests (if available)
test-gpu:
	@echo "$(GREEN)Running GPU tests...$(NC)"
	@$(CARGO) xtask gpu-smoke

## test-integration: Run integration tests
test-integration:
	@$(CARGO) test --locked --workspace --test '*' --no-default-features --features $(FEATURES)

## bench: Run benchmarks
bench:
	@echo "$(GREEN)Running benchmarks...$(NC)"
	@$(CARGO) bench --workspace --no-default-features --features $(FEATURES)

#############################################################################
# RUN TARGETS
#############################################################################

## run: Run the CLI
run:
	@$(CARGO) run --release --no-default-features --features $(FEATURES) -- $(ARGS)

## serve: Start the server
serve:
	@$(CARGO) run --release -p bitnet-server --no-default-features --features $(FEATURES)

## repl: Start interactive REPL
repl:
	@$(CARGO) run --release --no-default-features --features $(FEATURES) -- repl

## demo: Run demos
demo:
	@$(CARGO) xtask demo --which all

#############################################################################
# DEVELOPMENT TARGETS
#############################################################################

## fmt: Format all code
fmt:
	@echo "$(GREEN)Formatting code...$(NC)"
	@$(CARGO) fmt --all
	@echo "$(GREEN)✓ Code formatted$(NC)"

## lint: Run clippy lints
lint:
	@echo "$(GREEN)Running clippy...$(NC)"
	@$(CARGO) clippy --workspace --all-targets --all-features -- -D warnings

## check: Run all checks (fmt, lint, test)
check: fmt lint test
	@echo "$(GREEN)✓ All checks passed$(NC)"

## fix: Auto-fix issues
fix:
	@echo "$(GREEN)Auto-fixing issues...$(NC)"
	@$(CARGO) fix --workspace --allow-dirty --allow-staged
	@$(CARGO) fmt --all
	@$(CARGO) clippy --workspace --fix --allow-dirty --allow-staged -- -D warnings

## docs: Generate and open documentation
docs:
	@echo "$(GREEN)Generating documentation...$(NC)"
	@$(CARGO) doc --workspace --no-deps --open

#############################################################################
# GPU TARGETS
#############################################################################

## gpu: Check GPU availability
gpu:
	@$(CARGO) xtask gpu-preflight

## gpu-smoke: Run GPU smoke tests
gpu-smoke:
	@$(CARGO) xtask gpu-smoke

#############################################################################
# MAINTENANCE TARGETS
#############################################################################

## clean: Clean all build artifacts
clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	@$(CARGO) clean
	@rm -rf target/
	@echo "$(GREEN)✓ Clean complete$(NC)"

## update: Update all dependencies
update:
	@echo "$(GREEN)Updating dependencies...$(NC)"
	@$(CARGO) update
	@rustup update
	@echo "$(GREEN)✓ Updates complete$(NC)"

## setup: Initial setup
setup:
	@echo "$(GREEN)Running initial setup...$(NC)"
	@scripts/deploy.sh full

#############################################################################
# CI/CD TARGETS
#############################################################################

## ci: Run CI checks
ci:
	@echo "$(GREEN)Running CI checks...$(NC)"
	@$(CARGO) fmt --all -- --check
	@$(CARGO) clippy --workspace --all-targets --all-features -- -D warnings
	@$(CARGO) test --locked --workspace --no-default-features --features cpu
	@echo "$(GREEN)✓ CI checks passed$(NC)"

## deploy: Deploy to production
deploy:
	@echo "$(GREEN)Deploying to production...$(NC)"
	@scripts/deploy.sh prod

#############################################################################
# UTILITY TARGETS
#############################################################################

## download-model: Download BitNet model
download-model:
	@$(CARGO) xtask download-model

## crossval: Run cross-validation tests
crossval:
	@$(CARGO) xtask full-crossval

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
	@$(CARGO) watch -x 'build --locked --release --no-default-features --features $(FEATURES)'

## flame: Generate flamegraph (requires cargo-flamegraph)
flame:
	@$(CARGO) flamegraph --release --no-default-features --features $(FEATURES)

## audit: Security audit
audit:
	@$(CARGO) audit

## outdated: Check for outdated dependencies
outdated:
	@$(CARGO) outdated

## bloat: Analyze binary bloat
bloat:
	@$(CARGO) bloat --release --no-default-features --features $(FEATURES)

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
	@$(CARGO) build --locked --release --no-default-features --features $(FEATURES)
	@perf record -g target/release/bitnet
	@perf report

## valgrind: Run with valgrind (Linux only)
valgrind:
	@$(CARGO) build --locked --release --no-default-features --features $(FEATURES)
	@valgrind --leak-check=full --show-leak-kinds=all target/release/bitnet

## heaptrack: Memory profiling (requires heaptrack)
heaptrack:
	@$(CARGO) build --locked --release --no-default-features --features $(FEATURES)
	@heaptrack target/release/bitnet

#############################################################################
# EXPERIMENTAL
#############################################################################

## wasm: Build WebAssembly target
wasm:
	@$(CARGO) build --target wasm32-unknown-unknown -p bitnet-wasm

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
	@RUST_LOG=debug $(CARGO) run --release --no-default-features --features $(FEATURES)

.SILENT: help list
