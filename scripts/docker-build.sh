#!/bin/bash
# Build Docker images with Git metadata injected
set -euo pipefail

# Extract Git metadata
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_DESCRIBE=$(git describe --tags --always 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Export for docker-compose
export GIT_SHA
export GIT_BRANCH
export GIT_DESCRIBE

# Build target (cpu, gpu, or all)
TARGET="${1:-cpu}"

echo "Building BitNet-rs Docker images with Git metadata:"
echo "  SHA: $GIT_SHA"
echo "  Branch: $GIT_BRANCH"
echo "  Describe: $GIT_DESCRIBE"
echo ""

case "$TARGET" in
  cpu)
    echo "Building CPU image..."
    docker build \
      --build-arg VCS_REF="$GIT_SHA" \
      --build-arg VCS_BRANCH="$GIT_BRANCH" \
      --build-arg VCS_DESCRIBE="$GIT_DESCRIBE" \
      --build-arg FEATURES=cpu \
      --target runtime \
      -t bitnet-rs:cpu \
      -t bitnet-rs:cpu-"$GIT_SHA" \
      .
    ;;
  gpu)
    echo "Building GPU image..."
    docker build \
      --build-arg VCS_REF="$GIT_SHA" \
      --build-arg VCS_BRANCH="$GIT_BRANCH" \
      --build-arg VCS_DESCRIBE="$GIT_DESCRIBE" \
      --build-arg FEATURES=gpu \
      --target runtime-gpu \
      -t bitnet-rs:gpu \
      -t bitnet-rs:gpu-"$GIT_SHA" \
      .
    ;;
  all)
    echo "Building both CPU and GPU images..."
    # Build CPU
    docker build \
      --build-arg VCS_REF="$GIT_SHA" \
      --build-arg VCS_BRANCH="$GIT_BRANCH" \
      --build-arg VCS_DESCRIBE="$GIT_DESCRIBE" \
      --build-arg FEATURES=cpu \
      --target runtime \
      -t bitnet-rs:cpu \
      -t bitnet-rs:cpu-"$GIT_SHA" \
      .
    # Build GPU
    docker build \
      --build-arg VCS_REF="$GIT_SHA" \
      --build-arg VCS_BRANCH="$GIT_BRANCH" \
      --build-arg VCS_DESCRIBE="$GIT_DESCRIBE" \
      --build-arg FEATURES=gpu \
      --target runtime-gpu \
      -t bitnet-rs:gpu \
      -t bitnet-rs:gpu-"$GIT_SHA" \
      .
    ;;
  *)
    echo "Usage: $0 [cpu|gpu|all]"
    exit 1
    ;;
esac

echo ""
echo "Build complete! Images tagged with:"
echo "  - bitnet-rs:$TARGET (latest)"
echo "  - bitnet-rs:$TARGET-$GIT_SHA (versioned)"
echo ""
echo "To run with docker-compose:"
echo "  GIT_SHA=$GIT_SHA GIT_BRANCH=$GIT_BRANCH GIT_DESCRIBE=$GIT_DESCRIBE docker-compose up bitnet-$TARGET"
