#!/bin/bash

# BitNet Docker Image Build Script
# This script builds optimized Docker images for CPU and GPU deployments

set -euo pipefail

# Configuration
REGISTRY="${REGISTRY:-bitnet}"
TAG="${TAG:-latest}"
PLATFORM="${PLATFORM:-linux/amd64,linux/arm64}"
PUSH="${PUSH:-false}"
BUILD_ARGS="${BUILD_ARGS:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker buildx version &> /dev/null; then
        log_error "Docker Buildx is not available"
        exit 1
    fi
    
    # Check if buildx builder exists
    if ! docker buildx ls | grep -q "bitnet-builder"; then
        log_info "Creating buildx builder..."
        docker buildx create --name bitnet-builder --use
        docker buildx inspect --bootstrap
    else
        docker buildx use bitnet-builder
    fi
    
    log_success "Prerequisites check passed"
}

# Build CPU image
build_cpu_image() {
    log_info "Building CPU image..."
    
    local image_name="${REGISTRY}/bitnet:cpu-${TAG}"
    local build_cmd="docker buildx build"
    
    build_cmd+=" --platform ${PLATFORM}"
    build_cmd+=" --file docker/Dockerfile.cpu"
    build_cmd+=" --tag ${image_name}"
    build_cmd+=" --target runtime"
    
    # Add build arguments
    if [[ -n "${BUILD_ARGS}" ]]; then
        build_cmd+=" ${BUILD_ARGS}"
    fi
    
    # Add cache options
    build_cmd+=" --cache-from type=local,src=/tmp/.buildx-cache"
    build_cmd+=" --cache-to type=local,dest=/tmp/.buildx-cache-new,mode=max"
    
    if [[ "${PUSH}" == "true" ]]; then
        build_cmd+=" --push"
    else
        build_cmd+=" --load"
    fi
    
    build_cmd+=" ."
    
    log_info "Executing: ${build_cmd}"
    eval "${build_cmd}"
    
    # Move cache
    if [[ -d "/tmp/.buildx-cache-new" ]]; then
        rm -rf /tmp/.buildx-cache
        mv /tmp/.buildx-cache-new /tmp/.buildx-cache
    fi
    
    log_success "CPU image built successfully: ${image_name}"
}

# Build GPU image
build_gpu_image() {
    log_info "Building GPU image..."
    
    local image_name="${REGISTRY}/bitnet:gpu-${TAG}"
    local build_cmd="docker buildx build"
    
    build_cmd+=" --platform linux/amd64"  # GPU images are x86_64 only
    build_cmd+=" --file docker/Dockerfile.gpu"
    build_cmd+=" --tag ${image_name}"
    build_cmd+=" --target runtime"
    
    # Add build arguments
    if [[ -n "${BUILD_ARGS}" ]]; then
        build_cmd+=" ${BUILD_ARGS}"
    fi
    
    # Add cache options
    build_cmd+=" --cache-from type=local,src=/tmp/.buildx-cache-gpu"
    build_cmd+=" --cache-to type=local,dest=/tmp/.buildx-cache-gpu-new,mode=max"
    
    if [[ "${PUSH}" == "true" ]]; then
        build_cmd+=" --push"
    else
        build_cmd+=" --load"
    fi
    
    build_cmd+=" ."
    
    log_info "Executing: ${build_cmd}"
    eval "${build_cmd}"
    
    # Move cache
    if [[ -d "/tmp/.buildx-cache-gpu-new" ]]; then
        rm -rf /tmp/.buildx-cache-gpu
        mv /tmp/.buildx-cache-gpu-new /tmp/.buildx-cache-gpu
    fi
    
    log_success "GPU image built successfully: ${image_name}"
}

# Test images
test_images() {
    log_info "Testing built images..."
    
    # Test CPU image
    local cpu_image="${REGISTRY}/bitnet:cpu-${TAG}"
    if docker image inspect "${cpu_image}" &> /dev/null; then
        log_info "Testing CPU image..."
        if docker run --rm "${cpu_image}" --version; then
            log_success "CPU image test passed"
        else
            log_error "CPU image test failed"
            return 1
        fi
    fi
    
    # Test GPU image (only if NVIDIA runtime is available)
    local gpu_image="${REGISTRY}/bitnet:gpu-${TAG}"
    if docker image inspect "${gpu_image}" &> /dev/null; then
        log_info "Testing GPU image..."
        if docker run --rm "${gpu_image}" --version; then
            log_success "GPU image test passed"
        else
            log_warning "GPU image test failed (this is expected without NVIDIA runtime)"
        fi
    fi
}

# Clean up build cache
cleanup() {
    log_info "Cleaning up build cache..."
    
    # Remove old cache directories
    rm -rf /tmp/.buildx-cache-old /tmp/.buildx-cache-gpu-old
    
    # Prune build cache
    docker buildx prune -f
    
    log_success "Cleanup completed"
}

# Main function
main() {
    log_info "Starting BitNet Docker image build process..."
    log_info "Registry: ${REGISTRY}"
    log_info "Tag: ${TAG}"
    log_info "Platform: ${PLATFORM}"
    log_info "Push: ${PUSH}"
    
    check_prerequisites
    
    # Build images based on arguments
    if [[ $# -eq 0 ]] || [[ "$*" == *"cpu"* ]]; then
        build_cpu_image
    fi
    
    if [[ $# -eq 0 ]] || [[ "$*" == *"gpu"* ]]; then
        build_gpu_image
    fi
    
    # Test images if not pushing
    if [[ "${PUSH}" != "true" ]]; then
        test_images
    fi
    
    log_success "Build process completed successfully!"
    
    # Show built images
    log_info "Built images:"
    docker images | grep "${REGISTRY}/bitnet" | grep "${TAG}"
}

# Help function
show_help() {
    cat << EOF
BitNet Docker Image Build Script

Usage: $0 [OPTIONS] [cpu|gpu]

OPTIONS:
    -r, --registry REGISTRY    Docker registry (default: bitnet)
    -t, --tag TAG             Image tag (default: latest)
    -p, --platform PLATFORM   Target platform (default: linux/amd64,linux/arm64)
    --push                    Push images to registry
    --build-arg ARG           Pass build argument to docker build
    --cleanup                 Clean up build cache after build
    -h, --help                Show this help message

EXAMPLES:
    # Build both CPU and GPU images
    $0

    # Build only CPU image
    $0 cpu

    # Build and push to custom registry
    $0 --registry myregistry.com/bitnet --tag v1.0.0 --push

    # Build with custom build arguments
    $0 --build-arg RUST_VERSION=1.75 --build-arg FEATURES=cpu,optimized

    # Build for specific platform
    $0 --platform linux/amd64 cpu

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        --push)
            PUSH="true"
            shift
            ;;
        --build-arg)
            BUILD_ARGS+=" --build-arg $2"
            shift 2
            ;;
        --cleanup)
            cleanup
            exit 0
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        cpu|gpu)
            # These are handled in main()
            break
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main "$@"