# GPU Docker Containers for BitNet-rs

Docker images for building and testing BitNet-rs with GPU acceleration.

## Images

| Image | Backend | Base | Use case |
|-------|---------|------|----------|
| `Dockerfile.intel-gpu` | Intel OpenCL/Level Zero | `rust:1.92-bookworm` | Intel Arc/UHD GPU testing |
| `Dockerfile.multi-gpu` | CUDA, Intel, or CPU | Varies by `GPU_BACKEND` arg | Unified multi-backend builds |

## Quick Start

### Intel GPU

```bash
# Build
docker build -f docker/Dockerfile.intel-gpu -t bitnet-rs:intel-gpu .

# Run with GPU passthrough (requires /dev/dri)
docker run --device /dev/dri bitnet-rs:intel-gpu

# Verify GPU is visible
docker run --device /dev/dri bitnet-rs:intel-gpu clinfo --list
```

### Multi-GPU (backend selection via build arg)

```bash
# CUDA backend (default)
docker build -f docker/Dockerfile.multi-gpu \
  --build-arg GPU_BACKEND=cuda \
  -t bitnet-rs:cuda .

# Intel GPU backend
docker build -f docker/Dockerfile.multi-gpu \
  --build-arg GPU_BACKEND=intel \
  -t bitnet-rs:intel .

# CPU-only backend
docker build -f docker/Dockerfile.multi-gpu \
  --build-arg GPU_BACKEND=cpu \
  -t bitnet-rs:cpu .
```

### Docker Compose (local testing)

```bash
# Run CPU tests (no GPU required)
docker compose -f docker/docker-compose.gpu-test.yml up cpu-test

# Run Intel GPU build + tests
docker compose -f docker/docker-compose.gpu-test.yml up intel-gpu-build

# Run CUDA GPU build + tests (requires NVIDIA Container Toolkit)
docker compose -f docker/docker-compose.gpu-test.yml up cuda-gpu-build
```

## Environment Requirements

### Intel GPU

- Linux host with Intel GPU (Arc, UHD, or Xe)
- Intel GPU kernel driver loaded (`i915` or `xe`)
- `/dev/dri` device nodes available
- Pass `--device /dev/dri` to `docker run`

### NVIDIA CUDA

- Linux host with NVIDIA GPU
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed
- Pass `--gpus all` or use `deploy.resources.reservations.devices` in Compose

### Testing Without Hardware

All images can build without GPU hardware — the build steps only require
development headers and libraries, not a physical GPU. Runtime tests that
exercise the GPU will fail gracefully or be skipped when no device is present.

```bash
# Build-only validation (no GPU needed)
docker build -f docker/Dockerfile.intel-gpu --target builder -t bitnet-rs:intel-gpu-build .
docker build -f docker/Dockerfile.multi-gpu --build-arg GPU_BACKEND=cuda --target builder-cuda .

# CPU tests always work
docker compose -f docker/docker-compose.gpu-test.yml up cpu-test
```

## CI Integration

The `docker-gpu-build.yml` workflow:

- **Trigger**: Weekly (Sunday 3 AM UTC), on push/PR to `docker/` or the workflow file
- **Matrix**: Builds `intel-gpu`, `multi-gpu-cuda`, `multi-gpu-intel`, `multi-gpu-cpu`
- **Registry**: Pushes to GHCR (`ghcr.io/<owner>/bitnet-rs/<image>`) on non-PR events
- **Caching**: Uses GitHub Actions cache (`type=gha`) for layer caching

### Pulling CI images

```bash
# After CI pushes to GHCR
docker pull ghcr.io/<owner>/bitnet-rs/intel-gpu:main
docker pull ghcr.io/<owner>/bitnet-rs/multi-gpu-cuda:main
```

## Build Args

All Dockerfiles accept these build arguments for OCI labels:

| Arg | Description | Default |
|-----|-------------|---------|
| `VCS_REF` | Git commit SHA | `unknown` |
| `VCS_BRANCH` | Git branch name | `unknown` |
| `VCS_DESCRIBE` | Git describe output | `unknown` |
| `GPU_BACKEND` | Backend selection (multi-gpu only) | `cuda` |

## Troubleshooting

### Intel GPU not detected

```bash
# Check kernel driver
lsmod | grep -E 'i915|xe'

# Check device nodes
ls -la /dev/dri/

# Test inside container
docker run --device /dev/dri ubuntu:22.04 ls -la /dev/dri/
```

### CUDA not available

```bash
# Check NVIDIA Container Toolkit
nvidia-container-cli info

# Test GPU access
docker run --gpus all nvidia/cuda:12.3.1-runtime-ubuntu22.04 nvidia-smi
```
