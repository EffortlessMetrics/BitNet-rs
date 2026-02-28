# Intel GPU Docker Test Environment

Docker Compose environment for running BitNet-rs GPU integration tests on
Intel Arc GPUs using the Intel Compute Runtime (NEO) OpenCL driver.

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Linux host | Kernel 6.2+ | `/dev/dri` must be present |
| Intel GPU | Arc A-series / Xe | Must support OpenCL 3.0 |
| Docker | 24.0+ | With GPU device pass-through |
| Docker Compose | v2.20+ | — |

Install the host-side kernel driver:

```bash
# Ubuntu 22.04+
sudo apt-get install -y intel-opencl-icd
# Verify
clinfo --list
```

## Quick Start

```bash
# From repository root
docker compose -f docker-compose.gpu-test.yml up --build

# Run tests only (no monitor)
docker compose -f docker-compose.gpu-test.yml run gpu-tests

# Tear down
docker compose -f docker-compose.gpu-test.yml down -v
```

## Services

### `gpu-tests`

Runs the OpenCL-tagged tests in the BitNet-rs workspace using
`cargo nextest`.  Passes through `/dev/dri` for GPU access and sets
`BITNET_GPU_BACKEND=opencl`.

Health check: `clinfo --list` verifies the OpenCL ICD loader can see
at least one device.

### `gpu-monitor`

Runs `intel_gpu_top` in JSON mode (2 s sample interval) so you can
observe GPU utilisation while the test suite runs.  Falls back
gracefully on non-Intel hosts.

## Makefile Targets

The repository `Makefile` provides convenience targets:

```bash
make gpu-test-build   # Build the GPU test image
make gpu-test-run     # Run GPU integration tests
make gpu-test-monitor # Start the GPU monitor sidecar
make gpu-test-down    # Tear down all GPU test services
make gpu-test-check   # Health-check: verify GPU is visible
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `clinfo` shows 0 platforms | Install `intel-opencl-icd` on host |
| Permission denied on `/dev/dri` | Add user to `video` + `render` groups |
| Container starts but no GPU | Ensure `--device /dev/dri` is passed |
| `intel_gpu_top` unavailable | Expected on non-Intel hosts; harmless |

## CI Integration

The GitHub Actions workflow `.github/workflows/gpu-smoke.yml` uses this
compose file on self-hosted runners tagged `intel-gpu`.  See the workflow
for caching and artifact upload details.
