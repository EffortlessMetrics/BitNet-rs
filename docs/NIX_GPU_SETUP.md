# Nix GPU Development Setup

## Quick Start

```bash
# Enter GPU development shell
nix develop .#gpu

# Build with GPU features
cargo build --no-default-features --features gpu

# Run GPU tests (requires hardware)
cargo nextest run --workspace --no-default-features --features gpu

# Check OpenCL availability
clinfo --list
```

## Available Shells

| Shell | Command | Description |
|-------|---------|-------------|
| Default | `nix develop` | CPU-only development |
| GPU | `nix develop .#gpu` | OpenCL + Vulkan + tools |
| MSRV | `nix develop .#msrv` | Minimum supported Rust version |

## Hermetic GPU Builds

```bash
# Build GPU server binary
nix build .#bitnet-server-gpu

# Build GPU CLI binary
nix build .#bitnet-cli-gpu

# Run GPU checks
nix flake check .#gpu-workspace
```

## Environment Variables

The GPU development shell automatically sets:

| Variable | Purpose |
|----------|---------|
| `OCL_ICD_VENDORS` | OpenCL ICD discovery path |
| `VK_ICD_FILENAMES` | Vulkan driver discovery path |
| `LIBCLANG_PATH` | Clang library path for bindgen |

## CI Integration

The Nix flake provides GPU-aware checks that can be used in CI:

```yaml
- name: GPU build check
  run: nix build .#bitnet-server-gpu

- name: GPU workspace check
  run: nix flake check .#gpu-workspace
```

## Troubleshooting

### OpenCL not found

Ensure `OCL_ICD_VENDORS` is set. In the Nix shell, this is automatic:

```bash
echo $OCL_ICD_VENDORS
clinfo --list
```

### Vulkan validation errors

Set `VK_LAYER_PATH` to the validation layers directory:

```bash
export VK_LAYER_PATH="${VULKAN_SDK}/share/vulkan/explicit_layer.d"
```

### GPU not detected at runtime

GPU hardware must be available on the host. The Nix shell provides the
SDK and tooling, but actual GPU execution requires compatible hardware
and drivers installed on the host system.
