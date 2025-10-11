# Device Information Test Fixtures (Issue #439 AC3)

## Purpose

Provide GPU device information fixtures for testing device capability detection and feature validation.

## Fixture Files

### Valid GPU Configurations

1. **`cuda_available.json`**
   - Single NVIDIA RTX 4090 GPU
   - Compute capability 8.9 (modern architecture)
   - Tensor Core support, FP16/BF16 capable
   - 24 GB VRAM
   - Usage: Standard GPU test scenario

2. **`multi_gpu.json`**
   - Dual NVIDIA A100 GPUs
   - Compute capability 8.0 (high-end datacenter)
   - Full mixed precision support
   - 40 GB VRAM per GPU
   - Usage: Multi-GPU selection testing

3. **`old_gpu.json`**
   - NVIDIA GTX 1080 Ti
   - Compute capability 6.1 (legacy architecture)
   - No Tensor Core support, limited mixed precision
   - 11 GB VRAM
   - Usage: Testing fallback for older GPUs

### No GPU Configuration

4. **`no_gpu.json`**
   - No CUDA devices available
   - CPU-only environment
   - Usage: CPU fallback testing

## Testing Usage

### Load Device Information
```rust
use std::fs;
use serde::Deserialize;

#[derive(Deserialize)]
struct DeviceInfo {
    cuda: bool,
    driver_version: Option<String>,
    device_count: usize,
}

#[test]
fn test_gpu_available() {
    let device_info: DeviceInfo = serde_json::from_str(
        &fs::read_to_string("tests/fixtures/device_info/cuda_available.json").unwrap()
    ).unwrap();

    assert!(device_info.cuda);
    assert_eq!(device_info.device_count, 1);
}
```

### Validate Device Capabilities
```rust
#[test]
fn test_tensor_core_support() {
    let device_info: serde_json::Value = serde_json::from_str(
        &fs::read_to_string("tests/fixtures/device_info/cuda_available.json").unwrap()
    ).unwrap();

    let supports_tensor_cores = device_info["devices"][0]["supports_tensor_cores"]
        .as_bool()
        .unwrap();

    assert!(supports_tensor_cores, "RTX 4090 should support Tensor Cores");
}
```

## Integration with Tests

These fixtures are consumed by:
- `crates/bitnet-kernels/tests/device_features.rs` (AC3 tests)
- `xtask/tests/preflight.rs` (Device capability reporting)
- GPU kernel selection logic

## Specification Reference

- **Issue**: #439 GPU feature-gate hardening
- **Acceptance Criteria**: AC3 - Shared helpers (device features)
- **Specification**: `docs/explanation/issue-439-spec.md#ac3-shared-helpers`

## Device Information Schema

### Top-Level Fields
- `cuda` (bool): CUDA availability
- `driver_version` (string|null): CUDA driver version (e.g., "12.2")
- `runtime_version` (string|null): CUDA runtime version
- `device_count` (number): Number of CUDA devices

### Device Fields
- `id` (number): Device ID (0-indexed)
- `name` (string): GPU name (e.g., "NVIDIA RTX 4090")
- `compute_capability` (string): CUDA compute capability (e.g., "8.9")
- `memory_total_gb` (number): Total VRAM in GB
- `memory_available_gb` (number): Available VRAM in GB
- `multiprocessor_count` (number): Number of SMs
- `max_threads_per_block` (number): Max threads per block
- `supports_tensor_cores` (bool): Tensor Core availability
- `supports_fp16` (bool): FP16 support
- `supports_bf16` (bool): BF16 support
- `pcie_generation` (number): PCIe generation (3 or 4)
- `pcie_width` (number): PCIe lane width

## Device Capability Matrix

| Fixture | Compute Cap | Tensor Cores | FP16 | BF16 | Use Case |
|---------|-------------|--------------|------|------|----------|
| `cuda_available.json` | 8.9 | ✓ | ✓ | ✓ | Modern GPU testing |
| `multi_gpu.json` | 8.0 | ✓ | ✓ | ✓ | Multi-GPU scenarios |
| `old_gpu.json` | 6.1 | ✗ | ✓ | ✗ | Legacy GPU fallback |
| `no_gpu.json` | N/A | ✗ | ✗ | ✗ | CPU-only testing |

## Common Usage Patterns

### Check Mixed Precision Support
```rust
fn supports_mixed_precision(device_info: &DeviceInfo) -> bool {
    device_info.cuda
        && device_info.devices.iter().any(|d| {
            d.supports_fp16 && d.supports_bf16 && d.supports_tensor_cores
        })
}
```

### Select Best Device
```rust
fn select_best_device(device_info: &DeviceInfo) -> Option<usize> {
    device_info.devices.iter()
        .enumerate()
        .max_by_key(|(_, d)| d.memory_available_gb as u64)
        .map(|(idx, _)| idx)
}
```

### Validate Minimum Requirements
```rust
fn meets_minimum_requirements(device_info: &DeviceInfo) -> bool {
    device_info.cuda
        && device_info.devices.iter().any(|d| {
            d.compute_capability >= "7.0"
                && d.memory_total_gb >= 8.0
        })
}
```

## Validation Checklist

Device information should include:
- [ ] `cuda` boolean flag
- [ ] `driver_version` (null if no CUDA)
- [ ] `device_count` matching devices array length
- [ ] Device `compute_capability` as string (e.g., "8.9")
- [ ] Memory fields in GB (not bytes)
- [ ] Boolean capability flags (tensor_cores, fp16, bf16)

## Example Validation Test

```rust
#[test]
fn ac3_device_info_fixture_validity() {
    let fixtures = [
        "tests/fixtures/device_info/cuda_available.json",
        "tests/fixtures/device_info/multi_gpu.json",
        "tests/fixtures/device_info/old_gpu.json",
        "tests/fixtures/device_info/no_gpu.json",
    ];

    for fixture_path in &fixtures {
        let content = std::fs::read_to_string(fixture_path).unwrap();
        let device_info: serde_json::Value =
            serde_json::from_str(&content).unwrap();

        // Validate schema
        assert!(device_info.get("cuda").is_some());
        assert!(device_info.get("device_count").is_some());
        assert!(device_info.get("devices").is_some());

        println!("AC:3 PASS - Valid device info: {}", fixture_path);
    }
}
```
