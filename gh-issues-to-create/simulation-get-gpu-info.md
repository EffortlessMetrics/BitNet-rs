# Simulation: `get_gpu_info` in `gpu_utils.rs` uses `BITNET_GPU_FAKE` environment variable

The `get_gpu_info` function in `crates/bitnet-kernels/src/gpu_utils.rs` checks for the `BITNET_GPU_FAKE` environment variable to simulate GPU availability. This is a form of simulation.

**File:** `crates/bitnet-kernels/src/gpu_utils.rs`

**Function:** `get_gpu_info`

**Code:**
```rust
pub fn get_gpu_info() -> GpuInfo {
    if let Ok(fake) = env::var("BITNET_GPU_FAKE") {
        if env::var("BITNET_STRICT_NO_FAKE_GPU").as_deref() == Ok("1") {
            panic!(
                "BITNET_GPU_FAKE is set but strict mode forbids fake GPU (BITNET_STRICT_NO_FAKE_GPU=1)"
            );
        }
        let lower = fake.to_lowercase();
        return GpuInfo {
            cuda: lower.contains("cuda"),
            cuda_version: None,
            metal: lower.contains("metal"),
            rocm: lower.contains("rocm"),
            rocm_version: None,
            wgpu: lower.contains("wgpu")
                || lower.contains("cuda")
                || lower.contains("rocm")
                || lower.contains("metal"),
        };
    }

    // ...
}
```

## Proposed Fix

The `get_gpu_info` function should not use the `BITNET_GPU_FAKE` environment variable for simulating GPU availability. Instead, it should directly query the system for actual GPU information. The simulation functionality should be moved to a dedicated test utility.

### Example Implementation

```rust
pub fn get_gpu_info() -> GpuInfo {
    let _sys = System::new_all();

    let mut metal = System::name().unwrap_or_default().to_lowercase().contains("mac");

    let cuda = Command::new("nvidia-smi")
        .arg("--query-gpu=gpu_name")
        .arg("--format=csv,noheader")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    let cuda_version = if cuda { get_cuda_version() } else { None };

    let rocm = Command::new("rocm-smi")
        .arg("--showid")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    let rocm_version = if rocm { get_rocm_version() } else { None };

    if cfg!(target_os = "macos") {
        metal = true;
    }

    let wgpu = cuda || rocm || metal;

    GpuInfo { cuda, cuda_version, metal, rocm, rocm_version, wgpu }
}
```
