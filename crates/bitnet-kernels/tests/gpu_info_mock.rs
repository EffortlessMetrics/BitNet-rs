use bitnet_kernels::gpu_utils::get_gpu_info;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use tempfile::tempdir;

fn make_exec(path: &std::path::Path) {
    let mut perms = fs::metadata(path).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms).unwrap();
}

#[test]
fn test_gpu_info_mocked_scenarios() {
    let original = std::env::var("PATH").unwrap_or_default();

    // Scenario: no GPU present
    {
        let dir = tempdir().unwrap();
        unsafe {
            std::env::set_var("PATH", dir.path());
            std::env::remove_var("BITNET_GPU_FAKE");
        }
        let info = get_gpu_info();
        assert!(!info.any_available());
    }

    // Scenario: CUDA tools available
    {
        let dir = tempdir().unwrap();
        let smi = dir.path().join("nvidia-smi");
        fs::write(&smi, "#!/bin/sh\nexit 0\n").unwrap();
        make_exec(&smi);
        let nvcc = dir.path().join("nvcc");
        fs::write(&nvcc, "#!/bin/sh\necho 'Cuda compilation tools, release 12.1, V12.1.0'\n")
            .unwrap();
        make_exec(&nvcc);
        unsafe {
            std::env::set_var("PATH", format!("{}:{}", dir.path().display(), original));
            std::env::remove_var("BITNET_GPU_FAKE");
        }
        let info = get_gpu_info();
        assert!(info.cuda);
        assert!(info.cuda_version.unwrap_or_default().starts_with("12.1"));
    }

    unsafe {
        std::env::set_var("PATH", original);
    }
}
