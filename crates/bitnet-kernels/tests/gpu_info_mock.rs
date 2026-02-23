use bitnet_kernels::gpu_utils::get_gpu_info;
use bitnet_tests::support::env_guard::EnvScope;
use serial_test::serial;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use tempfile::tempdir;

fn make_exec(path: &std::path::Path) {
    let mut perms = fs::metadata(path).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms).unwrap();
}

#[test]
#[serial(bitnet_env)]
fn test_gpu_info_mocked_scenarios() {
    // Scenario: no GPU present
    {
        let dir = tempdir().unwrap();
        let mut scope = EnvScope::new();
        scope.set("PATH", dir.path().to_str().unwrap());
        scope.remove("BITNET_GPU_FAKE");

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

        let original_path = std::env::var("PATH").unwrap_or_default();
        let mut scope = EnvScope::new();
        scope.set("PATH", &format!("{}:{}", dir.path().display(), original_path));
        scope.remove("BITNET_GPU_FAKE");

        let info = get_gpu_info();
        assert!(info.cuda);
        assert!(info.cuda_version.unwrap_or_default().starts_with("12.1"));
    }

    // EnvScope automatically restores original PATH on drop
}
