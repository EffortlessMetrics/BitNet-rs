use bitnet_kernels::gpu_utils::get_gpu_info;
use bitnet_tests::support::env_guard::EnvGuard;
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
        let _path_guard = EnvGuard::new("PATH");
        _path_guard.set(dir.path().to_str().unwrap());

        let _fake_guard = EnvGuard::new("BITNET_GPU_FAKE");
        _fake_guard.remove(); // Ensure no fake GPU override

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

        let _path_guard = EnvGuard::new("PATH");
        let original_path = std::env::var("PATH").unwrap_or_default();
        _path_guard.set(&format!("{}:{}", dir.path().display(), original_path));

        let _fake_guard = EnvGuard::new("BITNET_GPU_FAKE");
        _fake_guard.remove(); // Ensure no fake GPU override

        let info = get_gpu_info();
        assert!(info.cuda);
        assert!(info.cuda_version.unwrap_or_default().starts_with("12.1"));
    }

    // Guards automatically restore original PATH on drop
}
