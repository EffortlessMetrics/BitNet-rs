//! Tests for strict GPU mode validation that prevents fake GPU backends

use bitnet_kernels::gpu_utils;

#[test]
#[should_panic(expected = "BITNET_GPU_FAKE is set but strict mode forbids fake GPU")]
fn strict_mode_disallows_fake_gpu() {
    // Set both fake GPU and strict mode
    unsafe {
        std::env::set_var("BITNET_GPU_FAKE", "cuda");
        std::env::set_var("BITNET_STRICT_NO_FAKE_GPU", "1");
    }
    
    // This should panic due to strict mode
    gpu_utils::get_gpu_info();
}

#[test]
fn normal_mode_allows_fake_gpu() {
    unsafe {
        // Clean up strict mode if set
        std::env::remove_var("BITNET_STRICT_NO_FAKE_GPU");
        
        // Set fake GPU
        std::env::set_var("BITNET_GPU_FAKE", "cuda");
    }
    
    // Should work fine in normal mode
    let info = gpu_utils::get_gpu_info();
    assert!(info.cuda, "fake cuda should be detected");
    
    // Clean up
    unsafe {
        std::env::remove_var("BITNET_GPU_FAKE");
    }
}

#[test]
fn strict_mode_works_with_real_gpu_detection() {
    unsafe {
        // Clean up fake GPU if set
        std::env::remove_var("BITNET_GPU_FAKE");
        
        // Set strict mode (should work fine with real GPU detection)
        std::env::set_var("BITNET_STRICT_NO_FAKE_GPU", "1");
    }
    
    // Should work fine with real GPU detection
    let _info = gpu_utils::get_gpu_info();
    
    // Clean up
    unsafe {
        std::env::remove_var("BITNET_STRICT_NO_FAKE_GPU");
    }
}