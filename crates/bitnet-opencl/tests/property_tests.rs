//! Property-based tests for the OpenCL kernel infrastructure.
//!
//! These tests use `proptest` to verify invariants that must hold across
//! all valid inputs:
//!
//! - **KernelSelector** always returns a valid `KernelVariant` (no panics).
//! - **WorkgroupConfig** always produces positive sizes that divide evenly.
//! - **Buffer calculations** never silently overflow.
//! - **Kernel source strings** are always valid UTF-8 and non-empty.

use bitnet_opencl::OpenClDeviceInfo;
use bitnet_opencl::buffers;
use bitnet_opencl::kernels;
use bitnet_opencl::selector::{KernelSelector, KernelVariant};
use bitnet_opencl::workgroup::WorkgroupConfig;
use proptest::prelude::*;

// ── Device info strategy ─────────────────────────────────────────────

fn arb_device_info() -> impl Strategy<Value = OpenClDeviceInfo> {
    (
        1_usize..=1024,                                     // max_workgroup_size
        1_u32..=128,                                        // max_compute_units
        1_u64..=16 * 1024 * 1024 * 1024,                    // global_mem_bytes (up to 16 GiB)
        1_u64..=256 * 1024,                                 // local_mem_bytes (up to 256 KiB)
        prop::sample::select(vec![1_usize, 8, 16, 32, 64]), // pref multiple
    )
        .prop_map(|(max_wg, cu, gmem, lmem, pref)| OpenClDeviceInfo {
            max_workgroup_size: max_wg,
            max_compute_units: cu,
            global_mem_bytes: gmem,
            local_mem_bytes: lmem,
            preferred_workgroup_multiple: pref,
            ..Default::default()
        })
}

// ── KernelSelector invariants ────────────────────────────────────────

proptest! {
    #[test]
    fn selector_gemv_always_returns_valid_variant(
        n_out in 0_usize..=65536,
        k in 0_usize..=65536,
        device in arb_device_info(),
    ) {
        let variant = KernelSelector::select_gemv(n_out, k, &device);
        // Must be one of the three valid variants.
        prop_assert!(matches!(
            variant,
            KernelVariant::Small | KernelVariant::Tiled | KernelVariant::LargeReduction
        ));
    }

    #[test]
    fn selector_rmsnorm_always_returns_valid_variant(
        hidden_dim in 0_usize..=65536,
        device in arb_device_info(),
    ) {
        let variant = KernelSelector::select_rmsnorm(hidden_dim, &device);
        prop_assert!(matches!(
            variant,
            KernelVariant::Small | KernelVariant::Tiled | KernelVariant::LargeReduction
        ));
    }

    #[test]
    fn selector_attention_always_returns_valid_variant(
        seq_q in 0_usize..=4096,
        seq_kv in 0_usize..=4096,
        head_dim in 0_usize..=256,
        device in arb_device_info(),
    ) {
        let variant = KernelSelector::select_attention(seq_q, seq_kv, head_dim, &device);
        prop_assert!(matches!(
            variant,
            KernelVariant::Small | KernelVariant::Tiled | KernelVariant::LargeReduction
        ));
    }

    #[test]
    fn selector_gemv_zero_dims_always_small(
        k in 0_usize..=65536,
        device in arb_device_info(),
    ) {
        prop_assert_eq!(
            KernelSelector::select_gemv(0, k, &device),
            KernelVariant::Small
        );
    }
}

// ── Workgroup invariants ─────────────────────────────────────────────

proptest! {
    #[test]
    fn workgroup_local_size_always_positive(
        problem_size in 1_usize..=1_000_000,
        device in arb_device_info(),
    ) {
        let cfg = WorkgroupConfig::for_1d(problem_size, &device).unwrap();
        prop_assert!(cfg.local_size > 0, "local_size must be positive");
    }

    #[test]
    fn workgroup_global_divides_evenly(
        problem_size in 1_usize..=1_000_000,
        device in arb_device_info(),
    ) {
        let cfg = WorkgroupConfig::for_1d(problem_size, &device).unwrap();
        prop_assert_eq!(
            cfg.global_size % cfg.local_size,
            0,
            "global_size {} must divide evenly by local_size {}",
            cfg.global_size,
            cfg.local_size
        );
    }

    #[test]
    fn workgroup_global_covers_problem(
        problem_size in 1_usize..=1_000_000,
        device in arb_device_info(),
    ) {
        let cfg = WorkgroupConfig::for_1d(problem_size, &device).unwrap();
        prop_assert!(
            cfg.global_size >= problem_size,
            "global_size {} must be >= problem_size {}",
            cfg.global_size,
            problem_size
        );
    }

    #[test]
    fn workgroup_local_respects_device_max(
        problem_size in 1_usize..=1_000_000,
        device in arb_device_info(),
    ) {
        let cfg = WorkgroupConfig::for_1d(problem_size, &device).unwrap();
        prop_assert!(
            cfg.local_size <= device.max_workgroup_size,
            "local_size {} exceeds device max {}",
            cfg.local_size,
            device.max_workgroup_size
        );
    }

    #[test]
    fn workgroup_num_groups_consistent(
        problem_size in 1_usize..=1_000_000,
        device in arb_device_info(),
    ) {
        let cfg = WorkgroupConfig::for_1d(problem_size, &device).unwrap();
        prop_assert_eq!(
            cfg.num_groups,
            cfg.global_size / cfg.local_size,
            "num_groups mismatch"
        );
    }

    #[test]
    fn workgroup_zero_always_errors(device in arb_device_info()) {
        prop_assert!(WorkgroupConfig::for_1d(0, &device).is_err());
    }
}

// ── Buffer allocation invariants ─────────────────────────────────────

proptest! {
    #[test]
    fn qk256_weight_bytes_never_overflows(
        n_out in 0_usize..=4096,
        k_blocks in 1_usize..=64,
    ) {
        let k = k_blocks * 256;
        let result = buffers::qk256_weight_bytes(n_out, k);
        // Must either succeed with a sensible value or return an explicit error.
        match result {
            Ok(bytes) => {
                // Verify the result is consistent.
                let expected = n_out * (k / 256) * 64;
                prop_assert_eq!(bytes, expected);
            }
            Err(_) => {
                // Overflow is acceptable if the inputs are huge.
            }
        }
    }

    #[test]
    fn fp32_buffer_never_silently_overflows(count in 0_usize..=usize::MAX) {
        let result = buffers::fp32_buffer_bytes(count);
        match result {
            Ok(bytes) => prop_assert_eq!(bytes, count * 4),
            Err(_) => prop_assert!(count > usize::MAX / 4),
        }
    }

    #[test]
    fn fp16_buffer_never_silently_overflows(count in 0_usize..=usize::MAX) {
        let result = buffers::fp16_buffer_bytes(count);
        match result {
            Ok(bytes) => prop_assert_eq!(bytes, count * 2),
            Err(_) => prop_assert!(count > usize::MAX / 2),
        }
    }

    #[test]
    fn qk256_shared_mem_at_least_minimum(k_blocks in 1_usize..=1024) {
        let k = k_blocks * 256;
        let mem = buffers::qk256_gemv_shared_mem(k).unwrap();
        prop_assert!(mem >= 4096, "shared mem {} below minimum 4096", mem);
    }
}

// ── Kernel source string invariants ──────────────────────────────────

#[test]
fn all_kernel_sources_valid_utf8_and_nonempty() {
    for (name, src) in kernels::all_kernel_sources() {
        assert!(!name.is_empty(), "kernel name must be non-empty");
        assert!(!src.is_empty(), "kernel '{name}' source must be non-empty");
        // Rust &str is UTF-8 by construction, but verify round-trip.
        assert!(
            std::str::from_utf8(src.as_bytes()).is_ok(),
            "kernel '{name}' failed UTF-8 round-trip"
        );
    }
}

#[test]
fn all_kernel_sources_have_unique_names() {
    let sources = kernels::all_kernel_sources();
    let names: Vec<&str> = sources.iter().map(|(n, _)| *n).collect();
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(names.len(), sorted.len(), "duplicate kernel names detected");
}
