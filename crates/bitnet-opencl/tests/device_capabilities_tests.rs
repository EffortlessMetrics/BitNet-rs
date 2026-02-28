//! Tests for GPU device capability query, validation, and selection.
//!
//! All tests use [`GpuDeviceCapabilities::mock`] so they run without
//! real GPU hardware.

use bitnet_common::kernel_registry::KernelBackend;
use bitnet_opencl::{
    DeviceCapabilityChecker, DeviceSelector, DeviceSelectorError, GpuDeviceCapabilities,
    ModelRequirements, ScoredDevice, format_device_info,
};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build a mock device with overrides applied via a closure.
fn mock_with(f: impl FnOnce(&mut GpuDeviceCapabilities)) -> GpuDeviceCapabilities {
    let mut d = GpuDeviceCapabilities::mock();
    f(&mut d);
    d
}

/// Convenience: 1 GB in bytes.
const GB: u64 = 1024 * 1024 * 1024;

// ── GpuDeviceCapabilities tests ──────────────────────────────────────────────

#[test]
fn mock_device_has_sane_defaults() {
    let d = GpuDeviceCapabilities::mock();
    assert_eq!(d.name, "Mock GPU Device");
    assert_eq!(d.vendor, "MockVendor");
    assert_eq!(d.compute_units, 32);
    assert_eq!(d.global_memory_bytes, 8 * GB);
    assert!(d.supports_fp16);
    assert!(d.supports_fp64);
    assert!(d.supports_subgroups);
    assert_eq!(d.max_subgroup_size, 32);
    assert_eq!(d.backend, KernelBackend::OneApi);
}

#[test]
fn mock_device_is_cloneable() {
    let d = GpuDeviceCapabilities::mock();
    let d2 = d.clone();
    assert_eq!(d, d2);
}

#[test]
fn mock_device_debug_format() {
    let d = GpuDeviceCapabilities::mock();
    let debug = format!("{d:?}");
    assert!(debug.contains("Mock GPU Device"));
}

// ── DeviceCapabilityChecker – compatible ─────────────────────────────────────

#[test]
fn model_fits_in_memory_approved() {
    let device = GpuDeviceCapabilities::mock(); // 8 GB
    let reqs = ModelRequirements { memory_bytes: 4 * GB, ..ModelRequirements::minimal() };
    let result = DeviceCapabilityChecker::check(&device, &reqs);
    assert!(result.is_compatible());
    assert!(result.reasons().is_empty());
}

#[test]
fn exact_memory_match_is_compatible() {
    let device = GpuDeviceCapabilities::mock(); // 8 GB
    let reqs = ModelRequirements { memory_bytes: 8 * GB, ..ModelRequirements::minimal() };
    assert!(DeviceCapabilityChecker::check(&device, &reqs).is_compatible());
}

#[test]
fn minimal_requirements_on_mock_device() {
    let device = GpuDeviceCapabilities::mock();
    let reqs = ModelRequirements::minimal();
    assert!(DeviceCapabilityChecker::check(&device, &reqs).is_compatible());
}

#[test]
fn fp16_available_and_required() {
    let device = GpuDeviceCapabilities::mock(); // supports_fp16 = true
    let reqs = ModelRequirements { requires_fp16: true, ..ModelRequirements::minimal() };
    assert!(DeviceCapabilityChecker::check(&device, &reqs).is_compatible());
}

#[test]
fn subgroup_available_and_required() {
    let device = GpuDeviceCapabilities::mock(); // supports_subgroups = true
    let reqs = ModelRequirements {
        requires_subgroups: true,
        min_subgroup_size: 16,
        ..ModelRequirements::minimal()
    };
    assert!(DeviceCapabilityChecker::check(&device, &reqs).is_compatible());
}

// ── DeviceCapabilityChecker – incompatible ───────────────────────────────────

#[test]
fn model_too_large_rejected_with_reason() {
    let device = mock_with(|d| d.global_memory_bytes = 2 * GB);
    let reqs = ModelRequirements { memory_bytes: 4 * GB, ..ModelRequirements::minimal() };
    let result = DeviceCapabilityChecker::check(&device, &reqs);
    assert!(!result.is_compatible());
    assert_eq!(result.reasons().len(), 1);
    assert!(result.reasons()[0].contains("insufficient memory"));
}

#[test]
fn fp16_required_but_not_supported_rejected() {
    let device = mock_with(|d| d.supports_fp16 = false);
    let reqs = ModelRequirements { requires_fp16: true, ..ModelRequirements::minimal() };
    let result = DeviceCapabilityChecker::check(&device, &reqs);
    assert!(!result.is_compatible());
    assert!(result.reasons().iter().any(|r| r.contains("FP16")));
}

#[test]
fn fp64_required_but_not_supported_rejected() {
    let device = mock_with(|d| d.supports_fp64 = false);
    let reqs = ModelRequirements { requires_fp64: true, ..ModelRequirements::minimal() };
    let result = DeviceCapabilityChecker::check(&device, &reqs);
    assert!(!result.is_compatible());
    assert!(result.reasons().iter().any(|r| r.contains("FP64")));
}

#[test]
fn subgroups_required_but_not_supported_rejected() {
    let device = mock_with(|d| d.supports_subgroups = false);
    let reqs = ModelRequirements { requires_subgroups: true, ..ModelRequirements::minimal() };
    let result = DeviceCapabilityChecker::check(&device, &reqs);
    assert!(!result.is_compatible());
    assert!(result.reasons().iter().any(|r| r.contains("subgroup")));
}

#[test]
fn subgroup_size_too_small_rejected() {
    let device = mock_with(|d| d.max_subgroup_size = 8);
    let reqs = ModelRequirements {
        requires_subgroups: true,
        min_subgroup_size: 16,
        ..ModelRequirements::minimal()
    };
    let result = DeviceCapabilityChecker::check(&device, &reqs);
    assert!(!result.is_compatible());
    assert!(result.reasons().iter().any(|r| r.contains("subgroup size")));
}

#[test]
fn work_group_size_too_small_rejected() {
    let device = mock_with(|d| d.max_work_group_size = 32);
    let reqs = ModelRequirements { min_work_group_size: 256, ..ModelRequirements::minimal() };
    let result = DeviceCapabilityChecker::check(&device, &reqs);
    assert!(!result.is_compatible());
    assert!(result.reasons().iter().any(|r| r.contains("work-group")));
}

#[test]
fn multiple_failures_reported() {
    let device = mock_with(|d| {
        d.global_memory_bytes = 512 * 1024 * 1024; // 512 MB
        d.supports_fp16 = false;
        d.supports_subgroups = false;
    });
    let reqs = ModelRequirements {
        memory_bytes: 4 * GB,
        requires_fp16: true,
        requires_subgroups: true,
        ..ModelRequirements::minimal()
    };
    let result = DeviceCapabilityChecker::check(&device, &reqs);
    assert!(!result.is_compatible());
    assert!(result.reasons().len() >= 3);
}

// ── DeviceSelector – scoring ─────────────────────────────────────────────────

#[test]
fn score_increases_with_more_compute_units() {
    let low = mock_with(|d| d.compute_units = 16);
    let high = mock_with(|d| d.compute_units = 64);
    assert!(DeviceSelector::score(&high) > DeviceSelector::score(&low));
}

#[test]
fn score_increases_with_more_memory() {
    let low = mock_with(|d| d.global_memory_bytes = 4 * GB);
    let high = mock_with(|d| d.global_memory_bytes = 16 * GB);
    assert!(DeviceSelector::score(&high) > DeviceSelector::score(&low));
}

#[test]
fn score_increases_with_higher_clock() {
    let low = mock_with(|d| d.max_clock_mhz = 800);
    let high = mock_with(|d| d.max_clock_mhz = 2400);
    assert!(DeviceSelector::score(&high) > DeviceSelector::score(&low));
}

#[test]
fn fp16_bonus_in_score() {
    let without = mock_with(|d| d.supports_fp16 = false);
    let with = GpuDeviceCapabilities::mock(); // fp16 = true
    assert!(DeviceSelector::score(&with) > DeviceSelector::score(&without));
}

#[test]
fn subgroup_bonus_in_score() {
    let without = mock_with(|d| d.supports_subgroups = false);
    let with = GpuDeviceCapabilities::mock(); // subgroups = true
    assert!(DeviceSelector::score(&with) > DeviceSelector::score(&without));
}

#[test]
fn unified_memory_bonus_in_score() {
    let without = GpuDeviceCapabilities::mock(); // unified = false
    let with = mock_with(|d| d.supports_unified_memory = true);
    assert!(DeviceSelector::score(&with) > DeviceSelector::score(&without));
}

// ── DeviceSelector – select ──────────────────────────────────────────────────

#[test]
fn select_picks_highest_scored_device() {
    let weak = mock_with(|d| {
        d.name = "Weak GPU".into();
        d.compute_units = 8;
        d.global_memory_bytes = 2 * GB;
    });
    let strong = mock_with(|d| {
        d.name = "Strong GPU".into();
        d.compute_units = 64;
        d.global_memory_bytes = 16 * GB;
    });
    let mid = mock_with(|d| {
        d.name = "Mid GPU".into();
        d.compute_units = 32;
        d.global_memory_bytes = 8 * GB;
    });

    let result = DeviceSelector::select(vec![weak, mid, strong], None).unwrap();
    assert_eq!(result.device.name, "Strong GPU");
}

#[test]
fn select_filters_by_requirements() {
    let small = mock_with(|d| {
        d.name = "Small GPU".into();
        d.global_memory_bytes = 2 * GB;
        d.compute_units = 64; // highest CUs but too little memory
    });
    let big = mock_with(|d| {
        d.name = "Big GPU".into();
        d.global_memory_bytes = 16 * GB;
        d.compute_units = 32;
    });

    let reqs = ModelRequirements { memory_bytes: 8 * GB, ..ModelRequirements::minimal() };

    let result = DeviceSelector::select(vec![small, big], Some(&reqs)).unwrap();
    assert_eq!(result.device.name, "Big GPU");
}

#[test]
fn empty_device_list_returns_error() {
    let result = DeviceSelector::select(vec![], None);
    assert_eq!(result, Err(DeviceSelectorError::EmptyDeviceList));
}

#[test]
fn empty_device_list_error_display() {
    let err = DeviceSelectorError::EmptyDeviceList;
    assert_eq!(err.to_string(), "no devices provided for selection");
}

#[test]
fn no_compatible_device_returns_error() {
    let device = mock_with(|d| d.global_memory_bytes = 1 * GB);
    let reqs = ModelRequirements { memory_bytes: 32 * GB, ..ModelRequirements::minimal() };

    let result = DeviceSelector::select(vec![device], Some(&reqs));
    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceSelectorError::NoCompatibleDevice(reasons) => {
            assert!(!reasons.is_empty());
            assert!(reasons[0].contains("Mock GPU Device"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn no_compatible_device_error_display() {
    let err = DeviceSelectorError::NoCompatibleDevice(vec!["GPU A: too little memory".into()]);
    let msg = err.to_string();
    assert!(msg.contains("no compatible device found"));
    assert!(msg.contains("GPU A"));
}

#[test]
fn select_single_device_no_requirements() {
    let device = GpuDeviceCapabilities::mock();
    let result = DeviceSelector::select(vec![device.clone()], None).unwrap();
    assert_eq!(result.device, device);
}

// ── DeviceSelector – rank ────────────────────────────────────────────────────

#[test]
fn rank_sorts_descending_by_score() {
    let devices = vec![
        mock_with(|d| {
            d.name = "Low".into();
            d.compute_units = 4;
        }),
        mock_with(|d| {
            d.name = "High".into();
            d.compute_units = 128;
        }),
        mock_with(|d| {
            d.name = "Mid".into();
            d.compute_units = 32;
        }),
    ];

    let ranked = DeviceSelector::rank(devices);
    assert_eq!(ranked.len(), 3);
    assert_eq!(ranked[0].device.name, "High");
    assert_eq!(ranked[1].device.name, "Mid");
    assert_eq!(ranked[2].device.name, "Low");
    assert!(ranked[0].score >= ranked[1].score);
    assert!(ranked[1].score >= ranked[2].score);
}

#[test]
fn rank_empty_list_returns_empty() {
    let ranked = DeviceSelector::rank(vec![]);
    assert!(ranked.is_empty());
}

// ── format_device_info ───────────────────────────────────────────────────────

#[test]
fn format_includes_device_name() {
    let info = format_device_info(&GpuDeviceCapabilities::mock());
    assert!(info.contains("Mock GPU Device"));
}

#[test]
fn format_includes_vendor() {
    let info = format_device_info(&GpuDeviceCapabilities::mock());
    assert!(info.contains("MockVendor"));
}

#[test]
fn format_includes_backend() {
    let info = format_device_info(&GpuDeviceCapabilities::mock());
    assert!(info.contains("oneapi"));
}

#[test]
fn format_includes_memory() {
    let info = format_device_info(&GpuDeviceCapabilities::mock());
    assert!(info.contains("8.00 GB"));
}

#[test]
fn format_includes_compute_units() {
    let info = format_device_info(&GpuDeviceCapabilities::mock());
    assert!(info.contains("32"));
}

#[test]
fn format_includes_fp16_status() {
    let info = format_device_info(&GpuDeviceCapabilities::mock());
    assert!(info.contains("FP16"));
    assert!(info.contains("true"));
}

#[test]
fn format_includes_subgroup_info() {
    let info = format_device_info(&GpuDeviceCapabilities::mock());
    assert!(info.contains("Subgroups"));
}

#[test]
fn format_includes_unified_memory() {
    let info = format_device_info(&GpuDeviceCapabilities::mock());
    assert!(info.contains("Unified memory"));
}

#[test]
fn format_includes_all_fields() {
    let info = format_device_info(&GpuDeviceCapabilities::mock());
    for expected in [
        "Device:",
        "Vendor:",
        "Driver:",
        "Backend:",
        "Compute units:",
        "Max clock:",
        "Global memory:",
        "Local memory:",
        "Max workgroup:",
        "Work-item dims:",
        "Vec width",
        "FP16:",
        "FP64:",
        "Subgroups:",
        "Unified memory:",
    ] {
        assert!(info.contains(expected), "format output missing '{expected}'");
    }
}

// ── ScoredDevice ─────────────────────────────────────────────────────────────

#[test]
fn scored_device_debug_format() {
    let sd = ScoredDevice { device: GpuDeviceCapabilities::mock(), score: 42.0 };
    let debug = format!("{sd:?}");
    assert!(debug.contains("42.0"));
}

#[test]
fn scored_device_clone() {
    let sd = ScoredDevice { device: GpuDeviceCapabilities::mock(), score: 10.0 };
    let sd2 = sd.clone();
    assert_eq!(sd.score, sd2.score);
    assert_eq!(sd.device, sd2.device);
}

// ── ModelRequirements ────────────────────────────────────────────────────────

#[test]
fn minimal_requirements_defaults() {
    let reqs = ModelRequirements::minimal();
    assert_eq!(reqs.memory_bytes, 1 * GB);
    assert!(!reqs.requires_fp16);
    assert!(!reqs.requires_fp64);
    assert!(!reqs.requires_subgroups);
    assert_eq!(reqs.min_subgroup_size, 0);
    assert_eq!(reqs.min_work_group_size, 64);
}
