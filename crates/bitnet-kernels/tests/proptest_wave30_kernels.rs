//! Property-based tests — wave 30 (kernels).
//!
//! Covers: perf_tracker aggregation invariants, SIMD diagnostics consistency,
//! capability matrix queries, kernel timing arithmetic, dispatch
//! recommendations, and shaped reduction properties.
//!
//! 40+ property tests validating: timing arithmetic, tracker monotonicity,
//! capability matrix consistency, SIMD level ordering, dispatch completeness,
//! and performance report formatting.

#![cfg(feature = "cpu")]

use std::collections::HashSet;
use std::time::Duration;

use bitnet_kernels::capability_matrix::{
    CapabilityEntry, CapabilityQuery, CompatibilityReport, DeviceCapabilityMatrix, DeviceClass,
    DeviceProfile, OperationCategory, PrecisionSupport, SupportLevel,
};
use bitnet_kernels::perf_tracker::{KernelTiming, PerfTracker, format_perf_report};
use bitnet_kernels::simd_diagnostics::{SimdCapabilities, SimdLevel, recommend_dispatch};
use proptest::prelude::*;

// ── Strategy helpers ────────────────────────────────────────────────────────

fn arb_duration() -> impl Strategy<Value = Duration> {
    (1u64..10_000_000).prop_map(Duration::from_nanos)
}

fn arb_kernel_name() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-z_]{1,20}").unwrap().prop_filter("non-empty", |s| !s.is_empty())
}

fn arb_timing() -> impl Strategy<Value = KernelTiming> {
    (arb_kernel_name(), arb_duration(), 1usize..1_000_000)
        .prop_map(|(name, dur, elems)| KernelTiming::new(&name, dur, elems))
}

fn arb_timing_with_flops() -> impl Strategy<Value = KernelTiming> {
    (arb_kernel_name(), arb_duration(), 1usize..1_000_000, 1u64..1_000_000_000).prop_map(
        |(name, dur, elems, flops)| KernelTiming::new(&name, dur, elems).with_flops(flops),
    )
}

fn arb_simd_caps() -> impl Strategy<Value = SimdCapabilities> {
    (
        any::<bool>(), // sse2
        any::<bool>(), // sse4_1
        any::<bool>(), // sse4_2
        any::<bool>(), // avx
        any::<bool>(), // avx2
        any::<bool>(), // avx512f
        any::<bool>(), // avx512bw
        any::<bool>(), // avx512vnni
        any::<bool>(), // fma
        any::<bool>(), // neon
    )
        .prop_map(
            |(sse2, sse4_1, sse4_2, avx, avx2, avx512f, avx512bw, avx512vnni, fma, neon)| {
                SimdCapabilities {
                    sse2,
                    sse4_1,
                    sse4_2,
                    avx,
                    avx2,
                    avx512f,
                    avx512bw,
                    avx512vnni,
                    fma,
                    neon,
                    arch: "x86_64".to_string(),
                }
            },
        )
}

// ── KernelTiming properties ─────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Throughput is always non-negative for valid timings.
    #[test]
    fn timing_throughput_non_negative(t in arb_timing()) {
        prop_assert!(t.throughput() >= 0.0);
    }

    /// Throughput is finite for non-zero durations.
    #[test]
    fn timing_throughput_finite(t in arb_timing()) {
        prop_assert!(t.throughput().is_finite());
    }

    /// GFlops is None when flops not set.
    #[test]
    fn timing_no_flops_returns_none(t in arb_timing()) {
        prop_assert!(t.gflops().is_none());
    }

    /// GFlops is Some and finite when flops are set.
    #[test]
    fn timing_with_flops_returns_finite_gflops(t in arb_timing_with_flops()) {
        let g = t.gflops();
        prop_assert!(g.is_some());
        prop_assert!(g.unwrap().is_finite());
    }

    /// GFlops value is positive for positive flops and positive duration.
    #[test]
    fn timing_gflops_positive(t in arb_timing_with_flops()) {
        prop_assert!(t.gflops().unwrap() > 0.0);
    }

    /// Throughput scales linearly with element count.
    #[test]
    fn timing_throughput_scales_with_elements(
        name in arb_kernel_name(),
        dur in arb_duration(),
        elems in 1usize..500_000,
    ) {
        let t1 = KernelTiming::new(&name, dur, elems);
        let t2 = KernelTiming::new(&name, dur, elems * 2);
        let ratio = t2.throughput() / t1.throughput();
        prop_assert!((ratio - 2.0).abs() < 1e-6);
    }

    /// Duration is preserved in constructor.
    #[test]
    fn timing_preserves_duration(dur_ns in 1u64..10_000_000, elems in 1usize..1000) {
        let dur = Duration::from_nanos(dur_ns);
        let t = KernelTiming::new("test", dur, elems);
        prop_assert_eq!(t.duration, dur);
    }

    /// Input elements are preserved.
    #[test]
    fn timing_preserves_elements(elems in 1usize..1_000_000) {
        let t = KernelTiming::new("test", Duration::from_nanos(100), elems);
        prop_assert_eq!(t.input_elements, elems);
    }
}

// ── PerfTracker properties ──────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Count equals number of recorded timings.
    #[test]
    fn tracker_count_matches_records(timings in prop::collection::vec(arb_timing(), 0..50)) {
        let mut tracker = PerfTracker::new();
        for t in &timings {
            tracker.record(t.clone());
        }
        prop_assert_eq!(tracker.count(), timings.len());
    }

    /// Total time is sum of individual durations.
    #[test]
    fn tracker_total_time_is_sum(timings in prop::collection::vec(arb_timing(), 1..30)) {
        let mut tracker = PerfTracker::new();
        let expected: Duration = timings.iter().map(|t| t.duration).sum();
        for t in timings {
            tracker.record(t);
        }
        prop_assert_eq!(tracker.total_time(), expected);
    }

    /// Slowest timing has maximum duration.
    #[test]
    fn tracker_slowest_has_max_duration(timings in prop::collection::vec(arb_timing(), 1..30)) {
        let mut tracker = PerfTracker::new();
        let max_dur = timings.iter().map(|t| t.duration).max().unwrap();
        for t in timings {
            tracker.record(t);
        }
        prop_assert_eq!(tracker.slowest().unwrap().duration, max_dur);
    }

    /// Fastest timing has minimum duration.
    #[test]
    fn tracker_fastest_has_min_duration(timings in prop::collection::vec(arb_timing(), 1..30)) {
        let mut tracker = PerfTracker::new();
        let min_dur = timings.iter().map(|t| t.duration).min().unwrap();
        for t in timings {
            tracker.record(t);
        }
        prop_assert_eq!(tracker.fastest().unwrap().duration, min_dur);
    }

    /// Clear resets count to zero.
    #[test]
    fn tracker_clear_resets(timings in prop::collection::vec(arb_timing(), 1..20)) {
        let mut tracker = PerfTracker::new();
        for t in timings {
            tracker.record(t);
        }
        tracker.clear();
        prop_assert_eq!(tracker.count(), 0);
        prop_assert_eq!(tracker.total_time(), Duration::ZERO);
    }

    /// by_kernel groups timings correctly: total entries across groups equals count.
    #[test]
    fn tracker_by_kernel_total_matches_count(timings in prop::collection::vec(arb_timing(), 1..30)) {
        let mut tracker = PerfTracker::new();
        for t in &timings {
            tracker.record(t.clone());
        }
        let grouped = tracker.by_kernel();
        let total: usize = grouped.values().map(|v| v.len()).sum();
        prop_assert_eq!(total, tracker.count());
    }

    /// kernel_stats returns one entry per distinct kernel name.
    #[test]
    fn tracker_stats_one_per_kernel(timings in prop::collection::vec(arb_timing(), 1..30)) {
        let mut tracker = PerfTracker::new();
        let unique_names: HashSet<_> = timings.iter().map(|t| t.kernel_name.clone()).collect();
        for t in timings {
            tracker.record(t);
        }
        let stats = tracker.kernel_stats();
        prop_assert_eq!(stats.len(), unique_names.len());
    }

    /// Each kernel stat's total_time <= tracker total_time.
    #[test]
    fn tracker_stat_time_bounded(timings in prop::collection::vec(arb_timing(), 1..30)) {
        let mut tracker = PerfTracker::new();
        for t in timings {
            tracker.record(t);
        }
        let total = tracker.total_time();
        for stat in tracker.kernel_stats() {
            prop_assert!(stat.total_time <= total);
        }
    }

    /// Each kernel stat has avg_time between min_time and max_time.
    #[test]
    fn tracker_stat_avg_between_min_max(timings in prop::collection::vec(arb_timing(), 1..30)) {
        let mut tracker = PerfTracker::new();
        for t in timings {
            tracker.record(t);
        }
        for stat in tracker.kernel_stats() {
            prop_assert!(stat.avg_time >= stat.min_time);
            prop_assert!(stat.avg_time <= stat.max_time);
        }
    }

    /// Each kernel stat count > 0.
    #[test]
    fn tracker_stat_count_positive(timings in prop::collection::vec(arb_timing(), 1..30)) {
        let mut tracker = PerfTracker::new();
        for t in timings {
            tracker.record(t);
        }
        for stat in tracker.kernel_stats() {
            prop_assert!(stat.count > 0);
        }
    }

    /// format_perf_report does not panic and produces non-empty output for non-empty tracker.
    #[test]
    fn tracker_report_non_empty(timings in prop::collection::vec(arb_timing(), 1..10)) {
        let mut tracker = PerfTracker::new();
        for t in timings {
            tracker.record(t);
        }
        let report = format_perf_report(&tracker);
        prop_assert!(!report.is_empty());
    }

    /// Empty tracker has zero total time.
    #[test]
    fn tracker_empty_total_zero(_seed in 0u32..100) {
        let tracker = PerfTracker::new();
        prop_assert_eq!(tracker.total_time(), Duration::ZERO);
        prop_assert_eq!(tracker.count(), 0);
        prop_assert!(tracker.slowest().is_none());
        prop_assert!(tracker.fastest().is_none());
    }
}

// ── SIMD diagnostics properties ─────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// vector_width_bits is always a power of two >= 64.
    #[test]
    fn simd_width_power_of_two(caps in arb_simd_caps()) {
        let width = caps.vector_width_bits();
        prop_assert!(width >= 64);
        prop_assert!(width.is_power_of_two());
    }

    /// f32_lanes equals vector_width_bits / 32.
    #[test]
    fn simd_f32_lanes_consistent(caps in arb_simd_caps()) {
        prop_assert_eq!(caps.f32_lanes(), caps.vector_width_bits() / 32);
    }

    /// best_level is Avx512 when avx512f is true.
    #[test]
    fn simd_avx512_implies_best_level(mut caps in arb_simd_caps()) {
        caps.avx512f = true;
        prop_assert_eq!(caps.best_level(), SimdLevel::Avx512);
    }

    /// best_level is at least Avx2 when avx2 is true and avx512f is false.
    #[test]
    fn simd_avx2_implies_at_least_avx2(mut caps in arb_simd_caps()) {
        caps.avx512f = false;
        caps.avx2 = true;
        prop_assert_eq!(caps.best_level(), SimdLevel::Avx2);
    }

    /// Scalar level when everything is false.
    #[test]
    fn simd_all_false_is_scalar(_seed in 0u32..100) {
        let caps = SimdCapabilities {
            sse2: false, sse4_1: false, sse4_2: false,
            avx: false, avx2: false, avx512f: false,
            avx512bw: false, avx512vnni: false,
            fma: false, neon: false,
            arch: "x86_64".to_string(),
        };
        prop_assert_eq!(caps.best_level(), SimdLevel::Scalar);
        prop_assert_eq!(caps.vector_width_bits(), 64);
        prop_assert_eq!(caps.f32_lanes(), 2);
    }

    /// summary() is non-empty for any capability set.
    #[test]
    fn simd_summary_non_empty(caps in arb_simd_caps()) {
        prop_assert!(!caps.summary().is_empty());
    }

    /// summary() contains the arch string.
    #[test]
    fn simd_summary_contains_arch(caps in arb_simd_caps()) {
        prop_assert!(caps.summary().contains(&caps.arch));
    }

    /// recommend_dispatch never panics and returns non-empty vec.
    #[test]
    fn simd_dispatch_recommendations_non_empty(caps in arb_simd_caps()) {
        let recs = recommend_dispatch(&caps);
        prop_assert!(!recs.is_empty());
    }

    /// All dispatch recommendations have non-empty operation names.
    #[test]
    fn simd_dispatch_operations_non_empty(caps in arb_simd_caps()) {
        let recs = recommend_dispatch(&caps);
        for rec in &recs {
            prop_assert!(!rec.operation.is_empty());
            prop_assert!(!rec.reason.is_empty());
        }
    }

    /// SimdLevel ordering: Scalar < Sse2 < Sse42 < Neon < Avx < Avx2 < Avx512.
    #[test]
    fn simd_level_ordering(_seed in 0u32..10) {
        prop_assert!(SimdLevel::Scalar < SimdLevel::Sse2);
        prop_assert!(SimdLevel::Sse2 < SimdLevel::Sse42);
        prop_assert!(SimdLevel::Sse42 < SimdLevel::Neon);
        prop_assert!(SimdLevel::Neon < SimdLevel::Avx);
        prop_assert!(SimdLevel::Avx < SimdLevel::Avx2);
        prop_assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
    }

    /// display_name never returns empty string.
    #[test]
    fn simd_level_display_name_non_empty(
        idx in 0usize..7,
    ) {
        let levels = [
            SimdLevel::Scalar, SimdLevel::Sse2, SimdLevel::Sse42,
            SimdLevel::Neon, SimdLevel::Avx, SimdLevel::Avx2, SimdLevel::Avx512,
        ];
        prop_assert!(!levels[idx].display_name().is_empty());
    }
}

// ── Capability matrix properties ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Builtin matrix has at least one profile.
    #[test]
    fn capability_matrix_has_profiles(_seed in 0u32..10) {
        let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
        prop_assert!(!matrix.profiles().is_empty());
    }

    /// Every builtin profile has at least one capability entry.
    #[test]
    fn capability_profile_has_entries(_seed in 0u32..10) {
        let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
        for profile in matrix.profiles() {
            prop_assert!(!profile.capabilities.is_empty());
        }
    }

    /// SupportLevel::Full is always supported.
    #[test]
    fn support_level_full_is_supported(eff in 0.0f64..1.0) {
        prop_assert!(SupportLevel::Full(eff).is_supported());
    }

    /// SupportLevel::Unsupported is never supported.
    #[test]
    fn support_level_unsupported_not_supported(_seed in 0u32..10) {
        prop_assert!(!SupportLevel::Unsupported.is_supported());
    }

    /// SupportLevel::Partial is supported.
    #[test]
    fn support_level_partial_is_supported(_seed in 0u32..10) {
        prop_assert!(SupportLevel::Partial("test".to_string()).is_supported());
    }

    /// SupportLevel::Emulated is supported.
    #[test]
    fn support_level_emulated_is_supported(_seed in 0u32..10) {
        prop_assert!(SupportLevel::Emulated.is_supported());
    }

    /// SupportLevel::Full efficiency is returned.
    #[test]
    fn support_level_full_efficiency(eff in 0.0f64..1.0) {
        let level = SupportLevel::Full(eff);
        prop_assert_eq!(level.efficiency(), Some(eff));
    }

    /// SupportLevel::Unsupported has no efficiency.
    #[test]
    fn support_level_unsupported_no_efficiency(_seed in 0u32..10) {
        prop_assert_eq!(SupportLevel::Unsupported.efficiency(), None);
    }

    /// full_support_count <= total capabilities for any profile.
    #[test]
    fn capability_full_count_bounded(_seed in 0u32..10) {
        let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
        for profile in matrix.profiles() {
            prop_assert!(profile.full_support_count() <= profile.capabilities.len());
        }
    }

    /// CompatibilityReport.generate never panics for builtin profiles.
    #[test]
    fn capability_compat_report_does_not_panic(_seed in 0u32..10) {
        let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
        let required = &[(OperationCategory::MatrixOps, PrecisionSupport::FP32)];
        for profile in matrix.profiles() {
            let report = CompatibilityReport::generate(profile, required);
            prop_assert!(!report.summary().is_empty());
        }
    }

    /// CapabilityQuery.supports is consistent with support_level.
    #[test]
    fn capability_query_supports_consistent(_seed in 0u32..10) {
        let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
        for profile in matrix.profiles() {
            let query = CapabilityQuery::new(profile);
            for entry in &profile.capabilities {
                let supported = query.supports(entry.operation, entry.precision);
                let level = query.support_level(entry.operation, entry.precision);
                prop_assert_eq!(supported, level.is_supported());
            }
        }
    }
}
