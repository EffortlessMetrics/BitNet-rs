//! CPU capability detection for kernel provider selection.

/// Detected CPU execution tier in descending performance preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CpuExecutionTier {
    Avx512,
    Avx2,
    Neon,
    Scalar,
}

impl CpuExecutionTier {
    /// Lower is better (higher priority).
    pub const fn priority(self) -> u8 {
        match self {
            Self::Avx512 => 0,
            Self::Avx2 => 1,
            Self::Neon => 2,
            Self::Scalar => 3,
        }
    }
}

/// Returns true when AVX-512 is both compiled and available at runtime.
#[must_use]
pub fn avx512_available() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        return is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
    }

    #[allow(unreachable_code)]
    false
}

/// Returns true when AVX2 is both compiled and available at runtime.
#[must_use]
pub fn avx2_available() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        return is_x86_feature_detected!("avx2");
    }

    #[allow(unreachable_code)]
    false
}

/// Returns true when AVX2 and FMA are both compiled and available at runtime.
#[must_use]
pub fn avx2_fma_available() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        return is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    }

    #[allow(unreachable_code)]
    false
}

/// Returns true when NEON is both compiled and available at runtime.
#[must_use]
pub fn neon_available() -> bool {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        return std::arch::is_aarch64_feature_detected!("neon");
    }

    #[allow(unreachable_code)]
    false
}

/// Detect all available CPU tiers ordered by preference.
#[must_use]
pub fn available_tiers() -> Vec<CpuExecutionTier> {
    let mut tiers = vec![CpuExecutionTier::Scalar];

    if avx512_available() {
        tiers.insert(0, CpuExecutionTier::Avx512);
    }

    if avx2_available() {
        let insert_pos = tiers.len().saturating_sub(1);
        tiers.insert(insert_pos, CpuExecutionTier::Avx2);
    }

    if neon_available() {
        let insert_pos = tiers.len().saturating_sub(1);
        tiers.insert(insert_pos, CpuExecutionTier::Neon);
    }

    tiers.sort_by_key(|tier| tier.priority());
    tiers.dedup();
    tiers
}

/// Detects the best available tier for the current process.
#[must_use]
pub fn best_tier() -> CpuExecutionTier {
    available_tiers().into_iter().next().unwrap_or(CpuExecutionTier::Scalar)
}

#[cfg(test)]
mod tests {
    use super::{
        CpuExecutionTier, available_tiers, avx2_available, avx2_fma_available, avx512_available,
        best_tier, neon_available,
    };

    #[test]
    fn scalar_tier_is_always_present() {
        let tiers = available_tiers();
        assert!(tiers.contains(&CpuExecutionTier::Scalar));
    }

    #[test]
    fn available_tiers_are_sorted_by_priority() {
        let tiers = available_tiers();
        for pair in tiers.windows(2) {
            assert!(pair[0].priority() <= pair[1].priority());
        }
    }

    #[test]
    fn best_tier_matches_first_available_tier() {
        let tiers = available_tiers();
        let first = tiers.first().copied().unwrap_or(CpuExecutionTier::Scalar);
        assert_eq!(best_tier(), first);
    }

    #[test]
    fn avx2_fma_implies_avx2() {
        if avx2_fma_available() {
            assert!(avx2_available());
        }
    }

    #[test]
    fn priority_values_are_monotonic_and_distinct() {
        // Lower priority number = higher preference. Distinct values let
        // `available_tiers()` dedup correctly after sort.
        assert_eq!(CpuExecutionTier::Avx512.priority(), 0);
        assert_eq!(CpuExecutionTier::Avx2.priority(), 1);
        assert_eq!(CpuExecutionTier::Neon.priority(), 2);
        assert_eq!(CpuExecutionTier::Scalar.priority(), 3);

        // Strictly monotonic: every higher-tier variant beats the next.
        assert!(CpuExecutionTier::Avx512.priority() < CpuExecutionTier::Avx2.priority());
        assert!(CpuExecutionTier::Avx2.priority() < CpuExecutionTier::Neon.priority());
        assert!(CpuExecutionTier::Neon.priority() < CpuExecutionTier::Scalar.priority());
    }

    #[test]
    fn cpu_execution_tier_ord_follows_enum_order() {
        // The derived Ord uses enum declaration order; verify the
        // ordering matches the priority numbering (lower = better).
        assert!(CpuExecutionTier::Avx512 < CpuExecutionTier::Avx2);
        assert!(CpuExecutionTier::Avx2 < CpuExecutionTier::Neon);
        assert!(CpuExecutionTier::Neon < CpuExecutionTier::Scalar);
    }

    #[test]
    fn cpu_execution_tier_is_copy_and_eq() {
        let t = CpuExecutionTier::Avx2;
        let copy = t;
        assert_eq!(t, copy);
        // Debug should not panic.
        let _ = format!("{t:?}");
    }

    #[test]
    fn available_tiers_are_deduplicated() {
        let tiers = available_tiers();
        let mut sorted = tiers.clone();
        sorted.sort_by_key(|t| t.priority());
        sorted.dedup();
        assert_eq!(tiers, sorted, "available_tiers should already be sorted and deduped");
    }

    #[test]
    fn runtime_detection_helpers_return_bool() {
        // These return false on platforms / feature combinations where the
        // ISA is not compiled or unavailable, but must always be callable.
        let _ = avx2_available();
        let _ = avx2_fma_available();
        let _ = avx512_available();
        let _ = neon_available();
    }

    #[test]
    fn neon_implies_aarch64() {
        // The runtime check returns true only when both compiled-in via the
        // `neon` feature *and* the target is aarch64 *and* the OS reports
        // NEON. So if neon_available() ever returns true, the target_arch
        // must be aarch64.
        if neon_available() {
            assert_eq!(std::env::consts::ARCH, "aarch64");
        }
    }

    #[test]
    fn avx_implies_x86_64() {
        if avx2_available() || avx512_available() {
            assert_eq!(std::env::consts::ARCH, "x86_64");
        }
    }

    #[test]
    fn available_tiers_strictly_sorted() {
        // Stronger than monotonic: after dedup, priorities are strictly
        // increasing.
        let tiers = available_tiers();
        for pair in tiers.windows(2) {
            assert!(pair[0].priority() < pair[1].priority(), "priorities should be unique");
        }
    }
}
