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
    use super::{CpuExecutionTier, available_tiers, avx2_available, avx2_fma_available, best_tier};

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
}
