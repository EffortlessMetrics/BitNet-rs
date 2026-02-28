//! Intel subgroup capability detection and kernel dispatch.
//!
//! Queries `CL_DEVICE_SUB_GROUP_SIZES_INTEL` to determine whether the OpenCL
//! device supports the `cl_intel_subgroups` extension. If supported, subgroup-
//! optimized kernels (matmul, reduce, softmax) are dispatched; otherwise we
//! fall back to the regular (non-subgroup) kernel variants.

/// Subgroup sizes reported by the device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupCapabilities {
    /// Whether `cl_intel_subgroups` extension is available.
    pub intel_subgroups_supported: bool,
    /// Supported subgroup sizes (e.g., `[8, 16, 32]`).
    pub supported_sizes: Vec<u32>,
    /// Preferred subgroup size for this device (typically 16 on Intel Arc).
    pub preferred_size: u32,
}

impl SubgroupCapabilities {
    /// Query subgroup capabilities from an OpenCL device.
    ///
    /// In a real implementation this would call `clGetDeviceInfo` with
    /// `CL_DEVICE_SUB_GROUP_SIZES_INTEL`. This stub returns a detection
    /// result based on the provided extension list.
    pub fn detect(extensions: &str) -> Self {
        let supported = extensions.contains("cl_intel_subgroups");
        if supported {
            Self {
                intel_subgroups_supported: true,
                // Intel Arc typically supports SIMD8, SIMD16, SIMD32
                supported_sizes: vec![8, 16, 32],
                preferred_size: 16,
            }
        } else {
            Self { intel_subgroups_supported: false, supported_sizes: vec![], preferred_size: 0 }
        }
    }

    /// Detect from a list of supported subgroup sizes (e.g., from
    /// `CL_DEVICE_SUB_GROUP_SIZES_INTEL`).
    pub fn from_sizes(sizes: Vec<u32>, has_extension: bool) -> Self {
        let preferred = if sizes.contains(&16) { 16 } else { sizes.first().copied().unwrap_or(0) };
        Self {
            intel_subgroups_supported: has_extension && !sizes.is_empty(),
            supported_sizes: sizes,
            preferred_size: preferred,
        }
    }

    /// Whether we should use subgroup-optimized kernels.
    pub fn use_subgroup_kernels(&self) -> bool {
        self.intel_subgroups_supported && self.preferred_size > 0
    }
}

/// Select the appropriate matmul kernel source based on capabilities.
pub fn select_matmul_kernel(caps: &SubgroupCapabilities) -> &'static str {
    if caps.use_subgroup_kernels() {
        include_str!("kernels/subgroup_matmul.cl")
    } else {
        include_str!("kernels/matmul_i2s.cl")
    }
}

/// Select the appropriate reduction kernel source.
pub fn select_reduce_kernel(caps: &SubgroupCapabilities) -> &'static str {
    if caps.use_subgroup_kernels() {
        include_str!("kernels/subgroup_reduce.cl")
    } else {
        include_str!("kernels/elementwise.cl")
    }
}

/// Select the appropriate softmax kernel source.
pub fn select_softmax_kernel(caps: &SubgroupCapabilities) -> &'static str {
    if caps.use_subgroup_kernels() {
        include_str!("kernels/subgroup_softmax.cl")
    } else {
        include_str!("kernels/elementwise.cl")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_with_intel_subgroups() {
        let caps = SubgroupCapabilities::detect(
            "cl_khr_fp16 cl_intel_subgroups cl_intel_required_subgroup_size",
        );
        assert!(caps.intel_subgroups_supported);
        assert_eq!(caps.preferred_size, 16);
        assert!(caps.supported_sizes.contains(&16));
        assert!(caps.use_subgroup_kernels());
    }

    #[test]
    fn test_detect_without_intel_subgroups() {
        let caps = SubgroupCapabilities::detect("cl_khr_fp16 cl_khr_fp64");
        assert!(!caps.intel_subgroups_supported);
        assert_eq!(caps.preferred_size, 0);
        assert!(caps.supported_sizes.is_empty());
        assert!(!caps.use_subgroup_kernels());
    }

    #[test]
    fn test_from_sizes_with_simd16() {
        let caps = SubgroupCapabilities::from_sizes(vec![8, 16, 32], true);
        assert!(caps.intel_subgroups_supported);
        assert_eq!(caps.preferred_size, 16);
        assert!(caps.use_subgroup_kernels());
    }

    #[test]
    fn test_from_sizes_without_simd16() {
        let caps = SubgroupCapabilities::from_sizes(vec![8, 32], true);
        assert!(caps.intel_subgroups_supported);
        assert_eq!(caps.preferred_size, 8);
        assert!(caps.use_subgroup_kernels());
    }

    #[test]
    fn test_from_sizes_empty() {
        let caps = SubgroupCapabilities::from_sizes(vec![], false);
        assert!(!caps.intel_subgroups_supported);
        assert!(!caps.use_subgroup_kernels());
    }

    #[test]
    fn test_select_matmul_kernel_subgroup() {
        let caps = SubgroupCapabilities::detect("cl_intel_subgroups");
        let src = select_matmul_kernel(&caps);
        assert!(src.contains("subgroup_matmul"));
    }

    #[test]
    fn test_select_matmul_kernel_fallback() {
        let caps = SubgroupCapabilities::detect("");
        let src = select_matmul_kernel(&caps);
        assert!(src.contains("matmul_i2s"));
    }

    #[test]
    fn test_select_reduce_kernel_subgroup() {
        let caps = SubgroupCapabilities::detect("cl_intel_subgroups");
        let src = select_reduce_kernel(&caps);
        assert!(src.contains("subgroup_reduce_sum") || src.contains("reduce_sum"));
    }

    #[test]
    fn test_select_softmax_kernel_subgroup() {
        let caps = SubgroupCapabilities::detect("cl_intel_subgroups");
        let src = select_softmax_kernel(&caps);
        assert!(src.contains("subgroup_softmax"));
    }

    #[test]
    fn test_select_softmax_kernel_fallback() {
        let caps = SubgroupCapabilities::detect("cl_khr_fp16");
        let src = select_softmax_kernel(&caps);
        assert!(src.contains("vec_add") || src.contains("silu"));
    }
}
