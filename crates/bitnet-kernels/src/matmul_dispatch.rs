//! Matrix multiplication dispatch and strategy selection.
//!
//! Select optimal matmul implementation based on matrix dimensions,
//! available SIMD features, and precision requirements.

/// Matmul backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulBackend {
    Scalar,
    Avx2,
    Avx512,
    Neon,
    Cuda,
    OpenCl,
}

impl MatmulBackend {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "AVX2",
            Self::Avx512 => "AVX-512",
            Self::Neon => "NEON",
            Self::Cuda => "CUDA",
            Self::OpenCl => "OpenCL",
        }
    }

    pub fn is_simd(&self) -> bool {
        matches!(self, Self::Avx2 | Self::Avx512 | Self::Neon)
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Cuda | Self::OpenCl)
    }
}

/// Precision requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulPrecision {
    Full,  // f32 accumulation
    Mixed, // f16 multiply + f32 accumulate
    Half,  // f16 throughout
}

/// Matrix dimensions for dispatch.
#[derive(Debug, Clone, Copy)]
pub struct MatmulShape {
    pub m: usize, // rows of output
    pub n: usize, // cols of output
    pub k: usize, // inner dimension
}

impl MatmulShape {
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k }
    }
    pub fn output_elements(&self) -> usize {
        self.m * self.n
    }
    pub fn flops(&self) -> usize {
        2 * self.m * self.n * self.k
    }

    pub fn is_small(&self) -> bool {
        self.m * self.n * self.k < 1024
    }

    pub fn is_large(&self) -> bool {
        self.m * self.n * self.k > 1_000_000
    }

    /// Whether the shape benefits from tiling.
    pub fn benefits_from_tiling(&self) -> bool {
        self.m >= 16 && self.n >= 16 && self.k >= 16
    }
}

/// Tiling configuration.
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
}

impl TileConfig {
    pub fn for_backend(backend: MatmulBackend, shape: &MatmulShape) -> Self {
        match backend {
            MatmulBackend::Avx2 => Self { tile_m: 8, tile_n: 8, tile_k: 16 },
            MatmulBackend::Avx512 => Self { tile_m: 16, tile_n: 16, tile_k: 16 },
            MatmulBackend::Neon => Self { tile_m: 4, tile_n: 4, tile_k: 16 },
            MatmulBackend::Cuda => {
                Self { tile_m: 128.min(shape.m), tile_n: 128.min(shape.n), tile_k: 32 }
            }
            _ => Self { tile_m: 4, tile_n: 4, tile_k: 4 },
        }
    }

    pub fn num_tiles(&self, shape: &MatmulShape) -> usize {
        let tm = shape.m.div_ceil(self.tile_m);
        let tn = shape.n.div_ceil(self.tile_n);
        let tk = shape.k.div_ceil(self.tile_k);
        tm * tn * tk
    }
}

/// Dispatch decision.
#[derive(Debug, Clone)]
pub struct DispatchDecision {
    pub backend: MatmulBackend,
    pub tiling: Option<TileConfig>,
    pub precision: MatmulPrecision,
    pub rationale: String,
}

/// Available hardware features.
#[derive(Debug, Clone, Default)]
pub struct HardwareFeatures {
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_neon: bool,
    pub has_cuda: bool,
    pub has_opencl: bool,
}

/// Select the best matmul backend for given shape and hardware.
pub fn select_backend(
    shape: &MatmulShape,
    features: &HardwareFeatures,
    precision: MatmulPrecision,
) -> DispatchDecision {
    // GPU for large matrices
    if shape.is_large() {
        if features.has_cuda {
            let tiling = TileConfig::for_backend(MatmulBackend::Cuda, shape);
            return DispatchDecision {
                backend: MatmulBackend::Cuda,
                tiling: Some(tiling),
                precision,
                rationale: "large matrix → CUDA".into(),
            };
        }
        if features.has_opencl {
            let tiling = TileConfig::for_backend(MatmulBackend::OpenCl, shape);
            return DispatchDecision {
                backend: MatmulBackend::OpenCl,
                tiling: Some(tiling),
                precision,
                rationale: "large matrix → OpenCL".into(),
            };
        }
    }

    // SIMD for medium/large matrices on CPU
    if shape.benefits_from_tiling() {
        if features.has_avx512 {
            let tiling = TileConfig::for_backend(MatmulBackend::Avx512, shape);
            return DispatchDecision {
                backend: MatmulBackend::Avx512,
                tiling: Some(tiling),
                precision,
                rationale: "tileable matrix → AVX-512".into(),
            };
        }
        if features.has_avx2 {
            let tiling = TileConfig::for_backend(MatmulBackend::Avx2, shape);
            return DispatchDecision {
                backend: MatmulBackend::Avx2,
                tiling: Some(tiling),
                precision,
                rationale: "tileable matrix → AVX2".into(),
            };
        }
        if features.has_neon {
            let tiling = TileConfig::for_backend(MatmulBackend::Neon, shape);
            return DispatchDecision {
                backend: MatmulBackend::Neon,
                tiling: Some(tiling),
                precision,
                rationale: "tileable matrix → NEON".into(),
            };
        }
    }

    // Scalar fallback
    DispatchDecision {
        backend: MatmulBackend::Scalar,
        tiling: None,
        precision: MatmulPrecision::Full,
        rationale: "small matrix or no SIMD → scalar".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_fallback() {
        let shape = MatmulShape::new(4, 4, 4);
        let features = HardwareFeatures::default();
        let decision = select_backend(&shape, &features, MatmulPrecision::Full);
        assert_eq!(decision.backend, MatmulBackend::Scalar);
    }

    #[test]
    fn test_avx2_selection() {
        let shape = MatmulShape::new(64, 64, 64);
        let features = HardwareFeatures { has_avx2: true, ..Default::default() };
        let decision = select_backend(&shape, &features, MatmulPrecision::Full);
        assert_eq!(decision.backend, MatmulBackend::Avx2);
    }

    #[test]
    fn test_avx512_preferred() {
        let shape = MatmulShape::new(64, 64, 64);
        let features = HardwareFeatures { has_avx2: true, has_avx512: true, ..Default::default() };
        let decision = select_backend(&shape, &features, MatmulPrecision::Full);
        assert_eq!(decision.backend, MatmulBackend::Avx512);
    }

    #[test]
    fn test_cuda_for_large() {
        let shape = MatmulShape::new(1024, 1024, 1024);
        let features = HardwareFeatures { has_cuda: true, has_avx2: true, ..Default::default() };
        let decision = select_backend(&shape, &features, MatmulPrecision::Mixed);
        assert_eq!(decision.backend, MatmulBackend::Cuda);
    }

    #[test]
    fn test_flops() {
        let shape = MatmulShape::new(10, 20, 30);
        assert_eq!(shape.flops(), 2 * 10 * 20 * 30);
    }

    #[test]
    fn test_shape_classification() {
        assert!(MatmulShape::new(2, 2, 2).is_small());
        assert!(!MatmulShape::new(2, 2, 2).is_large());
        assert!(MatmulShape::new(1000, 1000, 1000).is_large());
    }

    #[test]
    fn test_tiling_benefit() {
        assert!(MatmulShape::new(32, 32, 32).benefits_from_tiling());
        assert!(!MatmulShape::new(8, 8, 8).benefits_from_tiling());
    }

    #[test]
    fn test_tile_config_avx2() {
        let shape = MatmulShape::new(64, 64, 64);
        let cfg = TileConfig::for_backend(MatmulBackend::Avx2, &shape);
        assert_eq!(cfg.tile_m, 8);
    }

    #[test]
    fn test_num_tiles() {
        let shape = MatmulShape::new(32, 32, 32);
        let cfg = TileConfig { tile_m: 8, tile_n: 8, tile_k: 16 };
        assert_eq!(cfg.num_tiles(&shape), 4 * 4 * 2);
    }

    #[test]
    fn test_backend_properties() {
        assert!(MatmulBackend::Avx2.is_simd());
        assert!(!MatmulBackend::Avx2.is_gpu());
        assert!(MatmulBackend::Cuda.is_gpu());
        assert!(!MatmulBackend::Scalar.is_simd());
    }

    #[test]
    fn test_neon_fallback() {
        let shape = MatmulShape::new(64, 64, 64);
        let features = HardwareFeatures { has_neon: true, ..Default::default() };
        let decision = select_backend(&shape, &features, MatmulPrecision::Half);
        assert_eq!(decision.backend, MatmulBackend::Neon);
    }

    #[test]
    fn test_opencl_for_large() {
        let shape = MatmulShape::new(1024, 1024, 1024);
        let features = HardwareFeatures { has_opencl: true, ..Default::default() };
        let decision = select_backend(&shape, &features, MatmulPrecision::Full);
        assert_eq!(decision.backend, MatmulBackend::OpenCl);
    }
}
