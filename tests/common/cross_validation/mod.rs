pub mod cpp_implementation;
pub mod implementation;
pub mod rust_implementation;
pub mod test_implementation;

// Re-export commonly used types
pub use implementation::{
    BitNetImplementation, ImplementationCapabilities, ImplementationFactory,
    ImplementationRegistry, InferenceConfig, InferenceResult, ModelFormat, ModelInfo,
    PerformanceMetrics, ResourceInfo, ResourceLimits, ResourceManager, ResourceSummary,
};

pub use cpp_implementation::{CppImplementation, CppImplementationFactory};
pub use rust_implementation::{RustImplementation, RustImplementationFactory};
