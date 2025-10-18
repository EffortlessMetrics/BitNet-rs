pub mod comparison;
mod cpp_ffi;
pub mod cpp_implementation;
pub mod implementation;
pub mod rust_implementation;
pub mod test_cases;
pub mod test_implementation;
pub mod test_runner;

// Re-export commonly used types
#[allow(unused_imports)]
pub use implementation::{
    BitNetImplementation, ImplementationCapabilities, ImplementationFactory,
    ImplementationRegistry, InferenceConfig, InferenceResult, ModelFormat, ModelInfo,
    PerformanceMetrics, ResourceInfo, ResourceLimits, ResourceManager, ResourceSummary,
};

#[allow(unused_imports)]
pub use comparison::{
    AccuracyResult, ComparisonSummary, ComparisonTestCase, ComparisonTolerance,
    CrossValidationResult, CrossValidationSuite, PerformanceComparison, SingleComparisonResult,
    TokenMismatch,
};
#[cfg(feature = "cpp-ffi")]
#[allow(unused_imports)]
pub use cpp_implementation::{CppImplementation, CppImplementationFactory};
#[allow(unused_imports)]
pub use rust_implementation::{RustImplementation, RustImplementationFactory};
#[allow(unused_imports)]
pub use test_cases::{ComparisonTestCaseRegistry, TestCaseCategory, test_suites};
#[allow(unused_imports)]
pub use test_runner::{ComparisonTestRunner, CompleteValidationResult, TestSummaryStatistics};
