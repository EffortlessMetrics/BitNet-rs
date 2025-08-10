pub mod comparison;
pub mod cpp_implementation;
pub mod implementation;
pub mod rust_implementation;
pub mod test_cases;
pub mod test_implementation;
pub mod test_runner;

// Re-export commonly used types
pub use implementation::{
    BitNetImplementation, ImplementationCapabilities, ImplementationFactory,
    ImplementationRegistry, InferenceConfig, InferenceResult, ModelFormat, ModelInfo,
    PerformanceMetrics, ResourceInfo, ResourceLimits, ResourceManager, ResourceSummary,
};

pub use comparison::{
    AccuracyResult, ComparisonSummary, ComparisonTestCase, ComparisonTolerance,
    CrossValidationResult, CrossValidationSuite, PerformanceComparison, SingleComparisonResult,
    TokenMismatch,
};
pub use cpp_implementation::{CppImplementation, CppImplementationFactory};
pub use rust_implementation::{RustImplementation, RustImplementationFactory};
pub use test_cases::{test_suites, ComparisonTestCaseRegistry, TestCaseCategory};
pub use test_runner::{ComparisonTestRunner, CompleteValidationResult, TestSummaryStatistics};
