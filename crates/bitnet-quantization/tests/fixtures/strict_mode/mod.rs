//! Strict Mode Test Fixtures Module
//!
//! Provides test data for mock detection and strict mode validation
//! scenarios for Issue #260 mock elimination.

pub mod mock_detection_data;

// Re-export key types for convenience
#[allow(unused_imports)]
pub use mock_detection_data::{
    ComputationData, DetectionMethod, MockDetectionFixture, MockIndicator,
    StatisticalAnalysisFixture, StatisticalTest, StrictModeBehavior, StrictModeFixture,
    TestInterpretation, ValidationCriteria,
};
