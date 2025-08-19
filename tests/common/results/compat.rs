// Compatibility module for bridging old TestResult (type alias) with new trait expectations
// 
// The old integration tests use TestResult<T> which is an alias for Result<T, TestError>.
// The new harness expects explicit Result<(), TestError> and Result<TestMetrics, TestError>.
// This module is intentionally empty since the types already match - no conversion needed.
//
// The bridge is handled in the harness/legacy.rs module instead.