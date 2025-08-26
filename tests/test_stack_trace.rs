use bitnet_tests::common::errors::TestError;
use bitnet_tests::common::results::TestResult;
use std::time::Duration;

#[test]
fn test_stack_trace_is_recorded_and_reported() {
    let error = TestError::AssertionError { message: "boom".into() };
    let result = TestResult::failed("failing_test", error, Duration::from_millis(1));

    let trace = result.stack_trace.clone().expect("stack trace captured");
    assert!(trace.contains("test_stack_trace_is_recorded_and_reported"));
}
