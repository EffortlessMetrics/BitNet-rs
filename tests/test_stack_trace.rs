use bitnet_tests::TestError;
use bitnet_tests::reporting::{TestReporter, formats::MarkdownReporter};
use bitnet_tests::results::{TestResult, TestSuiteResult};
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;

#[tokio::test]
async fn test_stack_trace_is_recorded_and_reported() {
    let error = TestError::AssertionError { message: "boom".into() };
    let result = TestResult::failed("failing_test", error, Duration::from_millis(1));
    let trace = result.stack_trace.clone().expect("stack trace captured");
    assert!(!trace.is_empty());
    assert!(trace.contains("test_stack_trace_is_recorded_and_reported"));

    let suite = TestSuiteResult::new("suite", vec![result.clone()], result.duration);
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("report.md");
    MarkdownReporter::new().generate_report(&[suite], &path).await.unwrap();
    let content = fs::read_to_string(&path).await.unwrap();
    assert!(content.contains("Stack trace"));
    assert!(content.contains("test_stack_trace_is_recorded_and_reported"));
}
