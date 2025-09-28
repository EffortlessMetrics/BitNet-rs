//! Compute receipts and validation for BitNet.rs tests
//!
//! Provides structured receipts for CI validation and strict mode enforcement.

use serde::Serialize;

/// Structured compute receipt for CI validation
#[derive(Debug, Serialize)]
pub struct ComputeReceipt<'a> {
    pub compute_path: &'a str, // "real" | "mock_ci"
    pub backend: &'a str,      // "cpu" | "cuda"
    pub kernels: Vec<&'a str>,
    pub deterministic: bool,
    pub precision_mode: Option<&'a str>,
    pub correlation: Option<f32>,
    pub rel_error: Option<f32>,
}

impl<'a> ComputeReceipt<'a> {
    /// Create a new compute receipt for real computation
    #[allow(dead_code)]
    pub fn real(backend: &'a str, kernels: Vec<&'a str>) -> Self {
        Self {
            compute_path: "real",
            backend,
            kernels,
            deterministic: true,
            precision_mode: None,
            correlation: None,
            rel_error: None,
        }
    }

    /// Set precision mode
    #[allow(dead_code)]
    pub fn with_precision(mut self, precision: &'a str) -> Self {
        self.precision_mode = Some(precision);
        self
    }

    /// Set accuracy metrics
    #[allow(dead_code)]
    pub fn with_accuracy(mut self, correlation: f32, rel_error: f32) -> Self {
        self.correlation = Some(correlation);
        self.rel_error = Some(rel_error);
        self
    }

    /// Print receipt as JSON for CI consumption
    #[allow(dead_code)]
    pub fn print(&self) {
        if let Ok(json) = serde_json::to_string(self) {
            println!("{}", json);
        }
    }
}

/// Assert real computation in strict mode
///
/// In strict mode, tests must use real computation paths.
/// Mock or simulated computation should cause test failure.
#[allow(dead_code)]
pub fn assert_real_compute_strict(receipt: &ComputeReceipt<'_>) {
    if std::env::var("BITNET_STRICT_MODE").unwrap_or_default() == "1" {
        assert_eq!(receipt.compute_path, "real", "Mock compute in strict lane");
        assert_ne!(receipt.backend, "mock", "Mock backend in strict lane");
    }
}
