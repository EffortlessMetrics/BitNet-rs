//! Coverage reporting integration with cargo-tarpaulin
//!
//! This module provides comprehensive coverage reporting capabilities including:
//! - Integration with cargo-tarpaulin for coverage collection
//! - Line-by-line coverage analysis
//! - Coverage threshold validation
//! - Coverage trend tracking and reporting
//! - HTML coverage visualization

use crate::results::{TestResult, TestSuiteResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::fs;

/// Coverage reporting errors
#[derive(Debug, Error)]
pub enum CoverageError {
    #[error("Tarpaulin execution failed: {0}")]
    TarpaulinError(String),

    #[error("Coverage data parsing failed: {0}")]
    ParseError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Coverage threshold not met: {actual:.2}% < {threshold:.2}%")]
    ThresholdError { actual: f64, threshold: f64 },

    #[error("Trend analysis failed: {0}")]
    TrendError(String),
}

/// Coverage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageConfig {
    /// Minimum coverage threshold (percentage)
    pub threshold: f64,
    /// Output directory for coverage reports
    pub output_dir: PathBuf,
    /// Whether to generate HTML reports
    pub generate_html: bool,
    /// Whether to track coverage trends
    pub track_trends: bool,
    /// Packages to include in coverage
    pub include_packages: Vec<String>,
    /// Packages to exclude from coverage
    pub exclude_packages: Vec<String>,
    /// Additional tarpaulin arguments
    pub tarpaulin_args: Vec<String>,
}

impl Default for CoverageConfig {
    fn default() -> Self {
        Self {
            threshold: 90.0,
            output_dir: PathBuf::from("target/coverage"),
            generate_html: true,
            track_trends: true,
            include_packages: vec![
                "bitnet-common".to_string(),
                "bitnet-models".to_string(),
                "bitnet-quantization".to_string(),
                "bitnet-kernels".to_string(),
                "bitnet-inference".to_string(),
                "bitnet-tokenizers".to_string(),
            ],
            exclude_packages: vec![
                "bitnet-tests".to_string(),
                "bitnet-sys".to_string(),
            ],
            tarpaulin_args: vec![
                "--timeout".to_string(),
                "120".to_string(),
                "--exclude-files".to_string(),
                "target/*".to_string(),
            ],
        }
    }
}

/// Coverage data for a single file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCoverage {
    pub path: PathBuf,
    pub total_lines: usize,
    pub covered_lines: usize,
    pub coverage_percentage: f64,
    pub line_coverage: HashMap<usize, LineCoverage>,
    pub functions: Vec<FunctionCoverage>,
}

/// Cover
