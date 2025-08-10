//! Test reporting system with multiple format support
//!
//! This module provides a comprehensive test reporting system that supports
//! multiple output formats including HTML, JSON, JUnit XML, and Markdown.

pub mod formats;
pub mod reporter;
pub mod templates;

pub use formats::{HtmlReporter, JsonReporter, JunitReporter, MarkdownReporter};
pub use reporter::{ReportingManager, TestReporter};

use crate::common::results::{TestResult, TestSuiteResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

/// Configuration for test reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    pub output_dir: PathBuf,
    pub formats: Vec<ReportFormat>,
    pub include_artifacts: bool,
    pub generate_coverage: bool,
    pub interactive_html: bool,
}

/// Supported report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Html,
    Json,
    Junit,
    Markdown,
}

/// Report generation errors
#[derive(Debug, Error)]
pub enum ReportError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Template error: {0}")]
    TemplateError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("XML generation error: {0}")]
    XmlError(String),

    #[error("Report generation failed: {0}")]
    GenerationError(String),
}

/// Result of report generation
#[derive(Debug)]
pub struct ReportResult {
    pub format: ReportFormat,
    pub output_path: PathBuf,
    pub size_bytes: u64,
    pub generation_time: std::time::Duration,
}
