//! Test reporting system with multiple format support
//!
//! This module provides a comprehensive test reporting system that supports
//! multiple output formats including HTML, JSON, JUnit XML, and Markdown.

// pub mod comparison_analysis;
// pub mod comparison_html;
// pub mod dashboard;
pub mod formats;
// pub mod performance_viz;
pub mod reporter;
// pub mod templates;

#[allow(unused_imports)]
pub use reporter::{ReportingManager, TestReporter};

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReportFormat {
    Html,
    Json,
    Junit,
    Markdown,
}

impl std::fmt::Display for ReportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ReportFormat::Html => "HTML",
            ReportFormat::Json => "JSON",
            ReportFormat::Junit => "JUnit",
            ReportFormat::Markdown => "Markdown",
        };
        write!(f, "{}", s)
    }
}

/// Report generation errors
#[derive(Debug)]
pub enum ReportError {
    IoError(std::io::Error),
    TemplateError(String),
    SerializationError(serde_json::Error),
    XmlError(String),
    GenerationError(String),
}

impl std::fmt::Display for ReportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReportError::IoError(e) => write!(f, "IO error: {}", e),
            ReportError::TemplateError(msg) => write!(f, "Template error: {}", msg),
            ReportError::SerializationError(e) => write!(f, "Serialization error: {}", e),
            ReportError::XmlError(msg) => write!(f, "XML generation error: {}", msg),
            ReportError::GenerationError(msg) => write!(f, "Report generation failed: {}", msg),
        }
    }
}

impl std::error::Error for ReportError {}

impl From<std::io::Error> for ReportError {
    fn from(error: std::io::Error) -> Self {
        ReportError::IoError(error)
    }
}

impl From<serde_json::Error> for ReportError {
    fn from(error: serde_json::Error) -> Self {
        ReportError::SerializationError(error)
    }
}

/// Result of report generation
#[derive(Debug)]
pub struct ReportResult {
    pub format: ReportFormat,
    pub output_path: PathBuf,
    pub size_bytes: u64,
    pub generation_time: std::time::Duration,
}
