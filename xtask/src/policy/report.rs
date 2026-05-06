//! Shared types for policy report serialization.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
pub enum ReportSeverity {
    Ok,
    Warn,
    Error,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Report {
    pub report: &'static str,
    pub schema_version: u32,
    pub total_tracked: usize,
    pub in_scope: usize,
    pub matched: usize,
    pub uncovered: Vec<String>,
    pub unused_entries: Vec<String>,
    pub expired_entries: Vec<String>,
    pub schema_errors: Vec<String>,
}
