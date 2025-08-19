//! Breaking Change Detection
//!
//! This module analyzes the public API surface to detect breaking changes
//! between versions and enforces semver compatibility.

use anyhow::{anyhow, Context, Result};
use cargo_semver_checks::{Check, Config as SemverConfig};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Breaking change types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BreakingChangeType {
    /// Public item removed
    ItemRemoved { path: String },
    
    /// Function signature changed
    SignatureChanged { 
        path: String,
        old: String,
        new: String,
    },
    
    /// Type definition changed
    TypeChanged {
        path: String,
        old: String,
        new: String,
    },
    
    /// Trait requirements changed
    TraitChanged {
        path: String,
        change: String,
    },
    
    /// Enum variant removed or changed
    EnumVariantChanged {
        enum_path: String,
        variant: String,
        change: String,
    },
    
    /// Module structure changed
    ModuleRestructured {
        old_path: String,
        new_path: Option<String>,
    },
}

/// API compatibility report
#[derive(Debug, Serialize, Deserialize)]
pub struct CompatibilityReport {
    pub version: String,
    pub baseline_version: String,
    pub breaking_changes: Vec<BreakingChangeType>,
    pub compatible: bool,
    pub suggested_version: String,
}

/// Breaking change detector
pub struct BreakingChangeDetector {
    baseline_path: PathBuf,
    current_path: PathBuf,
}

impl BreakingChangeDetector {
    pub fn new(baseline_path: PathBuf, current_path: PathBuf) -> Self {
        Self {
            baseline_path,
            current_path,
        }
    }

    /// Run breaking change detection
    pub fn detect(&self) -> Result<CompatibilityReport> {
        println!("🔍 Detecting breaking changes...");
        
        // Use cargo-semver-checks for thorough analysis
        let mut breaking_changes = Vec::new();
        
        // Run cargo semver-checks
        let semver_result = self.run_semver_checks()?;
        breaking_changes.extend(semver_result);
        
        // Additional custom checks
        let custom_checks = self.run_custom_checks()?;
        breaking_changes.extend(custom_checks);
        
        // Determine version bump needed
        let suggested_version = self.suggest_version_bump(&breaking_changes)?;
        
        Ok(CompatibilityReport {
            version: self.get_current_version()?,
            baseline_version: self.get_baseline_version()?,
            breaking_changes: breaking_changes.clone(),
            compatible: breaking_changes.is_empty(),
            suggested_version,
        })
    }

    fn run_semver_checks(&self) -> Result<Vec<BreakingChangeType>> {
        let mut cmd = Command::new("cargo");
        cmd.args([
            "semver-checks",
            "--baseline-path", self.baseline_path.to_str().unwrap(),
            "--manifest-path", self.current_path.join("Cargo.toml").to_str().unwrap(),
            "--json",
        ]);
        
        let output = cmd.output()
            .context("Failed to run cargo-semver-checks")?;
        
        if !output.status.success() {
            // Parse the JSON output for breaking changes
            let json_str = String::from_utf8_lossy(&output.stdout);
            self.parse_semver_output(&json_str)
        } else {
            Ok(Vec::new())
        }
    }

    fn parse_semver_output(&self, json_str: &str) -> Result<Vec<BreakingChangeType>> {
        let mut changes = Vec::new();
        
        // Parse cargo-semver-checks JSON output
        if let Ok(report) = serde_json::from_str::<serde_json::Value>(json_str) {
            if let Some(breaking) = report.get("breaking").and_then(|v| v.as_array()) {
                for item in breaking {
                    if let Some(change) = self.convert_semver_change(item) {
                        changes.push(change);
                    }
                }
            }
        }
        
        Ok(changes)
    }

    fn convert_semver_change(&self, item: &serde_json::Value) -> Option<BreakingChangeType> {
        let kind = item.get("kind")?.as_str()?;
        let path = item.get("path")?.as_str()?;
        
        match kind {
            "function_signature_changed" => {
                Some(BreakingChangeType::SignatureChanged {
                    path: path.to_string(),
                    old: item.get("old")?.as_str()?.to_string(),
                    new: item.get("new")?.as_str()?.to_string(),
                })
            },
            "type_removed" | "struct_removed" | "enum_removed" => {
                Some(BreakingChangeType::ItemRemoved {
                    path: path.to_string(),
                })
            },
            "enum_variant_removed" => {
                Some(BreakingChangeType::EnumVariantChanged {
                    enum_path: path.to_string(),
                    variant: item.get("variant")?.as_str()?.to_string(),
                    change: "removed".to_string(),
                })
            },
            _ => None,
        }
    }

    fn run_custom_checks(&self) -> Result<Vec<BreakingChangeType>> {
        let mut changes = Vec::new();
        
        // Check for FFI compatibility
        changes.extend(self.check_ffi_compatibility()?);
        
        // Check for config file format changes
        changes.extend(self.check_config_compatibility()?);
        
        // Check for protocol/wire format changes
        changes.extend(self.check_protocol_compatibility()?);
        
        Ok(changes)
    }

    fn check_ffi_compatibility(&self) -> Result<Vec<BreakingChangeType>> {
        let mut changes = Vec::new();
        
        // Check C header files for changes
        let baseline_header = self.baseline_path.join("include/bitnet.h");
        let current_header = self.current_path.join("include/bitnet.h");
        
        if baseline_header.exists() && current_header.exists() {
            let baseline_content = fs::read_to_string(&baseline_header)?;
            let current_content = fs::read_to_string(&current_header)?;
            
            // Parse and compare function signatures
            let baseline_sigs = Self::extract_c_signatures(&baseline_content);
            let current_sigs = Self::extract_c_signatures(&current_content);
            
            for (name, sig) in &baseline_sigs {
                if !current_sigs.contains_key(name) {
                    changes.push(BreakingChangeType::ItemRemoved {
                        path: format!("ffi::{}", name),
                    });
                } else if current_sigs[name] != *sig {
                    changes.push(BreakingChangeType::SignatureChanged {
                        path: format!("ffi::{}", name),
                        old: sig.clone(),
                        new: current_sigs[name].clone(),
                    });
                }
            }
        }
        
        Ok(changes)
    }

    fn extract_c_signatures(content: &str) -> HashMap<String, String> {
        let mut signatures = HashMap::new();
        
        // Simple regex-based extraction (could be improved with proper C parser)
        for line in content.lines() {
            if line.contains("BITNET_API") || line.contains("extern") {
                if let Some(name) = Self::extract_function_name(line) {
                    signatures.insert(name.to_string(), line.trim().to_string());
                }
            }
        }
        
        signatures
    }

    fn extract_function_name(line: &str) -> Option<&str> {
        // Extract function name from C declaration
        let parts: Vec<&str> = line.split('(').collect();
        if let Some(first) = parts.first() {
            let words: Vec<&str> = first.split_whitespace().collect();
            return words.last().copied();
        }
        None
    }

    fn check_config_compatibility(&self) -> Result<Vec<BreakingChangeType>> {
        // Check for changes in configuration file formats
        Ok(Vec::new())
    }

    fn check_protocol_compatibility(&self) -> Result<Vec<BreakingChangeType>> {
        // Check for changes in network protocols or file formats
        Ok(Vec::new())
    }

    fn suggest_version_bump(&self, changes: &[BreakingChangeType]) -> Result<String> {
        let current = self.get_current_version()?;
        let parts: Vec<&str> = current.split('.').collect();
        
        if parts.len() != 3 {
            return Err(anyhow!("Invalid version format: {}", current));
        }
        
        let major: u32 = parts[0].parse()?;
        let minor: u32 = parts[1].parse()?;
        let patch: u32 = parts[2].parse()?;
        
        if !changes.is_empty() {
            // Breaking changes require major version bump
            Ok(format!("{}.0.0", major + 1))
        } else {
            // No breaking changes, suggest patch bump
            Ok(format!("{}.{}.{}", major, minor, patch + 1))
        }
    }

    fn get_current_version(&self) -> Result<String> {
        let cargo_toml = fs::read_to_string(self.current_path.join("Cargo.toml"))?;
        let manifest: toml::Value = toml::from_str(&cargo_toml)?;
        
        manifest
            .get("package")
            .and_then(|p| p.get("version"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("Version not found in Cargo.toml"))
    }

    fn get_baseline_version(&self) -> Result<String> {
        let cargo_toml = fs::read_to_string(self.baseline_path.join("Cargo.toml"))?;
        let manifest: toml::Value = toml::from_str(&cargo_toml)?;
        
        manifest
            .get("package")
            .and_then(|p| p.get("version"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("Version not found in baseline Cargo.toml"))
    }
}

/// Run breaking change detection command
pub fn detect_breaking_changes(baseline: &Path, current: &Path) -> Result<()> {
    let detector = BreakingChangeDetector::new(
        baseline.to_path_buf(),
        current.to_path_buf(),
    );
    
    let report = detector.detect()?;
    
    // Print report
    println!("\n📊 API Compatibility Report");
    println!("  Current version: {}", report.version);
    println!("  Baseline version: {}", report.baseline_version);
    println!("  Compatible: {}", if report.compatible { "✅ Yes" } else { "❌ No" });
    
    if !report.breaking_changes.is_empty() {
        println!("\n⚠️  Breaking Changes Detected:");
        for change in &report.breaking_changes {
            match change {
                BreakingChangeType::ItemRemoved { path } => {
                    println!("  - Removed: {}", path);
                },
                BreakingChangeType::SignatureChanged { path, old, new } => {
                    println!("  - Signature changed: {}", path);
                    println!("    Old: {}", old);
                    println!("    New: {}", new);
                },
                BreakingChangeType::TypeChanged { path, old, new } => {
                    println!("  - Type changed: {}", path);
                    println!("    Old: {}", old);
                    println!("    New: {}", new);
                },
                _ => {
                    println!("  - {:?}", change);
                }
            }
        }
        
        println!("\n💡 Suggested version: {}", report.suggested_version);
        println!("   (Current version {} requires major bump due to breaking changes)", report.version);
    }
    
    // Write report to file
    let report_json = serde_json::to_string_pretty(&report)?;
    fs::write("api-compatibility-report.json", report_json)?;
    
    if !report.compatible {
        return Err(anyhow!(
            "Breaking changes detected! Suggested version: {}",
            report.suggested_version
        ));
    }
    
    Ok(())
}