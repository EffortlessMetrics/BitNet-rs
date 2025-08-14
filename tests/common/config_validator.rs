use crate::config::{load_test_config, validate_config, TestConfig};
use crate::errors::TestError;
use std::path::PathBuf;

/// Configuration validation utility
pub struct ConfigValidator {
    config: TestConfig,
}

impl ConfigValidator {
    /// Create a new validator with the current configuration
    pub fn new() -> TestResult<Self> {
        let config = load_test_config()?;
        Ok(Self { config })
    }

    /// Create a validator with a specific configuration file
    pub fn from_file(path: &PathBuf) -> TestResult<Self> {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            TestError::config(format!("Failed to read config file {:?}: {}", path, e))
        })?;

        let config: TestConfig = toml::from_str(&contents).map_err(|e| {
            TestError::config(format!("Failed to parse config file {:?}: {}", path, e))
        })?;

        Ok(Self { config })
    }

    /// Validate the configuration and return detailed results
    pub fn validate(&self) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Basic validation
        if let Err(e) = validate_config(&self.config) {
            result.add_error(ValidationError::new("config", e.to_string()));
            return result;
        }

        // Additional detailed validations
        self.validate_parallel_settings(&mut result);
        self.validate_timeout_settings(&mut result);
        self.validate_cache_settings(&mut result);
        self.validate_fixture_settings(&mut result);
        self.validate_crossval_settings(&mut result);
        self.validate_reporting_settings(&mut result);

        result
    }

    fn validate_parallel_settings(&self, result: &mut ValidationResult) {
        let cores = num_cpus::get();

        if self.config.max_parallel_tests > cores * 2 {
            result.add_warning(ValidationWarning::new(
                "max_parallel_tests",
                format!(
                    "Value {} is much higher than CPU cores ({}). This may cause resource contention.",
                    self.config.max_parallel_tests, cores
                ),
            ));
        }

        if self.config.max_parallel_tests == 1 {
            result.add_info(ValidationInfo::new(
                "max_parallel_tests",
                "Running tests sequentially. Consider increasing for better performance."
                    .to_string(),
            ));
        }
    }

    fn validate_timeout_settings(&self, result: &mut ValidationResult) {
        let timeout_secs = self.config.test_timeout.as_secs();

        if timeout_secs < 30 {
            result.add_warning(ValidationWarning::new(
                "test_timeout",
                "Very short timeout may cause tests to fail prematurely.".to_string(),
            ));
        }

        if timeout_secs > 1800 {
            result.add_warning(ValidationWarning::new(
                "test_timeout",
                "Very long timeout may hide performance issues.".to_string(),
            ));
        }
    }

    fn validate_cache_settings(&self, result: &mut ValidationResult) {
        // Check if cache directory is writable
        if let Err(e) = std::fs::create_dir_all(&self.config.cache_dir) {
            result.add_error(ValidationError::new(
                "cache_dir",
                format!("Cannot create cache directory: {}", e),
            ));
        }

        // Check available disk space
        if let Ok(metadata) = std::fs::metadata(&self.config.cache_dir) {
            if metadata.is_dir() {
                // Try to get available space (platform-specific)
                if let Some(available_space) = get_available_disk_space(&self.config.cache_dir) {
                    if self.config.fixtures.max_cache_size > available_space {
                        result.add_warning(ValidationWarning::new(
                            "fixtures.max_cache_size",
                            format!(
                                "Cache size limit ({} bytes) exceeds available disk space ({} bytes)",
                                self.config.fixtures.max_cache_size, available_space
                            ),
                        ));
                    }
                }
            }
        }
    }

    fn validate_fixture_settings(&self, result: &mut ValidationResult) {
        if self.config.fixtures.auto_download {
            // Check internet connectivity for auto-download
            if !check_internet_connectivity() {
                result.add_warning(ValidationWarning::new(
                    "fixtures.auto_download",
                    "Auto-download is enabled but internet connectivity appears limited."
                        .to_string(),
                ));
            }
        }

        // Validate custom fixtures
        for fixture in &self.config.fixtures.custom_fixtures {
            if fixture.checksum.len() < 32 {
                result.add_warning(ValidationWarning::new(
                    "fixtures.custom_fixtures",
                    format!(
                        "Checksum for '{}' seems too short for security",
                        fixture.name
                    ),
                ));
            }

            // Check if URL is accessible (basic check)
            if fixture.url.starts_with("http://") {
                result.add_warning(ValidationWarning::new(
                    "fixtures.custom_fixtures",
                    format!("Fixture '{}' uses insecure HTTP URL", fixture.name),
                ));
            }
        }
    }

    fn validate_crossval_settings(&self, result: &mut ValidationResult) {
        if self.config.crossval.enabled {
            if let Some(ref cpp_path) = self.config.crossval.cpp_binary_path {
                if !cpp_path.exists() {
                    result.add_error(ValidationError::new(
                        "crossval.cpp_binary_path",
                        format!("C++ binary not found at {:?}", cpp_path),
                    ));
                } else if !is_executable(cpp_path) {
                    result.add_error(ValidationError::new(
                        "crossval.cpp_binary_path",
                        format!("C++ binary at {:?} is not executable", cpp_path),
                    ));
                }
            } else {
                result.add_error(ValidationError::new(
                    "crossval.cpp_binary_path",
                    "Cross-validation is enabled but no C++ binary path specified".to_string(),
                ));
            }

            if self.config.crossval.tolerance.min_token_accuracy > 0.99999 {
                result.add_info(ValidationInfo::new(
                    "crossval.tolerance.min_token_accuracy",
                    "Very strict token accuracy tolerance. Consider relaxing if tests fail frequently.".to_string(),
                ));
            }
        }
    }

    fn validate_reporting_settings(&self, result: &mut ValidationResult) {
        // Check if output directory is writable
        if let Err(e) = std::fs::create_dir_all(&self.config.reporting.output_dir) {
            result.add_error(ValidationError::new(
                "reporting.output_dir",
                format!("Cannot create report output directory: {}", e),
            ));
        }

        if self.config.reporting.formats.len() > 3 {
            result.add_info(ValidationInfo::new(
                "reporting.formats",
                "Many report formats selected. This may slow down test execution.".to_string(),
            ));
        }

        if self.config.reporting.generate_coverage && !coverage_tool_available() {
            result.add_warning(ValidationWarning::new(
                "reporting.generate_coverage",
                "Coverage generation enabled but cargo-tarpaulin not found.".to_string(),
            ));
        }
    }

    /// Get the validated configuration
    pub fn config(&self) -> &TestConfig {
        &self.config
    }
}

/// Result of configuration validation
#[derive(Debug, Default)]
pub struct ValidationResult {
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub info: Vec<ValidationInfo>,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }

    pub fn add_info(&mut self, info: ValidationInfo) {
        self.info.push(info);
    }

    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    pub fn summary(&self) -> String {
        format!(
            "Validation complete: {} errors, {} warnings, {} info messages",
            self.errors.len(),
            self.warnings.len(),
            self.info.len()
        )
    }
}

#[derive(Debug)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
}

impl ValidationError {
    pub fn new(field: &str, message: String) -> Self {
        Self {
            field: field.to_string(),
            message,
        }
    }
}

#[derive(Debug)]
pub struct ValidationWarning {
    pub field: String,
    pub message: String,
}

impl ValidationWarning {
    pub fn new(field: &str, message: String) -> Self {
        Self {
            field: field.to_string(),
            message,
        }
    }
}

#[derive(Debug)]
pub struct ValidationInfo {
    pub field: String,
    pub message: String,
}

impl ValidationInfo {
    pub fn new(field: &str, message: String) -> Self {
        Self {
            field: field.to_string(),
            message,
        }
    }
}

// Helper functions

fn get_available_disk_space(_path: &PathBuf) -> Option<u64> {
    // Platform-specific disk space checking
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::mem;

        let path_cstr = CString::new(_path.to_string_lossy().as_bytes()).ok()?;
        let mut statvfs: libc::statvfs = unsafe { mem::zeroed() };

        if unsafe { libc::statvfs(path_cstr.as_ptr(), &mut statvfs) } == 0 {
            Some(statvfs.f_bavail * statvfs.f_frsize)
        } else {
            None
        }
    }

    #[cfg(windows)]
    {
        // Simplified Windows implementation - disk space checking can be added later
        // For now, just return None to skip the validation
        None
    }

    #[cfg(not(any(unix, windows)))]
    {
        None
    }
}

fn check_internet_connectivity() -> bool {
    // Simple connectivity check
    std::process::Command::new("ping")
        .arg("-c")
        .arg("1")
        .arg("8.8.8.8")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn is_executable(path: &PathBuf) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = std::fs::metadata(path) {
            let permissions = metadata.permissions();
            permissions.mode() & 0o111 != 0
        } else {
            false
        }
    }

    #[cfg(windows)]
    {
        // On Windows, check if it's a .exe file or has executable extension
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| matches!(ext.to_lowercase().as_str(), "exe" | "bat" | "cmd"))
            .unwrap_or(false)
    }

    #[cfg(not(any(unix, windows)))]
    {
        true // Assume executable on other platforms
    }
}

fn coverage_tool_available() -> bool {
    std::process::Command::new("cargo")
        .arg("tarpaulin")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_validator_creation() {
        // This test might fail if no config is available, which is expected
        match ConfigValidator::new() {
            Ok(_) => {}  // Config loaded successfully
            Err(_) => {} // No config available, which is fine for testing
        }
    }

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::new();
        assert!(result.is_valid());
        assert!(!result.has_warnings());

        result.add_error(ValidationError::new("test", "Test error".to_string()));
        assert!(!result.is_valid());

        result.add_warning(ValidationWarning::new("test", "Test warning".to_string()));
        assert!(result.has_warnings());

        let summary = result.summary();
        assert!(summary.contains("1 errors"));
        assert!(summary.contains("1 warnings"));
    }

    #[test]
    fn test_disk_space_check() {
        let temp_dir = TempDir::new().unwrap();
        let space = get_available_disk_space(&temp_dir.path().to_path_buf());
        // Should return Some value on Unix platforms, None is acceptable on Windows for now
        #[cfg(unix)]
        assert!(space.is_some());
        #[cfg(not(unix))]
        let _ = space; // Just ensure it doesn't panic
    }
}
