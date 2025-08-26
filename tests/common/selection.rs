use super::config::TestConfig;
use super::errors::TestError;
#[cfg(feature = "fixtures")]
use super::fast_config::fast_config;
use super::parallel::{TestCategory, TestInfo, TestPriority};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use tokio::process::Command;
use tracing::{debug, info, warn};

#[cfg(feature = "fixtures")]
fn build_config() -> TestConfig {
    fast_config()
}

#[cfg(not(feature = "fixtures"))]
fn build_config() -> TestConfig {
    TestConfig::default()
}

/// Test selector that discovers and prioritizes tests for execution
pub struct TestSelector {
    config: TestConfig,
    test_cache: HashMap<String, Vec<TestInfo>>,
    priority_rules: PriorityRules,
}

impl TestSelector {
    pub fn new(config: TestConfig) -> Self {
        Self { config, test_cache: HashMap::new(), priority_rules: PriorityRules::default() }
    }

    /// Discover all available tests in the workspace
    pub async fn discover_tests(&mut self) -> Result<Vec<TestInfo>, TestError> {
        info!("Discovering tests in workspace...");

        let mut all_tests = Vec::new();

        // Get list of workspace members
        let workspace_members = self.get_workspace_members().await?;

        for member in workspace_members {
            let tests = self.discover_crate_tests(&member).await?;
            all_tests.extend(tests);
        }

        // Apply priority rules
        for test in &mut all_tests {
            test.priority = self.priority_rules.determine_priority(test);
        }

        // Cache results
        let crate_groups: HashMap<String, Vec<TestInfo>> =
            all_tests.iter().cloned().fold(HashMap::new(), |mut acc, test| {
                acc.entry(test.crate_name.clone()).or_default().push(test);
                acc
            });
        self.test_cache = crate_groups;

        info!("Discovered {} tests across {} crates", all_tests.len(), self.test_cache.len());

        Ok(all_tests)
    }

    /// Get tests for specific crates only
    pub async fn get_tests_for_crates(
        &mut self,
        crate_names: &[String],
    ) -> Result<Vec<TestInfo>, TestError> {
        let mut selected_tests = Vec::new();

        for crate_name in crate_names {
            if let Some(cached_tests) = self.test_cache.get(crate_name) {
                selected_tests.extend(cached_tests.clone());
            } else {
                // Discover tests for this crate if not cached
                let tests = self.discover_crate_tests(crate_name).await?;
                selected_tests.extend(tests);
            }
        }

        Ok(selected_tests)
    }

    /// Select tests based on changed files (incremental testing)
    pub async fn select_tests_for_changes(
        &mut self,
        changed_files: &[PathBuf],
    ) -> Result<Vec<TestInfo>, TestError> {
        let all_tests = self.discover_tests().await?;
        let mut selected_tests = Vec::new();

        // Determine affected crates
        let affected_crates = self.determine_affected_crates(changed_files);

        for test in all_tests {
            if self.is_test_affected(&test, changed_files, &affected_crates) {
                selected_tests.push(test);
            }
        }

        info!(
            "Selected {} tests affected by {} changed files",
            selected_tests.len(),
            changed_files.len()
        );

        Ok(selected_tests)
    }

    /// Select fast tests only (for quick feedback)
    pub async fn select_fast_tests(&mut self) -> Result<Vec<TestInfo>, TestError> {
        let all_tests = self.discover_tests().await?;

        let fast_tests = all_tests
            .into_iter()
            .filter(|test| {
                matches!(test.category, TestCategory::Unit)
                    && matches!(test.priority, TestPriority::Critical | TestPriority::High)
            })
            .collect();

        Ok(fast_tests)
    }

    /// Select tests by category
    pub async fn select_tests_by_category(
        &mut self,
        categories: &[TestCategory],
    ) -> Result<Vec<TestInfo>, TestError> {
        let all_tests = self.discover_tests().await?;

        let selected_tests =
            all_tests.into_iter().filter(|test| categories.contains(&test.category)).collect();

        Ok(selected_tests)
    }

    /// Get workspace members from Cargo.toml
    async fn get_workspace_members(&self) -> Result<Vec<String>, TestError> {
        let cargo_toml_path = PathBuf::from("Cargo.toml");

        if !cargo_toml_path.exists() {
            return Err(TestError::SetupError { message: "Cargo.toml not found".to_string() });
        }

        let content =
            tokio::fs::read_to_string(&cargo_toml_path).await.map_err(TestError::IoError)?;

        let cargo_toml: toml::Value = content.parse().map_err(|e| TestError::SetupError {
            message: format!("Failed to parse Cargo.toml: {}", e),
        })?;

        let members = cargo_toml
            .get("workspace")
            .and_then(|w| w.get("members"))
            .and_then(|m| m.as_array())
            .ok_or_else(|| TestError::SetupError {
                message: "No workspace members found".to_string(),
            })?;

        let member_names = members.iter()
            .filter_map(|m| m.as_str())
            .filter(|name| !name.starts_with('.') && *name != "xtask") // Skip hidden and utility crates
            .map(|name| {
                // Extract crate name from path
                if name.contains('/') {
                    name.split('/').next_back().unwrap_or(name).to_string()
                } else {
                    name.to_string()
                }
            })
            .collect();

        Ok(member_names)
    }

    /// Discover tests for a specific crate
    async fn discover_crate_tests(&self, crate_name: &str) -> Result<Vec<TestInfo>, TestError> {
        debug!("Discovering tests for crate: {}", crate_name);

        // Use cargo test --list to discover tests
        let mut cmd = Command::new("cargo");
        cmd.arg("test").arg("--package").arg(crate_name).arg("--list").arg("--message-format=json");

        let output = cmd.output().await.map_err(|e| TestError::ExecutionError {
            message: format!("Failed to list tests: {}", e),
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!("Failed to list tests for {}: {}", crate_name, stderr);
            return Ok(Vec::new()); // Return empty list instead of failing
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut tests = Vec::new();

        // Parse test list output
        for line in stdout.lines() {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(test_name) = json.get("name").and_then(|n| n.as_str()) {
                    if test_name.ends_with(": test") {
                        let clean_name = test_name.trim_end_matches(": test");
                        let test_info = TestInfo {
                            name: clean_name.to_string(),
                            crate_name: crate_name.to_string(),
                            file_path: self.infer_test_file_path(crate_name, clean_name),
                            category: self.categorize_test(clean_name, crate_name),
                            priority: TestPriority::Medium, // Will be updated later
                        };
                        tests.push(test_info);
                    }
                }
            } else {
                // Fallback: parse simple text output
                if line.contains(": test") {
                    let test_name = line.split(':').next().unwrap_or(line).trim();
                    let test_info = TestInfo {
                        name: test_name.to_string(),
                        crate_name: crate_name.to_string(),
                        file_path: self.infer_test_file_path(crate_name, test_name),
                        category: self.categorize_test(test_name, crate_name),
                        priority: TestPriority::Medium,
                    };
                    tests.push(test_info);
                }
            }
        }

        debug!("Found {} tests in crate {}", tests.len(), crate_name);
        Ok(tests)
    }

    /// Infer test file path from test name and crate
    fn infer_test_file_path(&self, crate_name: &str, test_name: &str) -> PathBuf {
        // Try to infer the file path based on naming conventions
        if crate_name == "tests" {
            // Integration tests
            PathBuf::from(format!("tests/{}.rs", test_name))
        } else if test_name.contains("integration") {
            PathBuf::from(format!("crates/{}/tests/integration.rs", crate_name))
        } else {
            // Unit tests (likely in lib.rs or specific module)
            PathBuf::from(format!("crates/{}/src/lib.rs", crate_name))
        }
    }

    /// Categorize test based on name and crate
    fn categorize_test(&self, test_name: &str, crate_name: &str) -> TestCategory {
        let name_lower = test_name.to_lowercase();

        if name_lower.contains("performance") || name_lower.contains("benchmark") {
            TestCategory::Performance
        } else if name_lower.contains("integration") || crate_name == "tests" {
            TestCategory::Integration
        } else if name_lower.contains("crossval") || name_lower.contains("cross_validation") {
            TestCategory::CrossValidation
        } else {
            TestCategory::Unit
        }
    }

    /// Determine which crates are affected by file changes
    fn determine_affected_crates(&self, changed_files: &[PathBuf]) -> HashSet<String> {
        let mut affected_crates = HashSet::new();

        for file in changed_files {
            let file_str = file.to_string_lossy();

            if file_str.starts_with("crates/") {
                // Extract crate name from path like "crates/bitnet-common/src/lib.rs"
                if let Some(crate_name) = file_str.split('/').nth(1) {
                    affected_crates.insert(crate_name.to_string());
                }
            } else if file_str.starts_with("tests/") {
                // Tests directory affects the tests crate
                affected_crates.insert("tests".to_string());
            } else if file_str == "Cargo.toml" || file_str == "Cargo.lock" {
                // Root changes affect all crates
                affected_crates.insert("*".to_string());
            }
        }

        affected_crates
    }

    /// Check if a test is affected by file changes
    fn is_test_affected(
        &self,
        test: &TestInfo,
        changed_files: &[PathBuf],
        affected_crates: &HashSet<String>,
    ) -> bool {
        // If all crates are affected
        if affected_crates.contains("*") {
            return true;
        }

        // If test's crate is directly affected
        if affected_crates.contains(&test.crate_name) {
            return true;
        }

        // Check for dependency relationships (simplified)
        for file in changed_files {
            let file_str = file.to_string_lossy();

            // If a core crate changed, it might affect other crates
            if file_str.contains("bitnet-common") || file_str.contains("bitnet-models") {
                return true;
            }

            // If test file path matches changed file
            if file_str.contains(test.file_path.to_string_lossy().as_ref()) {
                return true;
            }
        }

        false
    }
}

/// Rules for determining test priorities
#[derive(Debug, Default)]
pub struct PriorityRules {
    critical_patterns: Vec<String>,
    high_patterns: Vec<String>,
    low_patterns: Vec<String>,
}

impl PriorityRules {
    pub fn new() -> Self {
        Self {
            critical_patterns: vec![
                "core".to_string(),
                "basic".to_string(),
                "essential".to_string(),
                "smoke".to_string(),
            ],
            high_patterns: vec!["api".to_string(), "public".to_string(), "interface".to_string()],
            low_patterns: vec![
                "benchmark".to_string(),
                "performance".to_string(),
                "stress".to_string(),
                "fuzz".to_string(),
            ],
        }
    }

    pub fn determine_priority(&self, test: &TestInfo) -> TestPriority {
        let name_lower = test.name.to_lowercase();

        // Check critical patterns
        for pattern in &self.critical_patterns {
            if name_lower.contains(pattern) {
                return TestPriority::Critical;
            }
        }

        // Check low priority patterns
        for pattern in &self.low_patterns {
            if name_lower.contains(pattern) {
                return TestPriority::Low;
            }
        }

        // Check high priority patterns
        for pattern in &self.high_patterns {
            if name_lower.contains(pattern) {
                return TestPriority::High;
            }
        }

        // Default priority based on category
        match test.category {
            TestCategory::Unit => TestPriority::High,
            TestCategory::Integration => TestPriority::Medium,
            TestCategory::Performance => TestPriority::Low,
            TestCategory::CrossValidation => TestPriority::Low,
        }
    }
}

/// Test selection strategy
pub enum SelectionStrategy {
    All,
    Fast,
    Incremental(Vec<PathBuf>),
    Category(Vec<TestCategory>),
    Priority(TestPriority),
    Custom(Box<dyn Fn(&TestInfo) -> bool + Send + Sync>),
}

impl SelectionStrategy {
    pub async fn apply(&self, selector: &mut TestSelector) -> Result<Vec<TestInfo>, TestError> {
        match self {
            SelectionStrategy::All => selector.discover_tests().await,
            SelectionStrategy::Fast => selector.select_fast_tests().await,
            SelectionStrategy::Incremental(changed_files) => {
                selector.select_tests_for_changes(changed_files).await
            }
            SelectionStrategy::Category(categories) => {
                selector.select_tests_by_category(categories).await
            }
            SelectionStrategy::Priority(min_priority) => {
                let all_tests = selector.discover_tests().await?;
                let filtered = all_tests
                    .into_iter()
                    .filter(|test| test.priority as u8 <= *min_priority as u8)
                    .collect();
                Ok(filtered)
            }
            SelectionStrategy::Custom(_filter) => {
                // Custom filtering would be implemented here
                selector.discover_tests().await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_test_selector_creation() {
        let config = build_config();
        let selector = TestSelector::new(config);
        assert!(selector.test_cache.is_empty());
    }

    #[test]
    fn test_priority_rules() {
        let rules = PriorityRules::new();

        let critical_test = TestInfo {
            name: "test_core_functionality".to_string(),
            crate_name: "bitnet-common".to_string(),
            file_path: PathBuf::from("src/lib.rs"),
            category: TestCategory::Unit,
            priority: TestPriority::Medium,
        };

        assert_eq!(rules.determine_priority(&critical_test), TestPriority::Critical);
    }

    #[test]
    fn test_categorize_test() {
        let config = build_config();
        let selector = TestSelector::new(config);

        assert_eq!(
            selector.categorize_test("test_performance_benchmark", "bitnet-kernels"),
            TestCategory::Performance
        );

        assert_eq!(
            selector.categorize_test("test_integration_workflow", "tests"),
            TestCategory::Integration
        );

        assert_eq!(
            selector.categorize_test("test_unit_function", "bitnet-common"),
            TestCategory::Unit
        );
    }

    #[test]
    fn test_affected_crates_detection() {
        let config = build_config();
        let selector = TestSelector::new(config);

        let changed_files = vec![
            PathBuf::from("crates/bitnet-common/src/lib.rs"),
            PathBuf::from("tests/integration_tests.rs"),
        ];

        let affected = selector.determine_affected_crates(&changed_files);
        assert!(affected.contains("bitnet-common"));
        assert!(affected.contains("tests"));
    }

    #[test]
    #[cfg(feature = "fixtures")]
    fn test_fast_config_applied_selector() {
        let config = build_config();
        assert_eq!(config.max_parallel_tests, 2);
        assert_eq!(config.log_level, "error");
    }
}
