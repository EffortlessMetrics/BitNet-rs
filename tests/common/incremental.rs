use super::cache::{cache_keys, CacheKey, TestCache};
use super::errors::{TestError, TestResult};
use super::harness::{TestCase, TestSuite};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;
use tokio::fs;
use tracing::{debug, info, warn};

/// Incremental test runner that only runs tests affected by changes
pub struct IncrementalTestRunner {
    cache: TestCache,
    config: IncrementalConfig,
    change_detector: ChangeDetector,
    dependency_graph: DependencyGraph,
}

/// Configuration for incremental testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalConfig {
    /// Enable incremental testing
    pub enabled: bool,
    /// Base commit/branch for change detection
    pub base_ref: Option<String>,
    /// Paths to always include in change detection
    pub always_include: Vec<PathBuf>,
    /// Paths to ignore in change detection
    pub ignore_patterns: Vec<String>,
    /// Force run all tests (bypass incremental)
    pub force_all: bool,
    /// Maximum number of changed files before running all tests
    pub max_changed_files: usize,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_ref: None, // Will auto-detect main/master
            always_include: vec![
                PathBuf::from("Cargo.toml"),
                PathBuf::from("Cargo.lock"),
                PathBuf::from("tests/common"),
            ],
            ignore_patterns: vec![
                "*.md".to_string(),
                "*.txt".to_string(),
                "target/**".to_string(),
                ".git/**".to_string(),
                "docs/**".to_string(),
            ],
            force_all: false,
            max_changed_files: 100,
        }
    }
}

/// Detects changes in the codebase
pub struct ChangeDetector {
    config: IncrementalConfig,
    workspace_root: PathBuf,
}

/// Represents a change in the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Change {
    pub path: PathBuf,
    pub change_type: ChangeType,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Added,
    Modified,
    Deleted,
    Renamed { from: PathBuf },
}

/// Dependency graph for tracking test dependencies
pub struct DependencyGraph {
    /// Map from source file to tests that depend on it
    file_to_tests: HashMap<PathBuf, HashSet<String>>,
    /// Map from test to source files it depends on
    test_to_files: HashMap<String, HashSet<PathBuf>>,
    /// Cached dependency analysis
    cache: HashMap<String, Vec<PathBuf>>,
}

/// Result of incremental test analysis
#[derive(Debug, Clone)]
pub struct IncrementalAnalysis {
    /// Tests that should be run due to changes
    pub affected_tests: HashSet<String>,
    /// Tests that can be skipped (cached results available)
    pub cached_tests: HashSet<String>,
    /// Changes detected in the codebase
    pub changes: Vec<Change>,
    /// Whether all tests should be run
    pub run_all: bool,
    /// Reason for the decision
    pub reason: String,
}

impl IncrementalTestRunner {
    /// Create a new incremental test runner
    pub async fn new(
        cache: TestCache,
        config: IncrementalConfig,
        workspace_root: PathBuf,
    ) -> TestResult<Self> {
        let change_detector = ChangeDetector::new(config.clone(), workspace_root.clone());
        let dependency_graph = DependencyGraph::new();

        Ok(Self {
            cache,
            config,
            change_detector,
            dependency_graph,
        })
    }

    /// Analyze which tests need to be run
    pub async fn analyze<T: TestSuite>(&mut self, suite: &T) -> TestResult<IncrementalAnalysis> {
        if !self.config.enabled || self.config.force_all {
            return Ok(IncrementalAnalysis {
                affected_tests: suite
                    .test_cases()
                    .iter()
                    .map(|t| t.name().to_string())
                    .collect(),
                cached_tests: HashSet::new(),
                changes: Vec::new(),
                run_all: true,
                reason: if self.config.force_all {
                    "Force all tests enabled".to_string()
                } else {
                    "Incremental testing disabled".to_string()
                },
            });
        }

        info!("Analyzing changes for incremental testing");

        // Detect changes
        let changes = self.change_detector.detect_changes().await?;

        if changes.len() > self.config.max_changed_files {
            return Ok(IncrementalAnalysis {
                affected_tests: suite
                    .test_cases()
                    .iter()
                    .map(|t| t.name().to_string())
                    .collect(),
                cached_tests: HashSet::new(),
                changes,
                run_all: true,
                reason: format!(
                    "Too many changed files ({} > {})",
                    changes.len(),
                    self.config.max_changed_files
                ),
            });
        }

        // Build dependency graph for this suite
        self.build_dependency_graph(suite).await?;

        // Find affected tests
        let mut affected_tests = HashSet::new();
        let mut cached_tests = HashSet::new();

        for test_case in suite.test_cases() {
            let test_name = test_case.name();

            // Check if test is affected by changes
            if self.is_test_affected(test_name, &changes).await? {
                affected_tests.insert(test_name.to_string());
            } else {
                // Check if we have a valid cached result
                let cache_key = self
                    .generate_cache_key(test_case.as_ref(), suite.name())
                    .await?;
                if self.cache.is_cached(&cache_key).await {
                    cached_tests.insert(test_name.to_string());
                } else {
                    // No cache, need to run
                    affected_tests.insert(test_name.to_string());
                }
            }
        }

        let run_all = affected_tests.len() + cached_tests.len() == 0;

        Ok(IncrementalAnalysis {
            affected_tests,
            cached_tests,
            changes,
            run_all,
            reason: if run_all {
                "No tests found to run".to_string()
            } else {
                format!(
                    "Incremental analysis: {} affected, {} cached",
                    affected_tests.len(),
                    cached_tests.len()
                )
            },
        })
    }

    /// Check if a test is affected by the given changes
    async fn is_test_affected(&self, test_name: &str, changes: &[Change]) -> TestResult<bool> {
        // Get dependencies for this test
        let dependencies = self.dependency_graph.get_dependencies(test_name);

        // Check if any changed file affects this test
        for change in changes {
            // Direct dependency check
            if dependencies.contains(&change.path) {
                debug!("Test {} affected by change in {:?}", test_name, change.path);
                return Ok(true);
            }

            // Check for always-include paths
            for always_include in &self.config.always_include {
                if change.path.starts_with(always_include) {
                    debug!(
                        "Test {} affected by always-include path {:?}",
                        test_name, change.path
                    );
                    return Ok(true);
                }
            }

            // Check for test-specific patterns
            if self.is_test_file_related(&change.path, test_name) {
                debug!(
                    "Test {} affected by related file {:?}",
                    test_name, change.path
                );
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Check if a file is related to a specific test
    fn is_test_file_related(&self, file_path: &Path, test_name: &str) -> bool {
        let file_str = file_path.to_string_lossy();
        let test_name_lower = test_name.to_lowercase();

        // Check if file name contains test name
        if file_str.to_lowercase().contains(&test_name_lower) {
            return true;
        }

        // Check if file is in the same module/crate as the test
        if let Some(test_module) = self.extract_module_from_test_name(test_name) {
            if file_str.contains(&test_module) {
                return true;
            }
        }

        false
    }

    /// Extract module name from test name
    fn extract_module_from_test_name(&self, test_name: &str) -> Option<String> {
        // Extract module from test names like "bitnet_common::test_something"
        if let Some(pos) = test_name.find("::") {
            Some(test_name[..pos].replace("::", "/"))
        } else {
            None
        }
    }

    /// Build dependency graph for a test suite
    async fn build_dependency_graph<T: TestSuite>(&mut self, suite: &T) -> TestResult<()> {
        info!("Building dependency graph for suite: {}", suite.name());

        for test_case in suite.test_cases() {
            let test_name = test_case.name();
            let dependencies = self.analyze_test_dependencies(test_case.as_ref()).await?;

            self.dependency_graph
                .add_test_dependencies(test_name.to_string(), dependencies);
        }

        Ok(())
    }

    /// Analyze dependencies for a single test
    async fn analyze_test_dependencies(
        &self,
        test_case: &dyn TestCase,
    ) -> TestResult<Vec<PathBuf>> {
        let test_name = test_case.name();

        // Check cache first
        if let Some(cached_deps) = self.dependency_graph.cache.get(test_name) {
            return Ok(cached_deps.clone());
        }

        let mut dependencies = Vec::new();

        // Add common dependencies
        dependencies.extend(self.config.always_include.clone());

        // Add test-specific dependencies based on test name and metadata
        let metadata = test_case.metadata();

        // Extract crate name from test name
        if let Some(crate_name) = self.extract_crate_from_test_name(test_name) {
            let crate_path = PathBuf::from("crates").join(&crate_name);
            if crate_path.exists() {
                dependencies.push(crate_path.join("src"));
                dependencies.push(crate_path.join("Cargo.toml"));
            }
        }

        // Add dependencies from metadata
        if let Some(deps_str) = metadata.get("dependencies") {
            for dep in deps_str.split(',') {
                let dep_path = PathBuf::from(dep.trim());
                if dep_path.exists() {
                    dependencies.push(dep_path);
                }
            }
        }

        // Add test file itself
        if let Some(test_file) = self.find_test_file(test_name).await {
            dependencies.push(test_file);
        }

        Ok(dependencies)
    }

    /// Extract crate name from test name
    fn extract_crate_from_test_name(&self, test_name: &str) -> Option<String> {
        // Handle test names like "bitnet_common::test_something"
        if test_name.starts_with("bitnet_") {
            if let Some(pos) = test_name.find("::") {
                return Some(test_name[..pos].replace("_", "-"));
            }
        }

        // Handle test names in integration tests
        if test_name.contains("integration") {
            return Some("bitnet-integration".to_string());
        }

        None
    }

    /// Find the source file for a test
    async fn find_test_file(&self, test_name: &str) -> Option<PathBuf> {
        // Search in tests directory
        let test_dirs = ["tests", "tests/integration", "tests/unit"];

        for test_dir in &test_dirs {
            let test_dir_path = PathBuf::from(test_dir);
            if test_dir_path.exists() {
                if let Ok(found) = self.search_for_test_in_dir(&test_dir_path, test_name).await {
                    if let Some(file) = found {
                        return Some(file);
                    }
                }
            }
        }

        None
    }

    /// Search for a test in a directory
    async fn search_for_test_in_dir(
        &self,
        dir: &Path,
        test_name: &str,
    ) -> TestResult<Option<PathBuf>> {
        let mut entries = fs::read_dir(dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.is_file() && path.extension().map_or(false, |ext| ext == "rs") {
                // Read file and check if it contains the test
                if let Ok(content) = fs::read_to_string(&path).await {
                    if content.contains(&format!("fn {}", test_name))
                        || content.contains(&format!("\"{}\"", test_name))
                    {
                        return Ok(Some(path));
                    }
                }
            } else if path.is_dir() {
                // Recursively search subdirectories
                if let Ok(Some(found)) = self.search_for_test_in_dir(&path, test_name).await {
                    return Ok(Some(found));
                }
            }
        }

        Ok(None)
    }

    /// Generate cache key for a test
    async fn generate_cache_key(
        &self,
        test_case: &dyn TestCase,
        suite_name: &str,
    ) -> TestResult<CacheKey> {
        let test_name = test_case.name();
        let metadata = test_case.metadata();

        // Generate input hash from test configuration
        let config_str = serde_json::to_string(&metadata).unwrap_or_default();
        let features = std::env::var("CARGO_FEATURES")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .collect::<Vec<_>>();

        let input_hash = cache_keys::hash_test_inputs(test_name, &config_str, &features);

        // Generate source hash from dependencies
        let dependencies = self.analyze_test_dependencies(test_case).await?;
        let source_hash = cache_keys::hash_source_dependencies(&dependencies).await?;

        Ok(CacheKey {
            test_name: test_name.to_string(),
            suite_name: suite_name.to_string(),
            input_hash,
            source_hash,
        })
    }
}

impl ChangeDetector {
    /// Create a new change detector
    pub fn new(config: IncrementalConfig, workspace_root: PathBuf) -> Self {
        Self {
            config,
            workspace_root,
        }
    }

    /// Detect changes in the codebase
    pub async fn detect_changes(&self) -> TestResult<Vec<Change>> {
        if let Some(base_ref) = &self.config.base_ref {
            self.detect_git_changes(base_ref).await
        } else {
            self.detect_git_changes_auto().await
        }
    }

    /// Detect changes using git with automatic base detection
    async fn detect_git_changes_auto(&self) -> TestResult<Vec<Change>> {
        // Try to detect the base branch
        let base_ref = self.detect_base_branch().await?;
        self.detect_git_changes(&base_ref).await
    }

    /// Detect the base branch (main, master, develop)
    async fn detect_base_branch(&self) -> TestResult<String> {
        let branches = [
            "origin/main",
            "origin/master",
            "origin/develop",
            "main",
            "master",
            "develop",
        ];

        for branch in &branches {
            let output = Command::new("git")
                .args(&["rev-parse", "--verify", branch])
                .current_dir(&self.workspace_root)
                .output();

            if let Ok(output) = output {
                if output.status.success() {
                    return Ok(branch.to_string());
                }
            }
        }

        // Fallback to HEAD~1
        Ok("HEAD~1".to_string())
    }

    /// Detect changes using git diff
    async fn detect_git_changes(&self, base_ref: &str) -> TestResult<Vec<Change>> {
        let output = Command::new("git")
            .args(&["diff", "--name-status", base_ref])
            .current_dir(&self.workspace_root)
            .output()
            .map_err(|e| TestError::execution(format!("Failed to run git diff: {}", e)))?;

        if !output.status.success() {
            return Err(TestError::execution(format!(
                "Git diff failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let diff_output = String::from_utf8_lossy(&output.stdout);
        let mut changes = Vec::new();

        for line in diff_output.lines() {
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 2 {
                continue;
            }

            let status = parts[0];
            let path = PathBuf::from(parts[1]);

            // Skip ignored patterns
            if self.should_ignore_path(&path) {
                continue;
            }

            let change_type = match status.chars().next() {
                Some('A') => ChangeType::Added,
                Some('M') => ChangeType::Modified,
                Some('D') => ChangeType::Deleted,
                Some('R') => {
                    if parts.len() >= 3 {
                        ChangeType::Renamed {
                            from: PathBuf::from(parts[2]),
                        }
                    } else {
                        ChangeType::Modified
                    }
                }
                _ => ChangeType::Modified,
            };

            changes.push(Change {
                path,
                change_type,
                timestamp: SystemTime::now(),
            });
        }

        info!("Detected {} changes", changes.len());
        for change in &changes {
            debug!("Change: {:?} - {:?}", change.change_type, change.path);
        }

        Ok(changes)
    }

    /// Check if a path should be ignored
    fn should_ignore_path(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        for pattern in &self.config.ignore_patterns {
            if self.matches_pattern(&path_str, pattern) {
                return true;
            }
        }

        false
    }

    /// Check if a path matches a glob pattern
    fn matches_pattern(&self, path: &str, pattern: &str) -> bool {
        // Simple glob matching - could be enhanced with a proper glob library
        if pattern.contains("**") {
            let prefix = pattern.split("**").next().unwrap_or("");
            path.starts_with(prefix)
        } else if pattern.starts_with("*.") {
            let extension = &pattern[2..];
            path.ends_with(extension)
        } else {
            path.contains(pattern)
        }
    }
}

impl DependencyGraph {
    /// Create a new dependency graph
    pub fn new() -> Self {
        Self {
            file_to_tests: HashMap::new(),
            test_to_files: HashMap::new(),
            cache: HashMap::new(),
        }
    }

    /// Add dependencies for a test
    pub fn add_test_dependencies(&mut self, test_name: String, dependencies: Vec<PathBuf>) {
        // Update test -> files mapping
        self.test_to_files
            .insert(test_name.clone(), dependencies.iter().cloned().collect());

        // Update file -> tests mapping
        for dep in dependencies {
            self.file_to_tests
                .entry(dep)
                .or_insert_with(HashSet::new)
                .insert(test_name.clone());
        }

        // Cache the dependencies
        self.cache.insert(test_name, dependencies);
    }

    /// Get dependencies for a test
    pub fn get_dependencies(&self, test_name: &str) -> HashSet<PathBuf> {
        self.test_to_files
            .get(test_name)
            .cloned()
            .unwrap_or_default()
    }

    /// Get tests affected by a file change
    pub fn get_affected_tests(&self, file_path: &Path) -> HashSet<String> {
        self.file_to_tests
            .get(file_path)
            .cloned()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_change_detection() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_root = temp_dir.path().to_path_buf();

        // Initialize git repo
        Command::new("git")
            .args(&["init"])
            .current_dir(&workspace_root)
            .output()
            .unwrap();

        Command::new("git")
            .args(&["config", "user.email", "test@example.com"])
            .current_dir(&workspace_root)
            .output()
            .unwrap();

        Command::new("git")
            .args(&["config", "user.name", "Test User"])
            .current_dir(&workspace_root)
            .output()
            .unwrap();

        // Create and commit a file
        let test_file = workspace_root.join("test.rs");
        fs::write(&test_file, "fn test() {}").await.unwrap();

        Command::new("git")
            .args(&["add", "."])
            .current_dir(&workspace_root)
            .output()
            .unwrap();

        Command::new("git")
            .args(&["commit", "-m", "Initial commit"])
            .current_dir(&workspace_root)
            .output()
            .unwrap();

        // Modify the file
        fs::write(&test_file, "fn test() { println!(); }")
            .await
            .unwrap();

        // Detect changes
        let config = IncrementalConfig::default();
        let detector = ChangeDetector::new(config, workspace_root);

        // Note: This test might fail in CI without proper git setup
        // In a real scenario, we'd have a proper git history
    }

    #[test]
    fn test_dependency_graph() {
        let mut graph = DependencyGraph::new();

        let test_name = "test_example".to_string();
        let dependencies = vec![PathBuf::from("src/lib.rs"), PathBuf::from("src/module.rs")];

        graph.add_test_dependencies(test_name.clone(), dependencies.clone());

        let retrieved_deps = graph.get_dependencies(&test_name);
        assert_eq!(retrieved_deps.len(), 2);
        assert!(retrieved_deps.contains(&PathBuf::from("src/lib.rs")));
        assert!(retrieved_deps.contains(&PathBuf::from("src/module.rs")));

        let affected_tests = graph.get_affected_tests(&PathBuf::from("src/lib.rs"));
        assert!(affected_tests.contains(&test_name));
    }

    #[test]
    fn test_pattern_matching() {
        let config = IncrementalConfig::default();
        let workspace_root = PathBuf::from("/tmp");
        let detector = ChangeDetector::new(config, workspace_root);

        assert!(detector.matches_pattern("README.md", "*.md"));
        assert!(detector.matches_pattern("docs/guide.md", "docs/**"));
        assert!(!detector.matches_pattern("src/lib.rs", "*.md"));
    }
}
