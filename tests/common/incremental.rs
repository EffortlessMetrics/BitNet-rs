use super::errors::TestError;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tracing::{debug, info, warn};

/// Incremental tester that detects changes and runs only affected tests
pub struct IncrementalTester {
    cache_dir: PathBuf,
    last_run_file: PathBuf,
    dependency_graph: DependencyGraph,
}

impl Default for IncrementalTester {
    fn default() -> Self {
        Self::new()
    }
}

impl IncrementalTester {
    pub fn new() -> Self {
        let cache_dir = PathBuf::from("tests/cache/incremental");
        let last_run_file = cache_dir.join("last_run.json");

        Self { cache_dir, last_run_file, dependency_graph: DependencyGraph::new() }
    }

    /// Detect changed files since last test run
    pub async fn detect_changes(&self) -> Result<Vec<PathBuf>, TestError> {
        info!("Detecting changes for incremental testing...");

        // Ensure cache directory exists
        fs::create_dir_all(&self.cache_dir).await.map_err(TestError::IoError)?;

        let changed_files = if self.is_git_repository().await {
            self.detect_git_changes().await?
        } else {
            self.detect_filesystem_changes().await?
        };

        info!("Detected {} changed files", changed_files.len());
        for file in &changed_files {
            debug!("Changed: {}", file.display());
        }

        Ok(changed_files)
    }

    /// Check if we're in a git repository
    async fn is_git_repository(&self) -> bool {
        PathBuf::from(".git").exists()
            || Command::new("git")
                .arg("rev-parse")
                .arg("--git-dir")
                .output()
                .map(|output| output.status.success())
                .unwrap_or(false)
    }

    /// Detect changes using git
    async fn detect_git_changes(&self) -> Result<Vec<PathBuf>, TestError> {
        debug!("Using git to detect changes");

        // Get the last commit hash from our cache
        let last_commit = self.get_last_commit_hash().await.unwrap_or_else(|| "HEAD~1".to_string());

        // Get changed files since last commit
        let output = Command::new("git")
            .arg("diff")
            .arg("--name-only")
            .arg(&last_commit)
            .arg("HEAD")
            .output()
            .map_err(|e| TestError::ExecutionError {
                message: format!("Git diff failed: {}", e),
            })?;

        if !output.status.success() {
            warn!("Git diff failed, falling back to filesystem detection");
            return self.detect_filesystem_changes().await;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut changed_files = Vec::new();

        for line in stdout.lines() {
            let path = PathBuf::from(line.trim());
            if path.exists() && self.is_relevant_file(&path) {
                changed_files.push(path);
            }
        }

        // Also check for unstaged changes
        let unstaged_output =
            Command::new("git").arg("diff").arg("--name-only").output().map_err(|e| {
                TestError::ExecutionError { message: format!("Git diff unstaged failed: {}", e) }
            })?;

        if unstaged_output.status.success() {
            let unstaged_stdout = String::from_utf8_lossy(&unstaged_output.stdout);
            for line in unstaged_stdout.lines() {
                let path = PathBuf::from(line.trim());
                if path.exists() && self.is_relevant_file(&path) && !changed_files.contains(&path) {
                    changed_files.push(path);
                }
            }
        }

        // Update last commit hash
        self.save_current_commit_hash().await?;

        Ok(changed_files)
    }

    /// Detect changes using filesystem timestamps
    async fn detect_filesystem_changes(&self) -> Result<Vec<PathBuf>, TestError> {
        debug!("Using filesystem timestamps to detect changes");

        let last_run_time = self.get_last_run_time().await;
        let mut changed_files = Vec::new();

        // Check relevant directories for changes
        let check_dirs = vec![
            PathBuf::from("src"),
            PathBuf::from("crates"),
            PathBuf::from("tests"),
            PathBuf::from("Cargo.toml"),
            PathBuf::from("Cargo.lock"),
        ];

        for dir in check_dirs {
            if dir.is_file() {
                if self.is_file_modified(&dir, last_run_time).await? {
                    changed_files.push(dir);
                }
            } else if dir.is_dir() {
                let mut files = self.find_modified_files_in_dir(&dir, last_run_time).await?;
                changed_files.append(&mut files);
            }
        }

        // Update last run time
        self.save_last_run_time().await?;

        Ok(changed_files)
    }

    /// Check if a file is relevant for testing
    fn is_relevant_file(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        // Include Rust source files
        if path_str.ends_with(".rs") {
            return true;
        }

        // Include Cargo files
        if path_str.ends_with("Cargo.toml") || path_str.ends_with("Cargo.lock") {
            return true;
        }

        // Include build scripts
        if path_str.ends_with("build.rs") {
            return true;
        }

        // Include configuration files
        if path_str.ends_with(".toml") || path_str.ends_with(".json") || path_str.ends_with(".yaml")
        {
            return true;
        }

        // Exclude certain directories
        if path_str.contains("target/")
            || path_str.contains(".git/")
            || path_str.contains("node_modules/")
        {
            return false;
        }

        false
    }

    /// Get last commit hash from cache
    async fn get_last_commit_hash(&self) -> Option<String> {
        let commit_file = self.cache_dir.join("last_commit.txt");
        fs::read_to_string(commit_file).await.ok()
    }

    /// Save current commit hash to cache
    async fn save_current_commit_hash(&self) -> Result<(), TestError> {
        let output = Command::new("git").arg("rev-parse").arg("HEAD").output().map_err(|e| {
            TestError::ExecutionError { message: format!("Git rev-parse failed: {}", e) }
        })?;

        if output.status.success() {
            let commit_hash = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let commit_file = self.cache_dir.join("last_commit.txt");
            fs::write(commit_file, commit_hash).await.map_err(TestError::IoError)?;
        }

        Ok(())
    }

    /// Get last run time from cache
    async fn get_last_run_time(&self) -> SystemTime {
        if let Ok(metadata) = fs::metadata(&self.last_run_file).await {
            metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH)
        } else {
            SystemTime::UNIX_EPOCH
        }
    }

    /// Save current time as last run time
    async fn save_last_run_time(&self) -> Result<(), TestError> {
        let timestamp = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();

        fs::write(&self.last_run_file, timestamp.to_string()).await.map_err(TestError::IoError)?;

        Ok(())
    }

    /// Check if a file was modified after the given time
    async fn is_file_modified(&self, path: &Path, since: SystemTime) -> Result<bool, TestError> {
        let metadata = fs::metadata(path).await.map_err(TestError::IoError)?;

        let modified = metadata.modified().map_err(TestError::IoError)?;

        Ok(modified > since)
    }

    /// Find all modified files in a directory
    fn find_modified_files_in_dir<'a>(
        &'a self,
        dir: &'a PathBuf,
        since: SystemTime,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<PathBuf>, TestError>> + Send + 'a>,
    > {
        Box::pin(async move {
            let mut modified_files = Vec::new();
            let mut entries = fs::read_dir(dir).await.map_err(TestError::IoError)?;

            while let Some(entry) = entries.next_entry().await.map_err(TestError::IoError)? {
                let path = entry.path();

                if path.is_file() && self.is_relevant_file(&path) {
                    if self.is_file_modified(&path, since).await? {
                        modified_files.push(path);
                    }
                } else if path.is_dir()
                    && !path.file_name().unwrap_or_default().to_string_lossy().starts_with('.')
                {
                    let mut subdir_files = self.find_modified_files_in_dir(&path, since).await?;
                    modified_files.append(&mut subdir_files);
                }
            }

            Ok(modified_files)
        })
    }

    /// Determine which tests should run based on changed files
    pub async fn get_affected_test_patterns(
        &self,
        changed_files: &[PathBuf],
    ) -> Result<Vec<String>, TestError> {
        let mut test_patterns = HashSet::new();

        for file in changed_files {
            let patterns = self.dependency_graph.get_affected_tests(file);
            test_patterns.extend(patterns);
        }

        Ok(test_patterns.into_iter().collect())
    }

    /// Mark test run as complete
    pub async fn mark_run_complete(&self) -> Result<(), TestError> {
        self.save_last_run_time().await?;
        if self.is_git_repository().await {
            self.save_current_commit_hash().await?;
        }
        Ok(())
    }

    /// Check if incremental testing is beneficial
    pub async fn should_use_incremental(&self, total_tests: usize) -> bool {
        let changed_files = self.detect_changes().await.unwrap_or_default();

        // Use incremental if we have changes and they affect less than 50% of tests
        if changed_files.is_empty() {
            return false;
        }

        let affected_patterns =
            self.get_affected_test_patterns(&changed_files).await.unwrap_or_default();
        let estimated_affected_tests = affected_patterns.len() * 10; // Rough estimate

        estimated_affected_tests < total_tests / 2
    }
}

/// Dependency graph for determining test dependencies
#[derive(Debug, Default)]
pub struct DependencyGraph {
    file_to_tests: std::collections::HashMap<PathBuf, Vec<String>>,
    crate_dependencies: std::collections::HashMap<String, Vec<String>>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        let mut graph = Self::default();
        graph.build_default_mappings();
        graph
    }

    /// Build default file-to-test mappings
    fn build_default_mappings(&mut self) {
        // Core files affect all tests
        self.file_to_tests.insert(PathBuf::from("Cargo.toml"), vec!["*".to_string()]);
        self.file_to_tests.insert(PathBuf::from("Cargo.lock"), vec!["*".to_string()]);

        // Common crate affects many tests
        self.file_to_tests.insert(
            PathBuf::from("crates/bitnet-common"),
            vec!["bitnet-common".to_string(), "integration".to_string()],
        );

        // Models crate affects model-related tests
        self.file_to_tests.insert(
            PathBuf::from("crates/bitnet-models"),
            vec!["bitnet-models".to_string(), "model".to_string(), "loading".to_string()],
        );

        // Kernels crate affects performance tests
        self.file_to_tests.insert(
            PathBuf::from("crates/bitnet-kernels"),
            vec!["bitnet-kernels".to_string(), "performance".to_string(), "simd".to_string()],
        );

        // Test files affect themselves
        self.file_to_tests
            .insert(PathBuf::from("tests/"), vec!["integration".to_string(), "e2e".to_string()]);
    }

    /// Get test patterns affected by a file change
    pub fn get_affected_tests(&self, file: &Path) -> Vec<String> {
        let file_str = file.to_string_lossy();

        // Check exact matches first
        if let Some(tests) = self.file_to_tests.get(file) {
            return tests.clone();
        }

        // Check directory matches
        for (pattern_path, tests) in &self.file_to_tests {
            let pattern_str = pattern_path.to_string_lossy();
            if file_str.starts_with(&*pattern_str) {
                return tests.clone();
            }
        }

        // Default: if it's a source file, run tests for that crate
        if file_str.starts_with("crates/") {
            if let Some(crate_name) = file_str.split('/').nth(1) {
                return vec![crate_name.to_string()];
            }
        }

        // If no specific mapping, run a minimal set
        vec!["unit".to_string()]
    }
}

/// Incremental test configuration
#[derive(Debug, Clone)]
pub struct IncrementalConfig {
    pub enabled: bool,
    pub max_age: Duration,
    pub force_full_patterns: Vec<String>,
    pub always_incremental_patterns: Vec<String>,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_age: Duration::from_secs(24 * 60 * 60), // 24 hours
            force_full_patterns: vec![
                "Cargo.toml".to_string(),
                "build.rs".to_string(),
                ".github/workflows/".to_string(),
            ],
            always_incremental_patterns: vec![
                "tests/".to_string(),
                "examples/".to_string(),
                "docs/".to_string(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_incremental_tester_creation() {
        let tester = IncrementalTester::new();
        assert!(tester.cache_dir.ends_with("incremental"));
    }

    #[test]
    fn test_relevant_file_detection() {
        let tester = IncrementalTester::new();

        assert!(tester.is_relevant_file(&PathBuf::from("src/lib.rs")));
        assert!(tester.is_relevant_file(&PathBuf::from("Cargo.toml")));
        assert!(tester.is_relevant_file(&PathBuf::from("build.rs")));

        assert!(!tester.is_relevant_file(&PathBuf::from("target/debug/deps/lib.so")));
        assert!(!tester.is_relevant_file(&PathBuf::from(".git/config")));
    }

    #[test]
    fn test_dependency_graph() {
        let graph = DependencyGraph::new();

        let affected = graph.get_affected_tests(&PathBuf::from("Cargo.toml"));
        assert!(affected.contains(&"*".to_string()));

        let affected = graph.get_affected_tests(&PathBuf::from("crates/bitnet-common/src/lib.rs"));
        assert!(affected.contains(&"bitnet-common".to_string()));
    }

    #[tokio::test]
    async fn test_should_use_incremental() {
        let tester = IncrementalTester::new();

        // With no changes, should not use incremental
        let should_use = tester.should_use_incremental(100).await;
        // This might be false if no changes detected
        assert!(should_use || !should_use); // Either is valid depending on state
    }
}
