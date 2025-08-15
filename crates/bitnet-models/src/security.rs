// Security utilities for model loading and verification
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;

/// Model security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSecurity {
    /// Trusted model sources
    pub trusted_sources: Vec<String>,
    /// Required hash verification
    pub require_hash_verification: bool,
    /// Maximum model file size (in bytes)
    pub max_model_size: u64,
    /// Known model hashes
    pub known_hashes: HashMap<String, String>,
}

impl Default for ModelSecurity {
    fn default() -> Self {
        Self {
            trusted_sources: vec![
                "https://huggingface.co/".to_string(),
                "https://github.com/microsoft/BitNet/".to_string(),
            ],
            require_hash_verification: true,
            max_model_size: 50 * 1024 * 1024 * 1024, // 50GB max
            known_hashes: HashMap::new(),
        }
    }
}

/// Model integrity verifier
pub struct ModelVerifier {
    config: ModelSecurity,
}

impl ModelVerifier {
    /// Create a new model verifier with the given configuration
    pub fn new(config: ModelSecurity) -> Self {
        Self { config }
    }

    /// Verify a model file's integrity
    pub fn verify_model<P: AsRef<Path>>(&self, path: P, expected_hash: Option<&str>) -> Result<()> {
        let path = path.as_ref();

        // Check file size
        let metadata = std::fs::metadata(path)?;
        if metadata.len() > self.config.max_model_size {
            return Err(anyhow!(
                "Model file too large: {} bytes (max: {} bytes)",
                metadata.len(),
                self.config.max_model_size
            ));
        }

        // Verify hash if required or provided
        if self.config.require_hash_verification || expected_hash.is_some() {
            let computed_hash = self.compute_file_hash(path)?;

            if let Some(expected) = expected_hash {
                if computed_hash != expected {
                    return Err(anyhow!(
                        "Hash verification failed for {}: expected {}, got {}",
                        path.display(),
                        expected,
                        computed_hash
                    ));
                }
            } else if let Some(known_hash) =
                self.config.known_hashes.get(path.to_string_lossy().as_ref())
            {
                if computed_hash != *known_hash {
                    return Err(anyhow!(
                        "Hash verification failed for {}: expected {}, got {}",
                        path.display(),
                        known_hash,
                        computed_hash
                    ));
                }
            } else if self.config.require_hash_verification {
                return Err(anyhow!(
                    "Hash verification required but no expected hash provided for {}",
                    path.display()
                ));
            }
        }

        Ok(())
    }

    /// Compute SHA256 hash of a file
    pub fn compute_file_hash<P: AsRef<Path>>(&self, path: P) -> Result<String> {
        let mut file = std::fs::File::open(path)?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Verify a model source URL is trusted
    pub fn verify_source(&self, url: &str) -> Result<()> {
        for trusted_source in &self.config.trusted_sources {
            if url.starts_with(trusted_source) {
                return Ok(());
            }
        }

        Err(anyhow!(
            "Untrusted model source: {}. Trusted sources: {:?}",
            url,
            self.config.trusted_sources
        ))
    }

    /// Add a known hash for a model
    pub fn add_known_hash(&mut self, model_path: String, hash: String) {
        self.config.known_hashes.insert(model_path, hash);
    }

    /// Get the security configuration
    pub fn config(&self) -> &ModelSecurity {
        &self.config
    }
}

/// Secure model downloader with integrity verification
pub struct SecureModelDownloader {
    verifier: ModelVerifier,
    client: reqwest::Client,
}

impl SecureModelDownloader {
    /// Create a new secure model downloader
    pub fn new(config: ModelSecurity) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // 5 minute timeout
            .build()
            .expect("Failed to create HTTP client");

        Self { verifier: ModelVerifier::new(config), client }
    }

    /// Download a model with integrity verification
    pub async fn download_model(
        &self,
        url: &str,
        destination: &Path,
        expected_hash: Option<&str>,
    ) -> Result<()> {
        // Verify source is trusted
        self.verifier.verify_source(url)?;

        // Download the model
        tracing::info!("Downloading model from: {}", url);
        let response = self.client.get(url).send().await?;

        if !response.status().is_success() {
            return Err(anyhow!("Failed to download model: HTTP {}", response.status()));
        }

        // Check content length
        if let Some(content_length) = response.content_length() {
            if content_length > self.verifier.config().max_model_size {
                return Err(anyhow!(
                    "Model too large: {} bytes (max: {} bytes)",
                    content_length,
                    self.verifier.config().max_model_size
                ));
            }
        }

        // Write to temporary file first
        let temp_path = destination.with_extension("tmp");
        let mut file = std::fs::File::create(&temp_path)?;
        let mut stream = response.bytes_stream();

        use futures::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            std::io::Write::write_all(&mut file, &chunk)?;
        }

        drop(file); // Ensure file is closed

        // Verify the downloaded file
        self.verifier.verify_model(&temp_path, expected_hash)?;

        // Move to final destination
        std::fs::rename(&temp_path, destination)?;

        tracing::info!("Successfully downloaded and verified model: {}", destination.display());
        Ok(())
    }
}

/// Security audit utilities
pub mod audit {
    use super::*;

    /// Audit result for a model file
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ModelAuditResult {
        pub path: String,
        pub size: u64,
        pub hash: String,
        pub is_trusted_source: bool,
        pub has_known_hash: bool,
        pub security_issues: Vec<String>,
    }

    /// Security auditor for models
    pub struct ModelAuditor {
        verifier: ModelVerifier,
    }

    impl ModelAuditor {
        pub fn new(config: ModelSecurity) -> Self {
            Self { verifier: ModelVerifier::new(config) }
        }

        /// Audit a single model file
        pub fn audit_model<P: AsRef<Path>>(&self, path: P) -> Result<ModelAuditResult> {
            let path = path.as_ref();
            let path_str = path.to_string_lossy().to_string();

            let metadata = std::fs::metadata(path)?;
            let hash = self.verifier.compute_file_hash(path)?;
            let has_known_hash = self.verifier.config().known_hashes.contains_key(&path_str);

            let mut security_issues = Vec::new();

            // Check file size
            if metadata.len() > self.verifier.config().max_model_size {
                security_issues
                    .push(format!("File size exceeds maximum: {} bytes", metadata.len()));
            }

            // Check if hash verification would fail
            if self.verifier.config().require_hash_verification && !has_known_hash {
                security_issues
                    .push("Hash verification required but no known hash available".to_string());
            }

            // Check file permissions (Unix-like systems)
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let perms = metadata.permissions();
                if perms.mode() & 0o002 != 0 {
                    security_issues.push("File is world-writable".to_string());
                }
            }

            Ok(ModelAuditResult {
                path: path_str,
                size: metadata.len(),
                hash,
                is_trusted_source: false, // Can't determine from file alone
                has_known_hash,
                security_issues,
            })
        }

        /// Audit all models in a directory
        pub fn audit_directory<P: AsRef<Path>>(&self, dir: P) -> Result<Vec<ModelAuditResult>> {
            let mut results = Vec::new();
            let dir = dir.as_ref();

            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_file() {
                    // Check if it looks like a model file
                    if let Some(ext) = path.extension() {
                        if matches!(ext.to_str(), Some("gguf") | Some("safetensors") | Some("bin"))
                        {
                            match self.audit_model(&path) {
                                Ok(result) => results.push(result),
                                Err(e) => {
                                    tracing::warn!("Failed to audit {}: {}", path.display(), e);
                                }
                            }
                        }
                    }
                }
            }

            Ok(results)
        }

        /// Generate a security report
        pub fn generate_report(&self, results: &[ModelAuditResult]) -> String {
            let mut report = String::new();
            report.push_str("# Model Security Audit Report\n\n");

            let total_models = results.len();
            let models_with_issues =
                results.iter().filter(|r| !r.security_issues.is_empty()).count();
            let models_with_known_hashes = results.iter().filter(|r| r.has_known_hash).count();

            report.push_str("## Summary\n\n");
            report.push_str(&format!("- Total models audited: {}\n", total_models));
            report.push_str(&format!("- Models with security issues: {}\n", models_with_issues));
            report.push_str(&format!("- Models with known hashes: {}\n", models_with_known_hashes));
            report.push_str(&format!(
                "- Security coverage: {:.1}%\n\n",
                (models_with_known_hashes as f64 / total_models as f64) * 100.0
            ));

            if models_with_issues > 0 {
                report.push_str("## Security Issues\n\n");
                for result in results.iter().filter(|r| !r.security_issues.is_empty()) {
                    report.push_str(&format!("### {}\n\n", result.path));
                    for issue in &result.security_issues {
                        report.push_str(&format!("- {}\n", issue));
                    }
                    report.push('\n');
                }
            }

            report.push_str("## Model Details\n\n");
            for result in results {
                report.push_str(&format!("### {}\n\n", result.path));
                report.push_str(&format!("- Size: {} bytes\n", result.size));
                report.push_str(&format!("- SHA256: {}\n", result.hash));
                report.push_str(&format!("- Has known hash: {}\n", result.has_known_hash));
                if !result.security_issues.is_empty() {
                    report.push_str("- Issues:\n");
                    for issue in &result.security_issues {
                        report.push_str(&format!("  - {}\n", issue));
                    }
                }
                report.push('\n');
            }

            report
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_hash_computation() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"test content").unwrap();
        temp_file.flush().unwrap();

        let verifier = ModelVerifier::new(ModelSecurity::default());
        let hash = verifier.compute_file_hash(temp_file.path()).unwrap();

        // Verify that we get a valid SHA256 hash (64 hex characters)
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));

        // Test that same file produces same hash
        let hash2 = verifier.compute_file_hash(temp_file.path()).unwrap();
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_source_verification() {
        let verifier = ModelVerifier::new(ModelSecurity::default());

        // Should accept trusted sources
        assert!(verifier.verify_source("https://huggingface.co/model").is_ok());
        assert!(verifier.verify_source("https://github.com/microsoft/BitNet/model").is_ok());

        // Should reject untrusted sources
        assert!(verifier.verify_source("https://malicious.com/model").is_err());
    }

    #[test]
    fn test_file_size_limits() {
        let mut config = ModelSecurity::default();
        config.max_model_size = 100; // Very small limit for testing

        let verifier = ModelVerifier::new(config);

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&vec![0u8; 200]).unwrap(); // Larger than limit

        assert!(verifier.verify_model(temp_file.path(), None).is_err());
    }
}
