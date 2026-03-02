//! Model download management utilities.
//!
//! URL resolution, progress tracking, and integrity checks.

use std::collections::HashMap;

/// A downloadable model file.
#[derive(Debug, Clone)]
pub struct DownloadSpec {
    pub url: String,
    pub filename: String,
    pub expected_bytes: Option<u64>,
    pub sha256: Option<String>,
}

/// A model's complete download manifest.
#[derive(Debug, Clone)]
pub struct DownloadManifest {
    pub model_id: String,
    pub files: Vec<DownloadSpec>,
    pub total_bytes: Option<u64>,
}

impl DownloadManifest {
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    pub fn total_expected_bytes(&self) -> u64 {
        self.files.iter().filter_map(|f| f.expected_bytes).sum()
    }

    pub fn has_checksums(&self) -> bool {
        self.files.iter().all(|f| f.sha256.is_some())
    }
}

/// Well-known model download specs.
pub fn phi4_manifest() -> DownloadManifest {
    let base = "https://huggingface.co/microsoft/phi-4/resolve/main";
    DownloadManifest {
        model_id: "microsoft/phi-4".into(),
        files: (1..=6)
            .map(|i| DownloadSpec {
                url: format!("{base}/model-{i:05}-of-00006.safetensors"),
                filename: format!("model-{i:05}-of-00006.safetensors"),
                expected_bytes: Some(4_800_000_000),
                sha256: None,
            })
            .chain(std::iter::once(DownloadSpec {
                url: format!("{base}/tokenizer.json"),
                filename: "tokenizer.json".into(),
                expected_bytes: Some(17_000_000),
                sha256: None,
            }))
            .collect(),
        total_bytes: Some(29_000_000_000),
    }
}

pub fn llama3_8b_manifest() -> DownloadManifest {
    let base = "https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main";
    DownloadManifest {
        model_id: "meta-llama/Meta-Llama-3-8B".into(),
        files: (1..=4)
            .map(|i| DownloadSpec {
                url: format!("{base}/model-{i:05}-of-00004.safetensors"),
                filename: format!("model-{i:05}-of-00004.safetensors"),
                expected_bytes: Some(4_900_000_000),
                sha256: None,
            })
            .collect(),
        total_bytes: Some(16_000_000_000),
    }
}

/// Supported model IDs and their manifests.
pub fn known_models() -> HashMap<String, DownloadManifest> {
    let mut map = HashMap::new();
    map.insert("microsoft/phi-4".into(), phi4_manifest());
    map.insert("meta-llama/Meta-Llama-3-8B".into(), llama3_8b_manifest());
    map
}

/// Download progress tracking.
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub file_index: usize,
    pub total_files: usize,
    pub bytes_downloaded: u64,
    pub bytes_total: u64,
    pub current_file: String,
}

impl DownloadProgress {
    pub fn percent(&self) -> f32 {
        if self.bytes_total == 0 {
            return 0.0;
        }
        (self.bytes_downloaded as f32 / self.bytes_total as f32) * 100.0
    }

    pub fn file_percent(&self) -> f32 {
        if self.total_files == 0 {
            return 0.0;
        }
        (self.file_index as f32 / self.total_files as f32) * 100.0
    }
}

/// Resolve a model ID to its download manifest.
pub fn resolve_model(model_id: &str) -> Option<DownloadManifest> {
    known_models().remove(model_id)
}

/// Validate downloaded files match expected sizes.
pub fn validate_download(
    manifest: &DownloadManifest,
    actual_sizes: &HashMap<String, u64>,
) -> Vec<String> {
    let mut issues = Vec::new();

    for file in &manifest.files {
        match actual_sizes.get(&file.filename) {
            None => issues.push(format!("missing file: {}", file.filename)),
            Some(&actual) => {
                if let Some(expected) = file.expected_bytes {
                    let ratio = actual as f64 / expected as f64;
                    if !(0.9..=1.1).contains(&ratio) {
                        issues.push(format!(
                            "{}: size {actual} differs from expected \
                             {expected} (ratio: {ratio:.2})",
                            file.filename
                        ));
                    }
                }
            }
        }
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi4_manifest() {
        let m = phi4_manifest();
        assert_eq!(m.model_id, "microsoft/phi-4");
        assert_eq!(m.file_count(), 7); // 6 shards + tokenizer
    }

    #[test]
    fn test_phi4_total_bytes() {
        let m = phi4_manifest();
        assert!(m.total_expected_bytes() > 28_000_000_000);
    }

    #[test]
    fn test_llama3_manifest() {
        let m = llama3_8b_manifest();
        assert_eq!(m.file_count(), 4);
    }

    #[test]
    fn test_known_models() {
        let models = known_models();
        assert!(models.contains_key("microsoft/phi-4"));
        assert!(models.contains_key("meta-llama/Meta-Llama-3-8B"));
    }

    #[test]
    fn test_resolve_known() {
        let m = resolve_model("microsoft/phi-4");
        assert!(m.is_some());
    }

    #[test]
    fn test_resolve_unknown() {
        let m = resolve_model("nonexistent/model");
        assert!(m.is_none());
    }

    #[test]
    fn test_progress_percent() {
        let p = DownloadProgress {
            file_index: 1,
            total_files: 4,
            bytes_downloaded: 50,
            bytes_total: 100,
            current_file: "shard.safetensors".into(),
        };
        assert!((p.percent() - 50.0).abs() < 0.01);
        assert!((p.file_percent() - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_progress_zero() {
        let p = DownloadProgress {
            file_index: 0,
            total_files: 0,
            bytes_downloaded: 0,
            bytes_total: 0,
            current_file: String::new(),
        };
        assert_eq!(p.percent(), 0.0);
    }

    #[test]
    fn test_validate_ok() {
        let m = DownloadManifest {
            model_id: "test".into(),
            files: vec![DownloadSpec {
                url: "http://x".into(),
                filename: "a.bin".into(),
                expected_bytes: Some(100),
                sha256: None,
            }],
            total_bytes: Some(100),
        };
        let mut sizes = HashMap::new();
        sizes.insert("a.bin".into(), 100);
        assert!(validate_download(&m, &sizes).is_empty());
    }

    #[test]
    fn test_validate_missing() {
        let m = DownloadManifest {
            model_id: "test".into(),
            files: vec![DownloadSpec {
                url: "http://x".into(),
                filename: "a.bin".into(),
                expected_bytes: Some(100),
                sha256: None,
            }],
            total_bytes: Some(100),
        };
        let sizes = HashMap::new();
        let issues = validate_download(&m, &sizes);
        assert!(issues.iter().any(|i| i.contains("missing")));
    }

    #[test]
    fn test_validate_wrong_size() {
        let m = DownloadManifest {
            model_id: "test".into(),
            files: vec![DownloadSpec {
                url: "http://x".into(),
                filename: "a.bin".into(),
                expected_bytes: Some(1000),
                sha256: None,
            }],
            total_bytes: Some(1000),
        };
        let mut sizes = HashMap::new();
        sizes.insert("a.bin".into(), 100);
        let issues = validate_download(&m, &sizes);
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_has_checksums() {
        let m = DownloadManifest {
            model_id: "test".into(),
            files: vec![DownloadSpec {
                url: "http://x".into(),
                filename: "a.bin".into(),
                expected_bytes: Some(100),
                sha256: Some("abc123".into()),
            }],
            total_bytes: Some(100),
        };
        assert!(m.has_checksums());
    }

    #[test]
    fn test_no_checksums() {
        let m = phi4_manifest();
        assert!(!m.has_checksums());
    }
}
