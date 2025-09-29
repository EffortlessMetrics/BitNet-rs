# [Tokenizer] Implement SmartDownload strategy in TokenizerStrategyResolver

## Problem Description

The `SmartDownload` strategy in `TokenizerStrategyResolver::resolve_with_fallback` in `crates/bitnet-tokenizers/src/strategy.rs` returns `Ok(None)` instead of implementing actual tokenizer downloading from HuggingFace Hub.

## Environment

- **File**: `crates/bitnet-tokenizers/src/strategy.rs`
- **Function**: `TokenizerStrategyResolver::resolve_with_fallback`
- **Component**: Tokenizer discovery and downloading
- **Type**: Stub implementation replacement
- **MSRV**: Rust 1.90.0

## Current Stub Implementation

```rust
FallbackStrategy::SmartDownload => {
    // This would require download capability which is async
    // For now, just return None - actual implementation would download
    Ok(None)
}
```

## Proposed Solution

Implement actual HuggingFace Hub tokenizer downloading:

```rust
FallbackStrategy::SmartDownload => {
    match self.discovery.infer_download_source() {
        Ok(Some(download_info)) => {
            let downloaded_path = self.downloader.download_tokenizer(&download_info).await?;
            Ok(Some(TokenizerResolution::File(downloaded_path)))
        }
        Ok(None) => Ok(None),
        Err(e) => Err(e),
    }
}
```

### Complete Implementation

```rust
use tokio::fs;
use serde_json::Value;

pub struct SmartTokenizerDownloader {
    client: reqwest::Client,
    cache_dir: PathBuf,
}

impl SmartTokenizerDownloader {
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Self {
        Self {
            client: reqwest::Client::new(),
            cache_dir: cache_dir.as_ref().to_path_buf(),
        }
    }

    pub async fn download_tokenizer(&self, info: &DownloadInfo) -> Result<PathBuf> {
        // Create model-specific cache directory
        let model_cache_dir = self.cache_dir.join(&info.model_id);
        fs::create_dir_all(&model_cache_dir).await?;

        // Download tokenizer files
        let tokenizer_files = self.get_tokenizer_files(info).await?;

        for file_info in tokenizer_files {
            let local_path = model_cache_dir.join(&file_info.filename);

            if !local_path.exists() {
                self.download_file(&file_info.download_url, &local_path).await?;
            }
        }

        // Return path to main tokenizer file
        Ok(model_cache_dir.join("tokenizer.json"))
    }

    async fn get_tokenizer_files(&self, info: &DownloadInfo) -> Result<Vec<FileInfo>> {
        let repo_url = format!("https://huggingface.co/api/models/{}", info.model_id);
        let response: Value = self.client.get(&repo_url).send().await?.json().await?;

        let mut files = Vec::new();

        if let Some(siblings) = response["siblings"].as_array() {
            for sibling in siblings {
                if let Some(filename) = sibling["rfilename"].as_str() {
                    if Self::is_tokenizer_file(filename) {
                        files.push(FileInfo {
                            filename: filename.to_string(),
                            download_url: format!(
                                "https://huggingface.co/{}/resolve/main/{}",
                                info.model_id, filename
                            ),
                        });
                    }
                }
            }
        }

        Ok(files)
    }

    fn is_tokenizer_file(filename: &str) -> bool {
        matches!(filename,
            "tokenizer.json" |
            "tokenizer_config.json" |
            "vocab.json" |
            "merges.txt" |
            "special_tokens_map.json"
        )
    }

    async fn download_file(&self, url: &str, local_path: &Path) -> Result<()> {
        let response = self.client.get(url).send().await?;
        let bytes = response.bytes().await?;
        fs::write(local_path, &bytes).await?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct DownloadInfo {
    pub model_id: String,
    pub revision: Option<String>,
}

#[derive(Debug)]
struct FileInfo {
    filename: String,
    download_url: String,
}
```

## Implementation Plan

### Phase 1: Download Infrastructure (1 day)
- [ ] Implement HuggingFace Hub API client
- [ ] Add tokenizer file detection and downloading
- [ ] Create caching system for downloaded tokenizers
- [ ] Add error handling for network failures

### Phase 2: Integration (0.5 days)
- [ ] Integrate downloader with strategy resolver
- [ ] Update tokenizer discovery to infer download sources
- [ ] Add async support to resolution pipeline
- [ ] Test end-to-end tokenizer downloading

## Acceptance Criteria

- [ ] SmartDownload strategy downloads actual tokenizers
- [ ] Downloaded tokenizers cached locally
- [ ] Proper error handling for network issues
- [ ] Integration with existing tokenizer loading pipeline

## Labels

`tokenizer`, `download`, `huggingface-hub`, `networking`, `medium-priority`, `stub-removal`