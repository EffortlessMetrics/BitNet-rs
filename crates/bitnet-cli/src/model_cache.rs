//! User-facing model cache management.

use anyhow::{Context, Result, anyhow};
use clap::{Args, Subcommand};
use futures::StreamExt;
use humansize::{DECIMAL, format_size};
use serde::Serialize;
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::process::Command;
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};
use tokio::io::AsyncWriteExt;

const DEFAULT_CACHE_RELATIVE: &[&str] = &["bitnet-rs", "models"];
const LOW_DISK_HEADROOM_BYTES: u64 = 1_073_741_824;
#[cfg(feature = "full-cli")]
pub(crate) const M4_SLM_RUNTIME_MODEL_ID: &str = "qwen2.5-0.5b-instruct-q8_0";

/// Manage supported local model artifacts.
#[derive(Debug, Args)]
pub struct ModelCommand {
    #[command(subcommand)]
    pub action: ModelAction,
}

#[derive(Debug, Subcommand)]
pub enum ModelAction {
    /// Fetch a supported model artifact into the local cache.
    Fetch {
        /// Supported model id, for example qwen2.5-0.5b-instruct-q8_0.
        id: String,

        /// Override cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Do not use the network; pass if an already verified artifact is cached.
        #[arg(long, default_value_t = false)]
        offline: bool,

        /// Re-download even when a verified artifact already exists.
        #[arg(long, default_value_t = false)]
        force: bool,

        /// Emit JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Verify a cached or explicit supported model artifact.
    Verify {
        /// Supported model id.
        id: String,

        /// Verify this file instead of the cached path for the model id.
        #[arg(long, value_name = "PATH")]
        path: Option<PathBuf>,

        /// Override cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Emit JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// List supported model artifacts and cache status.
    List {
        /// Override cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Emit JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Remove cached model artifacts.
    Prune {
        /// Supported model id to remove. Use --all to remove every supported artifact.
        id: Option<String>,

        /// Remove every supported cached artifact.
        #[arg(long, default_value_t = false)]
        all: bool,

        /// Show what would be removed without deleting files.
        #[arg(long, default_value_t = false)]
        dry_run: bool,

        /// Override cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Emit JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}

#[derive(Debug, Clone, Copy, Serialize)]
struct SupportedModel {
    id: &'static str,
    display_name: &'static str,
    repo: &'static str,
    revision: &'static str,
    filename: &'static str,
    url: &'static str,
    sha256: &'static str,
    bytes: u64,
    architecture: &'static str,
    quantization: &'static str,
    tokenizer_model: &'static str,
    tokenizer_pre: &'static str,
    chat_template: bool,
    apple_m4_cpu_neon_supported: bool,
    support_note: &'static str,
}

#[derive(Debug, Serialize)]
struct CacheStatus {
    model: SupportedModel,
    cache_path: PathBuf,
    metadata_path: PathBuf,
    present: bool,
    cached: bool,
    size_matches: bool,
    metadata_present: bool,
    verified: Option<bool>,
}

#[derive(Debug, Serialize)]
struct VerifyResult {
    id: String,
    path: PathBuf,
    expected_sha256: String,
    actual_sha256: Option<String>,
    expected_bytes: u64,
    actual_bytes: Option<u64>,
    passed: bool,
    model: SupportedModel,
}

#[cfg(feature = "full-cli")]
#[derive(Debug, Clone)]
pub(crate) struct VerifiedCachedModel {
    pub id: String,
    pub display_name: String,
    pub path: PathBuf,
    pub cache_root: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    pub architecture: String,
    pub quantization: String,
    pub tokenizer_model: String,
    pub tokenizer_pre: String,
    pub chat_template: bool,
    pub support_note: String,
}

#[derive(Debug, Serialize)]
struct PruneResult {
    id: String,
    path: PathBuf,
    existed: bool,
    removed: bool,
    dry_run: bool,
}

impl ModelCommand {
    pub async fn execute(self) -> Result<()> {
        match self.action {
            ModelAction::Fetch { id, cache_dir, offline, force, json } => {
                fetch_model(&id, cache_dir, offline, force, json).await
            }
            ModelAction::Verify { id, path, cache_dir, json } => {
                verify_model_command(&id, path, cache_dir, json)
            }
            ModelAction::List { cache_dir, json } => list_models(cache_dir, json),
            ModelAction::Prune { id, all, dry_run, cache_dir, json } => {
                prune_models(id, all, dry_run, cache_dir, json)
            }
        }
    }
}

#[cfg(feature = "full-cli")]
pub(crate) fn verified_apple_m4_slm_model(
    id: &str,
    cache_dir: Option<PathBuf>,
) -> Result<VerifiedCachedModel> {
    let model = supported_model(id)?;
    if !model.apple_m4_cpu_neon_supported {
        anyhow::bail!(
            "model `{}` is not supported for the Rust-native Apple M4 CPU/NEON SLM path: {}",
            model.id,
            model.support_note
        );
    }
    let cache_root = resolve_cache_root(cache_dir)?;
    let path = model_path(&cache_root, model);
    let result = verify_model(model, &path)?;
    if !result.passed {
        let actual = match (result.actual_bytes, result.actual_sha256.as_deref()) {
            (Some(bytes), Some(sha)) => format!("bytes={bytes}, sha256={sha}"),
            (Some(bytes), None) => format!("bytes={bytes}, sha256=<unavailable>"),
            _ => "missing".to_string(),
        };
        anyhow::bail!(
            "cached Apple M4 SLM model `{}` is not verified at {} ({actual}); run `bitnet model fetch {}` first",
            model.id,
            path.display(),
            model.id
        );
    }
    write_cache_metadata(&cache_root, model, &path, &result)?;
    Ok(VerifiedCachedModel {
        id: model.id.to_string(),
        display_name: model.display_name.to_string(),
        path,
        cache_root,
        sha256: model.sha256.to_string(),
        bytes: model.bytes,
        architecture: model.architecture.to_string(),
        quantization: model.quantization.to_string(),
        tokenizer_model: model.tokenizer_model.to_string(),
        tokenizer_pre: model.tokenizer_pre.to_string(),
        chat_template: model.chat_template,
        support_note: model.support_note.to_string(),
    })
}

#[cfg(feature = "full-cli")]
pub(crate) fn apple_m4_slm_cache_status_json(
    id: &str,
    cache_dir: Option<PathBuf>,
    verify: bool,
) -> Result<serde_json::Value> {
    let model = supported_model(id)?;
    let cache_root = resolve_cache_root(cache_dir)?;
    let status = cache_status(&cache_root, *model, verify)?;
    let ready = status.present
        && status.size_matches
        && status.metadata_present
        && status.verified.unwrap_or(true);
    Ok(serde_json::json!({
        "artifact_kind": "apple_m4_slm_model_cache_check",
        "id": model.id,
        "display_name": model.display_name,
        "cache_root": cache_root,
        "cache_path": status.cache_path,
        "metadata_path": status.metadata_path,
        "state": cache_state_label(&status),
        "ready": ready,
        "present": status.present,
        "size_matches": status.size_matches,
        "metadata_present": status.metadata_present,
        "verified": status.verified,
        "expected": {
            "repo": model.repo,
            "revision": model.revision,
            "filename": model.filename,
            "sha256": model.sha256,
            "bytes": model.bytes,
            "architecture": model.architecture,
            "quantization": model.quantization,
            "tokenizer_model": model.tokenizer_model,
            "tokenizer_pre": model.tokenizer_pre,
            "chat_template": model.chat_template,
        },
        "runtime_support": {
            "apple_m4_cpu_neon": model.apple_m4_cpu_neon_supported,
            "note": model.support_note,
        },
        "next_step": if ready {
            serde_json::Value::Null
        } else {
            serde_json::json!(format!("run `bitnet model fetch {}`", model.id))
        },
    }))
}

const SUPPORTED_MODELS: &[SupportedModel] = &[
    SupportedModel {
        id: "qwen2.5-0.5b-instruct-q8_0",
        display_name: "Qwen2.5 0.5B Instruct Q8_0",
        repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        revision: "9217f5db79a29953eb74d5343926648285ec7e67",
        filename: "qwen2.5-0.5b-instruct-q8_0.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q8_0.gguf",
        sha256: "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
        bytes: 675_710_816,
        architecture: "qwen2",
        quantization: "Q8_0",
        tokenizer_model: "gpt2",
        tokenizer_pre: "qwen2",
        chat_template: true,
        apple_m4_cpu_neon_supported: true,
        support_note: "Rust-native Apple M4 CPU/NEON SLM baseline artifact.",
    },
    SupportedModel {
        id: "qwen2.5-0.5b-instruct-q4_k_m",
        display_name: "Qwen2.5 0.5B Instruct Q4_K_M",
        repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        revision: "9217f5db79a29953eb74d5343926648285ec7e67",
        filename: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        sha256: "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
        bytes: 491_400_032,
        architecture: "qwen2",
        quantization: "Q4_K_M",
        tokenizer_model: "gpt2",
        tokenizer_pre: "qwen2",
        chat_template: true,
        apple_m4_cpu_neon_supported: false,
        support_note: "Reference-good and storage-preferred, but strict Rust execution remains unsupported.",
    },
];

async fn fetch_model(
    id: &str,
    cache_dir: Option<PathBuf>,
    offline: bool,
    force: bool,
    json: bool,
) -> Result<()> {
    let model = supported_model(id)?;
    let cache_root = resolve_cache_root(cache_dir)?;
    let path = model_path(&cache_root, model);

    if path.exists() && !force {
        let result = verify_model(model, &path)?;
        if result.passed {
            write_cache_metadata(&cache_root, model, &path, &result)?;
            return print_fetch_result("cached", &result, json);
        }
    }

    if bitnet_download::offline_enabled(offline) {
        anyhow::bail!("model `{}` is not verified in cache and offline mode is enabled", model.id);
    }

    warn_if_low_disk(&cache_root, model.bytes);
    fs::create_dir_all(model_dir(&cache_root, model))
        .with_context(|| format!("failed to create cache dir {}", cache_root.display()))?;

    let tmp_path = path.with_extension("gguf.part");
    let client = reqwest::Client::new();
    let response = client
        .get(model.url)
        .send()
        .await
        .with_context(|| format!("failed to request {}", model.url))?
        .error_for_status()
        .with_context(|| format!("download request failed for {}", model.url))?;

    let expected_len = response.content_length();
    let mut stream = response.bytes_stream();
    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .with_context(|| format!("failed to create {}", tmp_path.display()))?;
    let mut downloaded = 0u64;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("download failed for {}", model.url))?;
        downloaded += chunk.len() as u64;
        file.write_all(&chunk)
            .await
            .with_context(|| format!("failed to write {}", tmp_path.display()))?;
    }
    file.flush().await.with_context(|| format!("failed to flush {}", tmp_path.display()))?;
    drop(file);

    if let Err(err) = bitnet_download::validate_downloaded_len(downloaded, expected_len) {
        let _ = fs::remove_file(&tmp_path);
        return Err(err)
            .with_context(|| format!("download length mismatch for {}", model.filename));
    }
    if let Err(err) = bitnet_download::validate_downloaded_len(downloaded, Some(model.bytes)) {
        let _ = fs::remove_file(&tmp_path);
        return Err(err).with_context(|| {
            format!("downloaded size for {} did not match manifest", model.filename)
        });
    }

    let result = verify_model(model, &tmp_path)?;
    if !result.passed {
        let _ = fs::remove_file(&tmp_path);
        anyhow::bail!(
            "downloaded `{}` failed verification: expected sha256 {}, got {:?}",
            model.id,
            model.sha256,
            result.actual_sha256
        );
    }
    replace_cached_file(&tmp_path, &path)
        .with_context(|| format!("failed to move {} to {}", tmp_path.display(), path.display()))?;
    let result = verify_model(model, &path)?;
    write_cache_metadata(&cache_root, model, &path, &result)?;
    print_fetch_result("downloaded", &result, json)
}

fn verify_model_command(
    id: &str,
    path: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    json: bool,
) -> Result<()> {
    let model = supported_model(id)?;
    let cache_root = resolve_cache_root(cache_dir)?;
    let cache_path = model_path(&cache_root, model);
    let path = path.unwrap_or_else(|| cache_path.clone());
    let result = verify_model(model, &path)?;
    if result.passed && path == cache_path {
        write_cache_metadata(&cache_root, model, &path, &result)?;
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else if result.passed {
        println!("verified {} at {}", model.id, path.display());
    } else {
        println!("verification failed for {} at {}", model.id, path.display());
    }
    if result.passed { Ok(()) } else { anyhow::bail!("model `{}` failed verification", model.id) }
}

fn list_models(cache_dir: Option<PathBuf>, json: bool) -> Result<()> {
    let cache_root = resolve_cache_root(cache_dir)?;
    let statuses: Vec<_> = SUPPORTED_MODELS
        .iter()
        .map(|model| cache_status(&cache_root, *model, false))
        .collect::<Result<_>>()?;

    if json {
        println!("{}", serde_json::to_string_pretty(&statuses)?);
        return Ok(());
    }

    println!("Cache: {}", cache_root.display());
    println!("{:<34} {:<13} {:<12} {:<11} Artifact", "ID", "Cache", "Quant", "M4 CPU");
    println!("{}", "-".repeat(92));
    for status in statuses {
        let m4_cpu = if status.model.apple_m4_cpu_neon_supported { "supported" } else { "no" };
        let cache_state = cache_state_label(&status);
        println!(
            "{:<34} {:<13} {:<12} {:<11} {}",
            status.model.id, cache_state, status.model.quantization, m4_cpu, status.model.filename,
        );
    }
    Ok(())
}

fn prune_models(
    id: Option<String>,
    all: bool,
    dry_run: bool,
    cache_dir: Option<PathBuf>,
    json: bool,
) -> Result<()> {
    if all && id.is_some() {
        anyhow::bail!("pass either a model id or --all, not both");
    }
    if !all && id.is_none() {
        anyhow::bail!("pass a model id or --all");
    }

    let cache_root = resolve_cache_root(cache_dir)?;
    let models: Vec<_> = if all {
        SUPPORTED_MODELS.iter().collect()
    } else {
        vec![supported_model(id.as_deref().unwrap())?]
    };
    let mut results = Vec::new();

    for model in models {
        let path = model_dir(&cache_root, model);
        let existed = path.exists();
        let removed = if existed && !dry_run {
            fs::remove_dir_all(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
            true
        } else {
            false
        };
        results.push(PruneResult { id: model.id.to_string(), path, existed, removed, dry_run });
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        for result in &results {
            let action = if result.dry_run {
                "would remove"
            } else if result.removed {
                "removed"
            } else if result.existed {
                "kept"
            } else {
                "not cached"
            };
            println!("{action}: {} ({})", result.id, result.path.display());
        }
    }
    Ok(())
}

fn supported_model(id: &str) -> Result<&'static SupportedModel> {
    SUPPORTED_MODELS.iter().find(|model| model.id == id).ok_or_else(|| {
        let known = SUPPORTED_MODELS.iter().map(|model| model.id).collect::<Vec<_>>().join(", ");
        anyhow!("unsupported model `{id}`. Supported models: {known}")
    })
}

fn resolve_cache_root(cache_dir: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = cache_dir {
        return Ok(path);
    }
    if let Some(path) = std::env::var_os("BITNET_MODEL_CACHE_DIR") {
        return Ok(PathBuf::from(path));
    }
    let mut root = dirs::cache_dir().ok_or_else(|| {
        anyhow!(
            "could not resolve user cache directory; pass --cache-dir or set BITNET_MODEL_CACHE_DIR"
        )
    })?;
    for segment in DEFAULT_CACHE_RELATIVE {
        root.push(segment);
    }
    Ok(root)
}

fn model_dir(cache_root: &Path, model: &SupportedModel) -> PathBuf {
    cache_root.join(model.id)
}

fn model_path(cache_root: &Path, model: &SupportedModel) -> PathBuf {
    model_dir(cache_root, model).join(model.filename)
}

fn metadata_path(cache_root: &Path, model: &SupportedModel) -> PathBuf {
    model_dir(cache_root, model).join("bitnet-model-cache.json")
}

fn cache_status(cache_root: &Path, model: SupportedModel, verify: bool) -> Result<CacheStatus> {
    let path = model_path(cache_root, &model);
    let metadata = metadata_path(cache_root, &model);
    let present = path.exists();
    let size_matches = present
        && fs::metadata(&path).map(|metadata| metadata.len() == model.bytes).unwrap_or(false);
    let metadata_present = metadata.exists();
    let cached = present && size_matches && metadata_present;
    let verified = if present && verify { Some(verify_model(&model, &path)?.passed) } else { None };
    Ok(CacheStatus {
        model,
        cache_path: path,
        metadata_path: metadata.clone(),
        present,
        cached,
        size_matches,
        metadata_present,
        verified,
    })
}

fn cache_state_label(status: &CacheStatus) -> &'static str {
    if !status.present {
        "missing"
    } else if !status.size_matches {
        "invalid-size"
    } else if !status.metadata_present {
        "unverified"
    } else {
        "ready"
    }
}

fn verify_model(model: &SupportedModel, path: &Path) -> Result<VerifyResult> {
    let metadata = fs::metadata(path).ok();
    let actual_bytes = metadata.as_ref().map(fs::Metadata::len);
    let actual_sha256 = if metadata.is_some() { Some(compute_sha256(path)?) } else { None };
    let passed =
        actual_bytes == Some(model.bytes) && actual_sha256.as_deref() == Some(model.sha256);
    Ok(VerifyResult {
        id: model.id.to_string(),
        path: path.to_path_buf(),
        expected_sha256: model.sha256.to_string(),
        actual_sha256,
        expected_bytes: model.bytes,
        actual_bytes,
        passed,
        model: *model,
    })
}

fn compute_sha256(path: &Path) -> Result<String> {
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let n =
            file.read(&mut buffer).with_context(|| format!("failed to read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn write_cache_metadata(
    cache_root: &Path,
    model: &SupportedModel,
    path: &Path,
    verify: &VerifyResult,
) -> Result<()> {
    let payload = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_model_cache_entry",
        "id": model.id,
        "display_name": model.display_name,
        "repo": model.repo,
        "revision": model.revision,
        "filename": model.filename,
        "source_url": model.url,
        "path": path,
        "sha256": model.sha256,
        "bytes": model.bytes,
        "architecture": model.architecture,
        "quantization": model.quantization,
        "tokenizer": {
            "model": model.tokenizer_model,
            "pre_tokenizer": model.tokenizer_pre,
            "chat_template_present": model.chat_template,
        },
        "runtime_support": {
            "apple_m4_cpu_neon": model.apple_m4_cpu_neon_supported,
            "note": model.support_note,
        },
        "verification": verify,
        "verified_at": chrono::Utc::now().to_rfc3339(),
    });
    let metadata = metadata_path(cache_root, model);
    fs::create_dir_all(metadata.parent().unwrap_or(cache_root))?;
    let bytes = serde_json::to_vec_pretty(&payload)?;
    bitnet_download::atomic_write(&metadata, &bytes)
        .with_context(|| format!("failed to write {}", metadata.display()))?;
    Ok(())
}

fn replace_cached_file(src: &Path, dst: &Path) -> Result<()> {
    if dst.exists() {
        fs::remove_file(dst)
            .with_context(|| format!("failed to remove old cache file {}", dst.display()))?;
    }
    fs::rename(src, dst)
        .with_context(|| format!("failed to rename {} to {}", src.display(), dst.display()))?;
    Ok(())
}

fn print_fetch_result(status: &str, verify: &VerifyResult, json: bool) -> Result<()> {
    let payload = serde_json::json!({
        "status": status,
        "id": verify.id,
        "path": verify.path,
        "sha256": verify.actual_sha256,
        "bytes": verify.actual_bytes,
        "verified": verify.passed,
        "apple_m4_cpu_neon_supported": verify.model.apple_m4_cpu_neon_supported,
        "support_note": verify.model.support_note,
    });
    if json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!(
            "{status}: {} at {} ({}, verified={})",
            verify.id,
            verify.path.display(),
            verify.actual_bytes.map(|bytes| format_size(bytes, DECIMAL)).unwrap_or_default(),
            verify.passed,
        );
        if !verify.model.apple_m4_cpu_neon_supported {
            println!("note: {}", verify.model.support_note);
        }
    }
    Ok(())
}

fn warn_if_low_disk(cache_root: &Path, expected_bytes: u64) {
    let parent =
        cache_root.ancestors().find(|path| path.exists()).unwrap_or_else(|| Path::new("."));
    let Some(available) = available_bytes(parent) else {
        return;
    };
    let recommended = expected_bytes.saturating_mul(2).saturating_add(LOW_DISK_HEADROOM_BYTES);
    if available < recommended {
        eprintln!(
            "warning: low disk headroom for model fetch: available={}, recommended>={}",
            format_size(available, DECIMAL),
            format_size(recommended, DECIMAL)
        );
    }
}

fn available_bytes(path: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        let output = Command::new("df").arg("-k").arg(path).output().ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout.lines().nth(1)?;
        let available_kib = line.split_whitespace().nth(3)?.parse::<u64>().ok()?;
        Some(available_kib.saturating_mul(1024))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_manifest_contains_m4_runtime_artifact() {
        let model = supported_model("qwen2.5-0.5b-instruct-q8_0").unwrap();
        assert!(model.apple_m4_cpu_neon_supported);
        assert_eq!(model.sha256.len(), 64);
        assert_eq!(model.bytes, 675_710_816);
        assert_eq!(model.tokenizer_pre, "qwen2");
    }

    #[test]
    fn supported_manifest_keeps_q4_reference_boundary() {
        let model = supported_model("qwen2.5-0.5b-instruct-q4_k_m").unwrap();
        assert!(!model.apple_m4_cpu_neon_supported);
        assert!(model.support_note.contains("unsupported"));
    }

    #[test]
    fn cache_paths_are_under_model_id() {
        let root = PathBuf::from("/tmp/bitnet-cache");
        let model = supported_model("qwen2.5-0.5b-instruct-q8_0").unwrap();
        let path = model_path(&root, model);
        assert!(path.ends_with("qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf"));
    }
}
