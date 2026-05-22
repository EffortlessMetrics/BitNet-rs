use anyhow::Result;

/// Backend resolution state needed by generation before model loading begins.
pub(crate) struct GenerationBackendSetup {
    pub(crate) identity: crate::RunBackendIdentity,
    pub(crate) strict_backend: bool,
    pub(crate) strict_cuda_backend_selected: bool,
    pub(crate) strict_a770_opencl_backend_selected: bool,
    pub(crate) cuda_memory_before_bytes: Option<u64>,
}

/// Resolves the effective backend, prepares strict CUDA runtime visibility, and
/// persists the backend identity for lower-level kernels and receipts.
pub(crate) fn prepare_generation_backend(
    requested_backend_label: &str,
    strict_loader: bool,
) -> Result<GenerationBackendSetup> {
    let strict_backend = strict_loader
        || std::env::var("BITNET_STRICT_MODE")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);

    crate::answer_corpus_child_phase(
        "backend_select_start",
        serde_json::json!({
            "requested_backend": requested_backend_label,
            "strict_backend": strict_backend,
        }),
    );
    let identity = crate::resolve_run_backend_identity(requested_backend_label, strict_backend)?;
    crate::answer_corpus_child_phase(
        "backend_select_complete",
        serde_json::json!({
            "requested_backend": identity.requested_backend.as_str(),
            "selected_backend": identity.selected_backend.as_str(),
            "runtime_api": identity.runtime_api.as_str(),
            "fallback_used": identity.fallback_used,
            "fallback_reason": identity.fallback_reason.as_deref(),
        }),
    );

    bitnet_qk256_dispatch::reset_qk256_dispatch_coverage();
    let strict_cuda_backend_selected = strict_backend
        && identity.selected_backend.as_str() == "nvidia-rtx-5070-ti-cuda"
        && identity.runtime_api.as_str() == "cuda"
        && !identity.fallback_used;
    let strict_a770_opencl_backend_selected = strict_backend
        && crate::is_a770_opencl_backend_label(identity.selected_backend.as_str())
        && identity.runtime_api.as_str() == "opencl"
        && !identity.fallback_used;

    if strict_cuda_backend_selected {
        crate::answer_corpus_child_phase(
            "cuda_runtime_libraries_start",
            serde_json::json!({
                "selected_backend": identity.selected_backend.as_str(),
                "runtime_api": identity.runtime_api.as_str(),
            }),
        );
        let cuda_bin = crate::ensure_strict_cuda_runtime_libraries_visible()?;
        crate::answer_corpus_child_phase(
            "cuda_runtime_libraries_complete",
            serde_json::json!({
                "added_cuda_toolkit_bin": cuda_bin.as_ref().map(|path| path.display().to_string()),
            }),
        );
        if let Some(cuda_bin) = cuda_bin {
            tracing::debug!(
                "added CUDA Toolkit bin directory to process PATH for strict CUDA run: {}",
                cuda_bin.display()
            );
        }
    }

    let cuda_memory_before_bytes = strict_cuda_backend_selected
        .then(|| crate::nvidia_smi_memory_used_bytes(Some(0)))
        .flatten();

    super::environment::apply_backend_identity_env(
        identity.requested_backend.as_str(),
        identity.selected_backend.as_str(),
        identity.runtime_api.as_str(),
        strict_cuda_backend_selected,
        strict_a770_opencl_backend_selected,
    );

    Ok(GenerationBackendSetup {
        identity,
        strict_backend,
        strict_cuda_backend_selected,
        strict_a770_opencl_backend_selected,
        cuda_memory_before_bytes,
    })
}
