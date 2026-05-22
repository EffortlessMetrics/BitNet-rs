use std::path::Path;

/// Qwen trace CLI switches that are materialized as process environment.
pub(crate) struct QwenTraceEnv<'a> {
    pub(crate) jsonl_path: Option<&'a Path>,
    pub(crate) layer: Option<usize>,
    pub(crate) full_prompt: bool,
    pub(crate) prompt_ids: Option<&'a str>,
}

impl QwenTraceEnv<'_> {
    pub(crate) fn apply(&self) {
        if let Some(path) = self.jsonl_path {
            unsafe {
                std::env::set_var("BITNET_QWEN_TRACE_JSONL", path);
            }
        }
        if let Some(layer) = self.layer {
            unsafe {
                std::env::set_var("BITNET_QWEN_TRACE_LAYER", layer.to_string());
            }
        }
        if self.full_prompt {
            unsafe {
                std::env::set_var("BITNET_QWEN_TRACE_FULL_PROMPT", "1");
            }
        }
        if let Some(prompt_ids) = self.prompt_ids {
            unsafe {
                std::env::set_var("BITNET_QWEN_TRACE_PROMPT_IDS", prompt_ids);
            }
        }
    }
}

/// Applies reproducibility settings requested by `--deterministic`.
pub(crate) fn apply_deterministic_env(deterministic: bool, threads: usize) {
    if !deterministic {
        return;
    }

    unsafe {
        std::env::set_var("BITNET_DETERMINISTIC", "1");
        std::env::set_var("RAYON_NUM_THREADS", "1");
        if threads > 0 {
            std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
        }
    }
}

/// Applies fail-fast loader settings requested by `--strict-loader`.
pub(crate) fn apply_strict_loader_env(strict_loader: bool) {
    if !strict_loader {
        return;
    }

    unsafe {
        std::env::set_var("BITNET_DISABLE_MINIMAL_LOADER", "1");
        std::env::set_var("BITNET_STRICT_MODE", "1");
    }
}

/// Persists the resolved backend identity for downstream kernels and receipts.
pub(crate) fn apply_backend_identity_env(
    requested_backend: &str,
    selected_backend: &str,
    runtime_api: &str,
    strict_cuda_backend_selected: bool,
    strict_a770_opencl_backend_selected: bool,
) {
    unsafe {
        std::env::set_var("BITNET_REQUESTED_BACKEND", requested_backend);
        std::env::set_var("BITNET_SELECTED_BACKEND", selected_backend);
        std::env::set_var("BITNET_RUNTIME_API", runtime_api);
        if strict_cuda_backend_selected {
            std::env::set_var("BITNET_STRICT_CUDA_BACKEND", "1");
        } else {
            std::env::remove_var("BITNET_STRICT_CUDA_BACKEND");
        }
        if strict_a770_opencl_backend_selected {
            std::env::set_var("BITNET_STRICT_A770_OPENCL_BACKEND", "1");
        } else {
            std::env::remove_var("BITNET_STRICT_A770_OPENCL_BACKEND");
        }
    }
}
