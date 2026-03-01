//! Shared, dependency-light primitives for download mechanics.
//! Focused on URL formation and HTTP resume safety decisions.

/// Builds a canonical Hugging Face `resolve/main` URL for a repository file.
#[must_use]
pub fn huggingface_resolve_main_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}

/// Returns true when a `Content-Range` value aligns with a requested resume offset.
#[must_use]
pub fn has_aligned_content_range(content_range: Option<&str>, start: u64) -> bool {
    content_range.map(|v| v.starts_with(&format!("bytes {start}-"))).unwrap_or(false)
}

/// Decides whether a resumed download must restart from zero.
///
/// `status_code` is an HTTP response code (e.g. 200, 206).
#[must_use]
pub fn should_restart_resume(
    requested_start: u64,
    status_code: u16,
    content_range: Option<&str>,
) -> bool {
    if requested_start == 0 {
        return false;
    }

    match status_code {
        // Server ignored Range request.
        200 => true,
        // Partial content must include aligned Content-Range.
        206 => !has_aligned_content_range(content_range, requested_start),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_hf_main_url() {
        let url = huggingface_resolve_main_url("meta-llama/Llama-3", "tokenizer.json");
        assert_eq!(url, "https://huggingface.co/meta-llama/Llama-3/resolve/main/tokenizer.json");
    }

    #[test]
    fn aligned_content_range() {
        assert!(has_aligned_content_range(Some("bytes 1024-4095/4096"), 1024));
        assert!(!has_aligned_content_range(Some("bytes 0-4095/4096"), 1024));
        assert!(!has_aligned_content_range(None, 1024));
    }

    #[test]
    fn restart_logic() {
        assert!(!should_restart_resume(0, 200, None));
        assert!(should_restart_resume(2048, 200, None));
        assert!(!should_restart_resume(2048, 206, Some("bytes 2048-4095/4096")));
        assert!(should_restart_resume(2048, 206, Some("bytes 0-4095/4096")));
    }
}
