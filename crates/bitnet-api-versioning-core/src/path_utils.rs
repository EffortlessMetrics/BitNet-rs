use crate::api_version::ApiVersion;

/// Extract API version from a URL path prefix like "/v1/..." or "/api/v1.0/...".
#[must_use]
pub fn extract_version_from_path(path: &str) -> Option<ApiVersion> {
    for segment in path.split('/') {
        if let Some(version) = ApiVersion::parse(segment)
            && version.major > 0
        {
            return Some(version);
        }
    }
    None
}

/// Format a version header value.
#[must_use]
pub fn version_header(version: &ApiVersion) -> String {
    format!("application/json; version={version}")
}
