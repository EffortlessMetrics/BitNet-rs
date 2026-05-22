//! Reusable API versioning primitives for `BitNet` services.

mod api_version;
mod negotiation;
mod path_utils;

pub use api_version::ApiVersion;
pub use negotiation::{NegotiationResult, VersionRange};
pub use path_utils::{extract_version_from_path, version_header};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_creation() {
        let v = ApiVersion::new(1, 2);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(format!("{v}"), "v1.2");
    }

    #[test]
    fn test_version_parse() {
        assert_eq!(ApiVersion::parse("v1.0"), Some(ApiVersion::new(1, 0)));
        assert_eq!(ApiVersion::parse("2.3"), Some(ApiVersion::new(2, 3)));
        assert_eq!(ApiVersion::parse("v1"), Some(ApiVersion::new(1, 0)));
    }

    #[test]
    fn test_version_compatibility() {
        let v1_2 = ApiVersion::new(1, 2);
        let v1_0 = ApiVersion::new(1, 0);
        assert!(v1_2.is_compatible_with(&v1_0));
        assert!(!v1_0.is_compatible_with(&v1_2));
    }

    #[test]
    fn test_version_ordering() {
        assert!(ApiVersion::new(1, 0) < ApiVersion::new(1, 1));
        assert!(ApiVersion::new(1, 1) < ApiVersion::new(2, 0));
    }

    #[test]
    fn test_negotiate_accepted() {
        let range = VersionRange::default_range();
        let result = range.negotiate(&ApiVersion::new(1, 0));
        assert_eq!(result, NegotiationResult::Accepted(ApiVersion::new(1, 0)));
    }

    #[test]
    fn test_negotiate_rejected() {
        let range = VersionRange::default_range();
        let result = range.negotiate(&ApiVersion::new(2, 0));
        assert!(matches!(result, NegotiationResult::Rejected { .. }));
    }

    #[test]
    fn test_negotiate_selects_best_compatible_minor() {
        let range = VersionRange::new(
            vec![ApiVersion::new(1, 0), ApiVersion::new(1, 1), ApiVersion::new(1, 2)],
            ApiVersion::new(1, 2),
            ApiVersion::new(1, 0),
        );
        let result = range.negotiate(&ApiVersion::new(1, 3));
        assert_eq!(result, NegotiationResult::Accepted(ApiVersion::new(1, 2)));
    }

    #[test]
    fn test_negotiate_marks_deprecated_best_match() {
        let range = VersionRange::new(
            vec![ApiVersion::new(1, 0), ApiVersion::new(1, 1)],
            ApiVersion::new(1, 1),
            ApiVersion::new(1, 1),
        );
        let result = range.negotiate(&ApiVersion::new(1, 0));
        assert_eq!(
            result,
            NegotiationResult::Deprecated {
                accepted: ApiVersion::new(1, 0),
                sunset_version: ApiVersion::new(1, 1),
            }
        );
    }

    #[test]
    fn test_extract_version_from_path() {
        assert_eq!(extract_version_from_path("/v1/chat"), Some(ApiVersion::new(1, 0)));
        assert_eq!(extract_version_from_path("/api/v2.1/models"), Some(ApiVersion::new(2, 1)));
        assert_eq!(extract_version_from_path("/health"), None);
    }

    #[test]
    fn test_version_header() {
        let h = version_header(&ApiVersion::new(1, 0));
        assert_eq!(h, "application/json; version=v1.0");
    }

    #[test]
    fn test_is_deprecated() {
        let v = ApiVersion::new(0, 9);
        let min = ApiVersion::new(1, 0);
        assert!(v.is_deprecated(&min));
        assert!(!min.is_deprecated(&min));
    }

    #[test]
    fn test_is_supported() {
        let range = VersionRange::default_range();
        assert!(range.is_supported(&ApiVersion::new(1, 0)));
        assert!(!range.is_supported(&ApiVersion::new(2, 0)));
    }

    #[test]
    fn test_parse_invalid() {
        assert!(ApiVersion::parse("abc").is_none());
        assert!(ApiVersion::parse("").is_none());
    }

    #[test]
    fn test_current_version() {
        assert_eq!(ApiVersion::CURRENT, ApiVersion::new(1, 0));
    }
}
