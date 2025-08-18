#![cfg(feature = "crossval")]
//! Integration tests for BitNet workspace

mod cross_validation;

#[cfg(test)]
mod tests {
    #[test]
    fn test_workspace_builds() {
        // This test just ensures the workspace compiles
        assert!(true);
    }
}
