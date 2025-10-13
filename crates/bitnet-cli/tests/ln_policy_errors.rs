//! Tests for LayerNorm policy error handling

use std::fs;
use tempfile::NamedTempFile;

#[test]
fn invalid_regex_in_policy_fails_nicely() {
    let yml = r#"
version: 1
rules:
  bad-regex:
    ln:
      - { pattern: "(", min: 0.5, max: 2.0 }  # invalid regex - unclosed paren
"#;
    let tmp = NamedTempFile::new().unwrap();
    fs::write(tmp.path(), yml).unwrap();

    let err = bitnet_cli::ln_rules::load_policy(tmp.path(), "bad-regex").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("invalid regex pattern") || msg.contains("regex"),
        "Expected error to mention invalid regex, got: {msg}"
    );
}

#[test]
fn missing_policy_key_fails_nicely() {
    let yml = r#"
version: 1
rules:
  some-key:
    ln:
      - { pattern: ".*", min: 0.5, max: 2.0 }
"#;
    let tmp = NamedTempFile::new().unwrap();
    fs::write(tmp.path(), yml).unwrap();

    let err = bitnet_cli::ln_rules::load_policy(tmp.path(), "missing-key").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("missing-key") || msg.contains("not found"),
        "Expected error to mention missing key, got: {msg}"
    );
}
