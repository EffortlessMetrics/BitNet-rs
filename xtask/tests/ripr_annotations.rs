use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;
use serde_json::json;
use std::fs;

#[test]
fn ripr_annotations_emits_legacy_comment_annotations() {
    let temp = tempfile::tempdir().unwrap();
    let comments = temp.path().join("comments.json");
    fs::write(
        &comments,
        serde_json::to_string(&json!({
            "comments": [
                {
                    "path": "src/lib.rs",
                    "line": 42,
                    "title": "RIPR, escaped",
                    "body": "mutation risk: 50%\nadd an assertion"
                }
            ]
        }))
        .unwrap(),
    )
    .unwrap();

    let mut cmd = cargo_bin_cmd!("xtask");
    cmd.args(["ripr-annotations", "--path", comments.to_str().unwrap()]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains(
            "::warning file=src/lib.rs,line=42,title=RIPR%2C escaped::mutation risk: 50%25%0Aadd an assertion",
        ));
}

#[test]
fn ripr_annotations_emits_finding_annotations() {
    let temp = tempfile::tempdir().unwrap();
    let comments = temp.path().join("comments.json");
    fs::write(
        &comments,
        serde_json::to_string(&json!({
            "findings": [
                {
                    "classification": "missing_assertion",
                    "severity": "note",
                    "confidence": 0.87,
                    "suggested_next_action": "Assert the boundary condition.",
                    "probe": {
                        "file": "crates/bitnet/src/lib.rs",
                        "line": "7",
                        "expression": "value > 0"
                    }
                }
            ]
        }))
        .unwrap(),
    )
    .unwrap();

    let mut cmd = cargo_bin_cmd!("xtask");
    cmd.args(["ripr-annotations", "--path", comments.to_str().unwrap()]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("::notice file=crates/bitnet/src/lib.rs,line=7,title=RIPR missing_assertion::Assert the boundary condition. | Expression: value > 0 | Confidence: 0.87"));
}

#[test]
fn ripr_annotations_missing_file_is_noop() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("missing.json");

    let mut cmd = cargo_bin_cmd!("xtask");
    cmd.args(["ripr-annotations", "--path", missing.to_str().unwrap()]);

    cmd.assert().success().stdout(predicate::str::is_empty());
}
