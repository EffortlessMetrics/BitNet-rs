use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;
use serde_json::json;
use std::fs;

#[test]
fn ripr_annotations_emits_legacy_comment_annotations() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
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
        }))?,
    )?;

    let mut cmd = cargo_bin_cmd!("xtask");
    cmd.arg("ripr-annotations").arg("--path").arg(&comments);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains(
            "::warning file=src/lib.rs,line=42,title=RIPR%2C escaped::mutation risk: 50%25%0Aadd an assertion",
        ));

    Ok(())
}

#[test]
fn ripr_annotations_emits_finding_annotations() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
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
        }))?,
    )?;

    let mut cmd = cargo_bin_cmd!("xtask");
    cmd.arg("ripr-annotations").arg("--path").arg(&comments);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("::notice file=crates/bitnet/src/lib.rs,line=7,title=RIPR missing_assertion::Assert the boundary condition. | Expression: value > 0 | Confidence: 0.87"));

    Ok(())
}

#[test]
fn ripr_annotations_missing_file_is_noop() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let missing = temp.path().join("missing.json");

    let mut cmd = cargo_bin_cmd!("xtask");
    cmd.arg("ripr-annotations").arg("--path").arg(&missing);

    cmd.assert().success().stdout(predicate::str::is_empty());

    Ok(())
}
