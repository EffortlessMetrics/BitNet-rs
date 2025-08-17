use assert_cmd::Command;

#[test]
fn help_works() {
    Command::cargo_bin("bitnet").unwrap()
        .arg("--help")
        .assert()
        .success();
}

#[test]
fn version_works() {
    Command::cargo_bin("bitnet").unwrap()
        .arg("--version")
        .assert()
        .success();
}

#[test]
fn help_mentions_core_subcommands() {
    let out = Command::cargo_bin("bitnet")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8(out).unwrap();

    // Looser contract: presence of key verbs without snapshot churn.
    for needle in ["serve", "infer", "--model", "--config"] {
        assert!(s.contains(needle), "help missing `{needle}`");
    }
}

#[cfg(feature = "full-cli")]
#[test]
fn benchmark_help_works() {
    Command::cargo_bin("bitnet").unwrap()
        .args(["benchmark", "--help"])
        .assert()
        .success();
}

#[test]
fn invalid_command_fails() {
    Command::cargo_bin("bitnet").unwrap()
        .arg("nonexistent-command")
        .assert()
        .failure();
}