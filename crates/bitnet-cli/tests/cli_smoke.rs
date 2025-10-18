use assert_cmd::Command;

#[test]
fn help_works() {
    Command::cargo_bin("bitnet").unwrap().arg("--help").assert().success();
}

#[test]
fn version_works() {
    Command::cargo_bin("bitnet").unwrap().arg("--version").assert().success();
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

    // Check always-available commands (not gated behind full-cli)
    for needle in ["score", "--model", "--config"] {
        assert!(s.contains(needle), "help missing `{needle}`");
    }
}

#[cfg(feature = "full-cli")]
#[test]
fn help_mentions_full_cli_subcommands() {
    let out = Command::cargo_bin("bitnet")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8(out).unwrap();

    // Check full-cli gated commands
    for needle in ["serve", "infer", "chat", "inspect"] {
        assert!(s.contains(needle), "help missing full-cli command `{needle}`");
    }
}

#[cfg(feature = "full-cli")]
#[test]
fn benchmark_help_works() {
    Command::cargo_bin("bitnet").unwrap().args(["benchmark", "--help"]).assert().success();
}

#[test]
fn score_help_works() {
    Command::cargo_bin("bitnet").unwrap().args(["score", "--help"]).assert().success();
}

#[test]
fn score_command_validates_args() {
    // Score command should fail gracefully with missing required args
    Command::cargo_bin("bitnet").unwrap().args(["score"]).assert().failure();

    // Score command should show error for missing model
    Command::cargo_bin("bitnet")
        .unwrap()
        .args(["score", "--file", "/nonexistent"])
        .assert()
        .failure();
}

#[test]
fn invalid_command_fails() {
    Command::cargo_bin("bitnet").unwrap().arg("nonexistent-command").assert().failure();
}
