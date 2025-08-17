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