#[test]
#[cfg(feature = "cpp-probe")]
fn cpp_binary_reports_version() -> anyhow::Result<()> {
    let Some(bin) = std::env::var_os("BITNET_CPP_BIN") else {
        eprintln!("skipped: set BITNET_CPP_BIN=/path/to/bitnet-cpp");
        return Ok(());
    };
    let out = std::process::Command::new(bin).arg("--version").output()?;
    assert!(out.status.success(), "cpp binary --version failed");
    Ok(())
}
