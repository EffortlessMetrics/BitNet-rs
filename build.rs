use vergen::EmitBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate build-time constants
    EmitBuilder::builder()
        .build_timestamp()
        .git_sha(false)
        .rustc_semver()
        .cargo_target_triple()
        .emit()?;

    Ok(())
}
