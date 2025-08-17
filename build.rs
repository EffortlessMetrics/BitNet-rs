fn main() {
    // Generate build-time constants, gracefully handling missing git
    if let Err(e) = vergen::EmitBuilder::builder()
        .build_timestamp()
        .git_sha(false)
        .rustc_semver()
        .cargo_target_triple()
        .emit()
    {
        println!("cargo:warning=vergen emit skipped: {}", e);
    }
}
