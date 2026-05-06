use super::*;

pub(crate) fn cmd_build_cpp_static(root: &Path, cpp_dir: Option<&Path>) -> Result<()> {
    let default_cpp_dir = env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.to_path_buf())
        .join(".cache")
        .join("bitnet_cpp");
    let cpp_dir = cpp_dir.unwrap_or(default_cpp_dir.as_path());

    if !cpp_dir.is_dir() {
        bail!("C++ checkout missing: {}. Run: cargo xtask fetch-cpp", cpp_dir.display());
    }

    println!("=== Building BitNet C++ with Static Linking ===");
    println!();
    run_stream(
        cpp_dir,
        "cmake",
        &[
            "-B",
            "build",
            "-S",
            ".",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DLLAMA_STATIC=ON",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DGGML_NATIVE=ON",
            "-DLLAMA_CURL=OFF",
        ],
        &[],
    )?;
    run_stream(cpp_dir, "cmake", &["--build", "build", "-j"], &[])?;

    println!();
    println!("✅ Static build complete!");
    println!();
    println!("Binaries available at:");
    let bin_dir = cpp_dir.join("build").join("bin");
    let mut found = false;
    if let Ok(entries) = fs::read_dir(&bin_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if path.is_file() && name.starts_with("llama") {
                found = true;
                println!("  {}", path.display());
            }
        }
    }
    if !found {
        println!("No llama binaries found");
    }

    println!();
    println!("Test with:");
    println!("  {}/build/bin/llama-cli -m <model.gguf> -p \"test\" -n 1 -ngl 0", cpp_dir.display());
    Ok(())
}

pub(crate) fn cmd_docker_build(root: &Path, target: &str) -> Result<()> {
    let target = target.to_lowercase();
    if target != "cpu" && target != "gpu" && target != "all" {
        bail!("unknown target '{target}'. expected cpu, gpu, or all");
    }

    let git_sha = String::from_utf8_lossy(
        &run_capture(root, "git", &["rev-parse", "HEAD"], &[], true)?.stdout,
    )
    .trim()
    .to_string();
    let git_sha = if git_sha.is_empty() { "unknown".to_string() } else { git_sha };

    let git_branch = String::from_utf8_lossy(
        &run_capture(root, "git", &["rev-parse", "--abbrev-ref", "HEAD"], &[], true)?.stdout,
    )
    .trim()
    .to_string();
    let git_branch = if git_branch.is_empty() { "unknown".to_string() } else { git_branch };

    let git_describe = String::from_utf8_lossy(
        &run_capture(root, "git", &["describe", "--tags", "--always"], &[], true)?.stdout,
    )
    .trim()
    .to_string();
    let git_describe = if git_describe.is_empty() {
        String::from_utf8_lossy(
            &run_capture(root, "git", &["rev-parse", "--short", "HEAD"], &[], true)?.stdout,
        )
        .trim()
        .to_string()
    } else {
        git_describe
    };
    let git_describe = if git_describe.is_empty() { "unknown".to_string() } else { git_describe };

    println!("Building BitNet-rs Docker images with Git metadata:");
    println!("  SHA: {git_sha}");
    println!("  Branch: {git_branch}");
    println!("  Describe: {git_describe}");
    println!();

    let build_image = |variant: &str, runtime: &str| -> Result<()> {
        println!("Building {variant} image...");
        run_stream(
            root,
            "docker",
            &[
                "build",
                "--build-arg",
                &format!("VCS_REF={git_sha}"),
                "--build-arg",
                &format!("VCS_BRANCH={git_branch}"),
                "--build-arg",
                &format!("VCS_DESCRIBE={git_describe}"),
                "--build-arg",
                &format!("FEATURES={variant}"),
                "--target",
                runtime,
                "-t",
                &format!("bitnet-rs:{variant}"),
                "-t",
                &format!("bitnet-rs:{variant}-{git_sha}"),
                ".",
            ],
            &[],
        )?;
        Ok(())
    };

    match target.as_str() {
        "cpu" => build_image("cpu", "runtime")?,
        "gpu" => build_image("gpu", "runtime-gpu")?,
        "all" => {
            build_image("cpu", "runtime")?;
            build_image("gpu", "runtime-gpu")?;
        }
        _ => unreachable!(),
    }

    println!();
    println!("Build complete! Images tagged with:");
    println!("  - bitnet-rs:{target} (latest)");
    println!("  - bitnet-rs:{target}-{git_sha} (versioned)");
    println!();
    println!("To run with docker-compose:");
    println!(
        "  GIT_SHA={git_sha} GIT_BRANCH={git_branch} GIT_DESCRIBE={git_describe} docker-compose up bitnet-{target}"
    );
    Ok(())
}
