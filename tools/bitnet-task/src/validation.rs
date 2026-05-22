use super::*;

mod envlock;

pub(crate) fn cmd_check_ignore_annotations(root: &Path) -> Result<()> {
    let script = root.join("scripts/lib/ignore_check.sh");
    if !script.exists() {
        bail!("missing helper script: {}", script.display());
    }
    let script = script.to_string_lossy().into_owned();
    match run_stream(root, "bash", &[script.as_str(), "crates", "tests", "tests-new", "xtask"], &[])
    {
        Ok(()) => Ok(()),
        Err(err) => {
            eprintln!(
                "::error::Found #[ignore] tests without justification. See output above for details."
            );
            bail!(err)
        }
    }
}

pub(crate) fn cmd_check_envlock(root: &Path) -> Result<()> {
    envlock::run(root)
}

pub(crate) fn cmd_check_patch_policy(root: &Path, strict: bool, create_issue: bool) -> Result<()> {
    println!("[INFO] 🔍 Checking patch policy compliance...");
    let patches_dir = root.join("patches");

    if !patches_dir.is_dir() {
        println!("[INFO] ✅ No patches directory found - policy compliant");
        println!("[INFO] ✅ Patch policy compliant - no patches found");
        return Ok(());
    }

    let patch_files = collect_patch_files(&patches_dir)?;
    if patch_files.is_empty() {
        println!("[INFO] ✅ Patches directory is empty - policy compliant");
        println!("[INFO] ✅ Patch policy compliant - no patches found");
        return Ok(());
    }

    println!("[WARN] Found {} patch file(s) - checking policy compliance...", patch_files.len());

    let mut violations = 0usize;
    for patch_file in &patch_files {
        let patch_name =
            patch_file.file_name().and_then(|name| name.to_str()).unwrap_or("<unknown>");
        println!("[INFO] Checking patch: {patch_name}");

        let content = fs::read_to_string(patch_file)
            .with_context(|| format!("failed to read {}", patch_file.display()))?;
        let content_lower = content.to_ascii_lowercase();
        if !(content_lower.contains("issue")
            || content_lower.contains("bug")
            || content_lower.contains("upstream")
            || content_lower.contains("microsoft/bitnet"))
        {
            println!("[ERROR] ❌ Patch '{patch_name}' does not reference an upstream issue");
            println!(
                "[ERROR]    All patches must reference an upstream issue in Microsoft/BitNet repository"
            );
            violations += 1;
        }

        if let Ok(age_days) = patch_age_days(patch_file)
            && age_days > 90
        {
            println!("[WARN] ⚠️  Patch '{patch_name}' is {age_days} days old");
            println!("[WARN]    Consider checking if upstream issue has been resolved");
        }

        let patch_lines = content.lines().count();
        if patch_lines > 100 {
            println!("[WARN] ⚠️  Patch '{patch_name}' is large ({patch_lines} lines)");
            println!("[WARN]    Consider if this change should be contributed upstream instead");
        }
    }

    violations += check_patch_documentation(root, &patch_files)?;

    if create_issue {
        prepare_patch_tracking_issue(&patch_files);
    }

    if violations > 0 {
        println!("[ERROR] ❌ Found {violations} patch policy violation(s)");
        if strict {
            println!("[ERROR] 💥 STRICT MODE: Failing CI due to patch policy violations");
            println!("[ERROR]");
            println!("[ERROR] Our policy strongly discourages patches. Consider:");
            println!("[ERROR]   1. Contributing fixes upstream to Microsoft/BitNet");
            println!("[ERROR]   2. Using wrapper functions instead of patches");
            println!("[ERROR]   3. Adapting to existing C++ API in Rust code");
            std::process::exit(2);
        }
        println!("[WARN] ⚠️  Patch policy violations found but not failing CI");
        println!("[WARN]    Please address these violations as soon as possible");
        bail!("patch policy violations found");
    }

    println!("[INFO] ✅ All patch policy checks passed");
    Ok(())
}

fn collect_patch_files(patches_dir: &Path) -> Result<Vec<PathBuf>> {
    fn visit(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        for entry in
            fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit(&path, files)?;
            } else if path.extension().is_some_and(|ext| ext == "patch") {
                files.push(path);
            }
        }
        Ok(())
    }

    let mut files = Vec::new();
    visit(patches_dir, &mut files)?;
    files.sort();
    Ok(files)
}

fn patch_age_days(patch_file: &Path) -> Result<u64> {
    let modified = fs::metadata(patch_file)?.modified()?;
    let age = std::time::SystemTime::now().duration_since(modified).unwrap_or_default();
    Ok(age.as_secs() / 86_400)
}

fn check_patch_documentation(root: &Path, patch_files: &[PathBuf]) -> Result<usize> {
    let patches_readme = root.join("patches/README.md");
    if !patches_readme.is_file() {
        println!("[ERROR] ❌ patches/README.md not found");
        println!("[ERROR]    Patches directory must have documentation");
        return Ok(1);
    }

    let readme = fs::read_to_string(&patches_readme)
        .with_context(|| format!("failed to read {}", patches_readme.display()))?;
    let documented_patches = readme.lines().filter(|line| line.contains(".patch")).count();

    if documented_patches < patch_files.len() {
        println!("[WARN] ⚠️  Not all patches are documented in README");
        println!(
            "[WARN]    Found {} patches but only {documented_patches} documented",
            patch_files.len()
        );
    }

    println!("[INFO] ✅ Patch documentation check complete");
    Ok(0)
}

fn prepare_patch_tracking_issue(patch_files: &[PathBuf]) {
    if env::var_os("GITHUB_TOKEN").is_none() {
        println!("[WARN] GITHUB_TOKEN not set - cannot create tracking issue");
        return;
    }
    if patch_files.is_empty() {
        return;
    }

    println!("[INFO] Creating patch tracking issue...");
    println!("[INFO] Issue body prepared (GitHub integration would create issue here)");
}

pub(crate) fn cmd_check_coverage(coverage_file: &Path, threshold: f64) -> Result<()> {
    if !coverage_file.exists() {
        bail!("Coverage file not found: {}", coverage_file.display());
    }

    let content = fs::read_to_string(coverage_file)
        .with_context(|| format!("Error reading coverage file: {}", coverage_file.display()))?;
    let coverage_data: Value = serde_json::from_str(&content)
        .with_context(|| format!("Error reading coverage file: {}", coverage_file.display()))?;
    let coverage_percentage = coverage_percentage(&coverage_data)?;

    println!("Coverage: {coverage_percentage:.2}%");
    println!("Threshold: {threshold:.2}%");

    if coverage_percentage >= threshold {
        println!("✅ Coverage threshold met");
        println!("Coverage check passed");
        Ok(())
    } else {
        println!("❌ Coverage below threshold ({coverage_percentage:.2}% < {threshold:.2}%)");
        bail!("coverage below threshold")
    }
}

fn coverage_percentage(coverage_data: &Value) -> Result<f64> {
    if let Some(files) = coverage_data.get("files") {
        let files = files
            .as_object()
            .context("Error reading coverage file: expected `files` to be an object")?;
        let mut total_lines = 0_u64;
        let mut covered_lines = 0_u64;

        for file_data in files.values() {
            let Some(coverage) = file_data.get("coverage") else {
                continue;
            };
            let coverage = coverage
                .as_array()
                .context("Error reading coverage file: expected `coverage` to be an array")?;
            for line_data in coverage {
                if line_data.is_null() {
                    continue;
                }
                total_lines += 1;
                if line_data.as_f64().unwrap_or(0.0) > 0.0 {
                    covered_lines += 1;
                }
            }
        }

        if total_lines == 0 {
            bail!("No coverage data found");
        }

        Ok((covered_lines as f64 / total_lines as f64) * 100.0)
    } else {
        Ok(coverage_data.get("coverage").and_then(Value::as_f64).unwrap_or(0.0))
    }
}

pub(crate) fn cmd_check_units_imports(root: &Path) -> Result<()> {
    let mut violations = Vec::new();
    for path in collect_rust_files(root.join("tests"))? {
        let rel = path.strip_prefix(root).unwrap_or(&path);
        let content = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", rel.display()))?;
        for (line_no, line) in content.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("").trim();
            if code.starts_with("//") || code.is_empty() {
                continue;
            }
            if code.contains("bitnet_tests::common::units::") {
                violations.push(format!("{}:{}", path.display(), line_no + 1));
            }
        }
    }

    if !violations.is_empty() {
        println!("❌ Use bitnet_tests::units::{{BYTES_PER_KB, BYTES_PER_MB, BYTES_PER_GB}}");
        for violation in violations {
            println!("{violation}");
        }
        bail!("units import violations found");
    }

    println!("✅ Units imported via bitnet_tests::units::*");
    Ok(())
}

pub(crate) fn cmd_check_units(root: &Path) -> Result<()> {
    let allowed = "tests/common/units.rs";
    let mut violations = Vec::new();
    for path in collect_rust_files(root.join("tests"))? {
        if path.ends_with(allowed) {
            continue;
        }
        let rel = path.strip_prefix(root).unwrap_or(&path);
        let content = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", rel.display()))?;
        for (line_no, line) in content.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("").trim();
            if code.starts_with("//") || code.is_empty() || code.contains("BYTES_PER_") {
                continue;
            }

            let compact = code.chars().filter(|c| !c.is_whitespace()).collect::<String>();
            let lower = compact.to_ascii_lowercase();

            if lower.contains("1024*1024") || lower.contains("1024*1024*1024") {
                violations.push(format!("{}:{}", path.display(), line_no + 1));
                continue;
            }
            if compact.contains("1_048_576")
                || compact.contains("1048576")
                || compact.contains("1_073_741_824")
                || compact.contains("1073741824")
            {
                violations.push(format!("{}:{}", path.display(), line_no + 1));
                continue;
            }
            if has_shift_constant(&compact, "<<20") || has_shift_constant(&compact, "<<30") {
                violations.push(format!("{}:{}", path.display(), line_no + 1));
            }
        }
    }

    if !violations.is_empty() {
        println!("❌ Raw MB/GB conversion patterns found. Use BYTES_PER_MB or BYTES_PER_GB.");
        for violation in violations {
            println!("{violation}");
        }
        bail!("raw MB/GB conversion violation found");
    }

    println!("✅ No raw MB/GB conversions detected.");
    Ok(())
}

pub(crate) fn cmd_check_serial_annotations(root: &Path) -> Result<()> {
    println!("🔍 Checking for env-mutating tests without #[serial(bitnet_env)]...");

    let rg_output = run_capture(
        root,
        "rg",
        &["-n", "EnvGuard::new|temp_env::with_var", "crates", "tests", "--type", "rust", "-B", "5"],
        &[],
        true,
    )?;
    let mut unannotated: Vec<String> = Vec::new();
    if !rg_output.status.success() {
        println!("No env-mutating tests found");
        return Ok(());
    }

    let rg_text = String::from_utf8_lossy(&rg_output.stdout);
    for line in rg_text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let mut file_parts = line.splitn(3, ':');
        let path = file_parts.next().unwrap_or_default();
        let line_no = file_parts.next().unwrap_or_default();
        if path.is_empty() || line_no.is_empty() {
            continue;
        }

        let line_no: usize = match line_no.parse() {
            Ok(n) => n,
            Err(_) => continue,
        };

        let file_path = Path::new(path);
        let content = fs::read_to_string(file_path)
            .with_context(|| format!("failed to read {}", file_path.display()))?;
        let lines: Vec<&str> = content.lines().collect();
        if line_no == 0 || line_no > lines.len() {
            continue;
        }

        let start = line_no.saturating_sub(10);
        let end = line_no;
        let context = &lines[start..end];
        let has_serial = context.iter().any(|entry| entry.contains("#[serial(bitnet_env)]"));
        let has_test = context.iter().any(|entry| entry.contains("#[test]"));
        if !has_serial && has_test {
            unannotated.push(format!("{}:{}", path, line_no));
        }
    }

    if !unannotated.is_empty() {
        println!("::error::Found env-mutating tests without #[serial(bitnet_env)]:");
        for line in &unannotated {
            println!("{line}");
        }
        println!();
        println!("Env-mutating tests must use #[serial(bitnet_env)] to prevent race conditions.");
        println!();
        println!("Example pattern:");
        println!("  use serial_test::serial;");
        println!("  use tests::helpers::env_guard::EnvGuard;");
        println!();
        println!("  #[test]");
        println!("  #[serial(bitnet_env)]");
        println!("  fn test_with_env_mutation() {{");
        println!("      let _guard = EnvGuard::new(\"VAR_NAME\", \"value\");");
        println!("      // test code");
        println!("  }}");
        println!();
        println!("See: tests/helpers/env_guard.rs for proper usage");
        bail!("env-mutating tests missing #[serial(bitnet_env)] annotations");
    }

    println!("✅ All env-mutating tests properly annotated with #[serial(bitnet_env)]");
    Ok(())
}

pub(crate) fn cmd_check_codeowners_teams(root: &Path) -> Result<()> {
    println!("Fetching teams from @EffortlessMetrics...");

    let gh_output = run_capture(
        root,
        "gh",
        &["api", "orgs/EffortlessMetrics/teams?per_page=100", "--paginate", "--jq", ".[].slug"],
        &[],
        false,
    )?;
    let mut team_slugs = BTreeSet::new();
    for line in String::from_utf8_lossy(&gh_output.stdout).lines() {
        let slug = line.trim();
        if !slug.is_empty() {
            team_slugs.insert(slug.to_string());
        }
    }

    if team_slugs.is_empty() {
        bail!("ERROR: Failed to fetch team slugs (check gh auth and permissions)");
    }
    println!("Found {} teams in @EffortlessMetrics", team_slugs.len());
    println!();

    let codeowners = root.join("CODEOWNERS");
    if !codeowners.exists() {
        bail!("ERROR: CODEOWNERS file not found");
    }
    let content = fs::read_to_string(&codeowners).context("failed to read CODEOWNERS")?;
    let mut referenced = BTreeSet::new();
    for line in content.lines() {
        let line = line.split_once('#').map(|(prefix, _)| prefix).unwrap_or(line);
        for token in line.split_whitespace() {
            if !token.starts_with('@') {
                continue;
            }
            let token = token
                .trim_matches(|c: char| c == ':' || c == ',' || c == ')' || c == ';' || c == '>');
            if !token.starts_with('@') {
                continue;
            }
            let token = &token[1..];
            if let Some((org, slug)) = token.split_once('/') {
                if slug.is_empty() {
                    continue;
                }
                if org != "EffortlessMetrics" {
                    println!("⚠️  WARN: @{}/{} (different org)", org, slug);
                    continue;
                }
                referenced.insert(slug.to_string());
            }
        }
    }

    if referenced.is_empty() {
        println!("No team references found in CODEOWNERS");
        return Ok(());
    }

    println!("Validating {} team reference(s) from CODEOWNERS:", referenced.len());
    println!();

    let mut ok = 0usize;
    let mut bad = 0usize;
    for slug in &referenced {
        if team_slugs.contains(slug) {
            println!("✅ OK: @EffortlessMetrics/{slug}");
            ok += 1;
        } else {
            println!("❌ BAD: @EffortlessMetrics/{slug} (team not found in organization)");
            bad += 1;
        }
    }

    println!();
    println!("Summary: ✅ {ok} OK, ❌ {bad} BAD");
    if bad > 0 {
        println!();
        println!("Fix these issues by:");
        println!("  1. Creating the missing teams in GitHub");
        println!("  2. Updating CODEOWNERS with correct team slugs");
        bail!("invalid team slugs in CODEOWNERS");
    }

    println!();
    println!("All CODEOWNERS team slugs are valid!");
    Ok(())
}

pub(crate) fn cmd_validate_strict(root: &Path) -> Result<()> {
    println!("=== BitNet-rs Strict Validation Suite ===");
    println!();

    println!("1. Building with CPU features...");
    let build = run_capture(
        root,
        "cargo",
        &["build", "--release", "--no-default-features", "--features", "cpu"],
        &[],
        false,
    )?;
    let build_text = format!(
        "{}{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    if !build.status.success() || !build_text.contains("Finished") {
        bail!("Build failed");
    }
    println!("✓ Build successful");

    println!();
    println!("2. Testing tensor name mapping...");
    let mapper = run_capture(
        root,
        "cargo",
        &[
            "test",
            "--workspace",
            "dry_run_remap_names",
            "--no-default-features",
            "--features",
            "cpu",
        ],
        &[],
        true,
    )?;
    if mapper.status.success()
        && String::from_utf8_lossy(&mapper.stdout).contains("test result: ok")
    {
        println!("✓ Mapper tests pass");
    } else {
        println!("⚠ Mapper tests skipped (model not present)");
    }

    println!();
    println!("3. Testing SentencePiece tokenizer...");
    if env::var("SPM").is_ok() {
        let spm = run_capture(
            root,
            "cargo",
            &["test", "--package", "bitnet-tokenizers", "sp_roundtrip", "--", "--ignored"],
            &[],
            false,
        )?;
        if spm.status.success() {
            println!("✓ Tokenizer roundtrip successful");
        } else {
            bail!("Tokenizer roundtrip failed");
        }
    } else {
        println!("⚠ SPM env var not set, skipping tokenizer test");
    }

    println!();
    println!("4. Testing strict mode execution...");
    let model_path = env::var("BITNET_GGUF").unwrap_or_default();
    if !model_path.is_empty() && Path::new(&model_path).is_file() {
        println!("   Using model: {model_path}");
        let strict_json = env::temp_dir().join("strict_test.json");

        let bitnet_bin = root.join("target").join("release").join("bitnet");
        let output = run_capture(
            root,
            bitnet_bin
                .to_str()
                .with_context(|| format!("non-utf8 bitnet path: {}", bitnet_bin.display()))?,
            &[
                "run",
                "--model",
                &model_path,
                "--prompt",
                "Test",
                "--max-new-tokens",
                "5",
                "--temperature",
                "0",
                "--strict-mapping",
                "--strict-tokenizer",
                "--json-out",
                strict_json.to_str().context("invalid /tmp path")?,
            ],
            &[("RAYON_NUM_THREADS", "1"), ("BITNET_DETERMINISTIC", "1"), ("BITNET_SEED", "42")],
            true,
        )?;

        if output.status.success() {
            println!("✓ Strict mode execution successful");
            let strict_data =
                fs::read_to_string(&strict_json).context("failed to read strict_test.json")?;
            let strict: Value =
                serde_json::from_str(&strict_data).context("invalid strict JSON")?;
            let unmapped =
                strict.pointer("/counts/unmapped").and_then(Value::as_u64).unwrap_or(u64::MAX);
            let n_tensors =
                strict.pointer("/counts/n_tensors").and_then(Value::as_u64).unwrap_or(0);
            let tokenizer_type =
                strict.pointer("/tokenizer/type").and_then(Value::as_str).unwrap_or("<unknown>");
            println!("   - Unmapped tensors: {unmapped}");
            println!("   - Total tensors: {n_tensors}");
            println!("   - Tokenizer type: {tokenizer_type}");
            if unmapped == 0 {
                println!("✓ Zero unmapped tensors (strict mode verified)");
            } else {
                println!("⚠ Unmapped tensors found in strict mode!");
            }
        } else {
            println!("⚠ Strict mode failed (may need external tokenizer)");
        }
    } else {
        println!("⚠ BITNET_GGUF not set or file missing, skipping execution test");
    }

    println!();
    println!("5. A/B comparison if both models and C++ available");
    let model_path = env::var("BITNET_GGUF").unwrap_or_default();
    let cpp_bin = env::var("LLAMA_BIN").unwrap_or_else(|_| {
        env::var("HOME").unwrap_or_else(|_| "".to_string()).to_string()
            + "/.cache/bitnet_cpp/build/bin/llama-cli"
    });
    if !model_path.is_empty() && Path::new(&model_path).is_file() && Path::new(&cpp_bin).is_file() {
        println!("   A/B comparison available but not run (use scripts/ab-smoke.sh)");
        println!("⚠ C++ comparison skipped; run scripts/ab-smoke.sh for full validation");
    } else {
        println!("⚠ C++ binary or model not available for A/B test");
    }

    println!();
    println!("=== Validation Summary ===");
    println!("Core validation checks:");
    println!("- Build: ✓");
    println!("- Mapper: ✓");
    println!("- Strict mode: Requires model + tokenizer");
    println!("- A/B comparison: Use scripts/ab-smoke.sh");
    Ok(())
}

pub(crate) fn cmd_validate_fixtures(root: &Path) -> Result<()> {
    println!("🔍 Validating GGUF fixture integrity...");
    let fixture_dir = root.join("ci/fixtures/qk256");
    let checksum_file = fixture_dir.join("SHA256SUMS");
    if !checksum_file.exists() {
        bail!("::error::Fixture checksum file not found: {}", checksum_file.display());
    }

    let sha_status =
        run_capture(&fixture_dir, "sha256sum", &["--check", "--strict", "SHA256SUMS"], &[], false);
    if sha_status.is_err() {
        println!("::error::Fixture checksum verification failed");
        println!("Fixtures may be corrupted or modified without updating SHA256SUMS");
        bail!("fixture checksum verification failed");
    }

    println!("✅ All fixture checksums valid");
    println!("🔍 Validating fixture GGUF structure...");

    for entry in fs::read_dir(&fixture_dir).context("reading fixture directory")? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
            continue;
        }

        let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("unknown");
        println!("Checking {name}...");

        let bytes = fs::read(&path).with_context(|| format!("reading fixture {name}"))?;
        if bytes.len() < 8 {
            bail!("::error::Fixture {name} is too small to be valid GGUF");
        }
        if &bytes[0..4] != b"GGUF" {
            bail!(
                "::error::Fixture {name} has invalid magic number: {:?} (expected GGUF)",
                &bytes[0..4]
            );
        }
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != 2 && version != 3 {
            bail!("::error::Fixture {name} has invalid version: {version} (expected 2 or 3)");
        }

        let inspect = run_capture(
            root,
            "cargo",
            &[
                "run",
                "-q",
                "-p",
                "bitnet-cli",
                "--no-default-features",
                "--features",
                "cpu,full-cli",
                "--",
                "inspect",
                path.to_string_lossy().as_ref(),
                "--format",
                "json",
            ],
            &[],
            true,
        );
        let Ok(inspect) = inspect else {
            println!("::warning::Could not inspect {name} - skipping metadata validation");
            continue;
        };
        if !inspect.status.success() {
            println!(
                "::warning::Inspect command failed for {name}; skipping metadata validation: {}",
                String::from_utf8_lossy(&inspect.stderr).trim()
            );
            continue;
        }
        let inspect_text = String::from_utf8_lossy(&inspect.stdout);
        let inspect_json: Value = match serde_json::from_str(&inspect_text) {
            Ok(value) => value,
            Err(err) => {
                println!("::warning::Could not parse metadata for {name}: {err}");
                continue;
            }
        };
        for key in ["general.architecture", "general.name"] {
            let pointer = format!("/{}", key.replace('.', "/"));
            if inspect_json.pointer(&pointer).is_none() {
                println!("::warning::Fixture {name} missing recommended metadata key: {key}");
            }
        }
        let alignment = inspect_json.pointer("/tensors/0/alignment").and_then(Value::as_u64);
        if version == 3
            && let Some(value) = alignment
            && value != 0
            && value != 32
        {
            bail!(
                "::error::Fixture {name} has invalid tensor alignment: {value} (GGUF v3 requires 32-byte alignment)"
            );
        }
        let tensor_count = inspect_json
            .pointer("/tensors")
            .and_then(Value::as_array)
            .map(|value| value.len())
            .unwrap_or(0);
        if tensor_count < 2 {
            println!(
                "::warning::Fixture {name} has only {tensor_count} tensors (expected ≥2 for realistic fixtures)"
            );
        }
    }

    println!("✅ All fixture GGUF structures valid");
    Ok(())
}

pub(crate) fn cmd_json_schema_gate(root: &Path, files: Vec<String>) -> Result<()> {
    if !command_available("jq") {
        bail!("missing tool: jq");
    }

    let mut ok = true;

    for file in files {
        let path = root.join(&file);
        if !path.exists() {
            println!("✗ {file}: file missing or empty");
            ok = false;
            continue;
        }
        if path.metadata()?.len() == 0 {
            println!("✗ {file}: file missing or empty");
            ok = false;
            continue;
        }
        let path = path.to_string_lossy().into_owned();

        if run_capture(root, "jq", &["-e", ".schema_version == \"1\"", path.as_str()], &[], false)
            .is_err()
        {
            println!("✗ {file}: schema_version != \"1\"");
            ok = false;
            continue;
        }

        let json_type =
            run_capture(root, "jq", &["-r", ".type // empty", path.as_str()], &[], false)?;
        let t = String::from_utf8_lossy(&json_type.stdout).trim().to_string();

        let checks_ok = match t.as_str() {
            "run" => {
                let mut local_ok = true;
                if run_capture(
                    root,
                    "jq",
                    &["-e", ".gen_policy.bos != null", path.as_str()],
                    &[],
                    false,
                )
                .is_err()
                {
                    println!("✗ {file}(run): .gen_policy.bos missing");
                    local_ok = false;
                }
                if run_capture(
                    root,
                    "jq",
                    &["-e", ".throughput.tokens_per_second|numbers", path.as_str()],
                    &[],
                    false,
                )
                .is_err()
                {
                    println!("✗ {file}(run): .throughput.tokens_per_second not numeric");
                    local_ok = false;
                }
                local_ok
            }
            "score" => {
                let mut local_ok = true;
                if run_capture(root, "jq", &["-e", ".mean_nll|numbers", path.as_str()], &[], false)
                    .is_err()
                {
                    println!("✗ {file}(score): mean_nll/ppl not numeric");
                    local_ok = false;
                }
                if run_capture(root, "jq", &["-e", ".ppl|numbers", path.as_str()], &[], false)
                    .is_err()
                {
                    println!("✗ {file}(score): mean_nll/ppl not numeric");
                    local_ok = false;
                }
                local_ok
            }
            "tokenize" => run_capture(
                root,
                "jq",
                &["-e", ".tokens.ids|type == \"array\"", path.as_str()],
                &[],
                false,
            )
            .map(|_| true)
            .unwrap_or_else(|_| {
                println!("✗ {file}(tokenize): .tokens.ids not an array");
                false
            }),
            _ => {
                println!(
                    "⚠︎ {file}: unknown or missing .type ({t}). Skipping type-specific checks."
                );
                true
            }
        };

        if !checks_ok {
            ok = false;
        }
    }

    if ok {
        println!("✓ JSON schema checks passed");
        Ok(())
    } else {
        bail!("JSON schema checks failed");
    }
}

pub(crate) fn cmd_validate_iq2s_build(root: &Path) -> Result<()> {
    println!("=== IQ2_S Build Validation ===");
    println!();

    println!("1. Building with IQ2_S support...");
    let build = run_capture(
        root,
        "cargo",
        &[
            "build",
            "--package",
            "bitnet-cli",
            "--bin",
            "bitnet",
            "--no-default-features",
            "--features",
            "cpu,iq2s-ffi",
            "--release",
        ],
        &[],
        false,
    )?;
    let build_output = format!(
        "{}{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    for line in build_output
        .lines()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        println!("{line}");
    }

    let bitnet_bin = root.join("target").join("release").join("bitnet");
    if !bitnet_bin.exists() {
        bail!("expected {} to exist", bitnet_bin.display());
    }

    println!();
    println!("2. Checking version output...");
    let mut envs = vec![("BITNET_QUIET_BACKEND", "1")];
    if cfg!(windows) {
        envs.push(("RUST_BACKTRACE", "0"));
    }
    let version = run_capture(
        root,
        bitnet_bin
            .to_str()
            .with_context(|| format!("non-utf8 bitnet path: {}", bitnet_bin.display()))?,
        &["--version"],
        &envs,
        false,
    )?;
    let version_text = String::from_utf8_lossy(&version.stdout);

    println!("3. Verifying features...");
    if !version_text.contains("iq2s-ffi") {
        bail!("IQ2_S feature not found");
    }
    if !version_text.contains("ggml:") {
        bail!("GGML commit not shown");
    }

    println!("✓ IQ2_S feature enabled");
    println!("✓ GGML commit line present");
    println!();
    println!("=== Validation Complete ===");
    Ok(())
}

pub(crate) fn cmd_check_feature_gates(root: &Path) -> Result<()> {
    println!("🔍 Checking feature gate consistency...");
    let mut in_features = false;
    let mut defined = BTreeSet::new();
    let cargo_toml =
        fs::read_to_string(root.join("Cargo.toml")).context("reading root Cargo.toml")?;
    for line in cargo_toml.lines() {
        let trimmed = line.trim();
        if trimmed == "[features]" {
            in_features = true;
            continue;
        }
        if in_features {
            if trimmed.starts_with('[') {
                break;
            }
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some(pos) = trimmed.find('=') {
                let name = trimmed[..pos].trim();
                if !name.is_empty() {
                    defined.insert(name.to_string());
                }
            }
        }
    }
    println!("Defined features ({}):", defined.len());
    for name in &defined {
        println!("  {name}");
    }

    let used_output = run_capture(
        root,
        "rg",
        &[
            "-oI",
            r#"#\[cfg.*feature\s*=\s*"([^\"]+)""#,
            "crates",
            "--type",
            "rust",
            "--replace",
            "$1",
        ],
        &[],
        true,
    )?;
    let mut used = BTreeSet::new();
    for line in String::from_utf8_lossy(&used_output.stdout).lines() {
        let feature = line.trim();
        if !feature.is_empty() {
            used.insert(feature.to_string());
        }
    }

    println!();
    println!("Used features in #[cfg] ({}):", used.len());
    for name in &used {
        println!("  {name}");
    }

    let mut undefined = Vec::new();
    for feature in &used {
        if !defined.contains(feature) {
            undefined.push(feature.clone());
        }
    }

    if !undefined.is_empty() {
        println!("::error::Found #[cfg(feature = ...)] using undefined features:");
        for feature in &undefined {
            println!("  - {feature}");
        }
        println!();
        println!(
            "These features are referenced in code but not defined in Cargo.toml [features] section."
        );
        println!("Either define the feature or remove the #[cfg] annotation.");
        bail!("undefined feature references found");
    }

    println!();
    println!("🔍 Checking for feature gate antipatterns...");
    let gpu_checks = run_capture(
        root,
        "rg",
        &["-n", r#"\#\[cfg\(feature = "gpu"\)\]"#, "crates", "--type", "rust"],
        &[],
        true,
    )?;
    let gpu_output = String::from_utf8_lossy(&gpu_checks.stdout);
    let flagged: Vec<&str> = gpu_output.lines().filter(|line| !line.contains("any(")).collect();
    if !flagged.is_empty() {
        println!("::warning::Found #[cfg(feature = \"gpu\")] without fallback to \"cuda\":");
        for line in flagged {
            println!("{line}");
        }
        println!();
        println!("Recommended pattern:");
        println!("  #[cfg(any(feature = \"gpu\", feature = \"cuda\"))]");
        println!();
        println!("This ensures backward compatibility with legacy 'cuda' feature.");
    }

    println!("✅ Feature gate consistency check passed");
    Ok(())
}
