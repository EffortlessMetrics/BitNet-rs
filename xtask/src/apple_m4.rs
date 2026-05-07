use anyhow::{Context, Result, bail};
use chrono::Utc;
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const MACHINE_ID: &str = "apple-m4-mac-mini";
const APPLE_M4_METAL: &str = "apple-m4-metal";
const APPLE_M4_CPU_NEON: &str = "apple-m4-cpu-neon";
const APPLE_M4_MPSGRAPH: &str = "apple-m4-mpsgraph";

#[derive(Subcommand)]
pub enum AppleM4Cmd {
    /// Run the Apple M4 operational validation bundle.
    Validate(ValidateArgs),
    /// Validate an Apple M4 receipt bundle.
    #[command(name = "receipts-check")]
    ReceiptsCheck(ReceiptsCheckArgs),
}

#[derive(Args)]
pub struct ValidateArgs {
    /// Receipt date segment, usually YYYY-MM-DD. Defaults to today's UTC date.
    #[arg(long)]
    date: Option<String>,
    /// Canonical BitNet GGUF model path.
    #[arg(long)]
    model: PathBuf,
    /// Output directory for the receipt bundle.
    #[arg(long, visible_alias = "out-dir")]
    out: Option<PathBuf>,
    /// Prompt for strict CPU/NEON proof and profile receipts.
    #[arg(long, default_value = "Answer with a single digit: 2+2=")]
    prompt: String,
    /// Cargo executable to invoke for proof commands.
    #[arg(long, default_value = "cargo")]
    cargo: String,
}

#[derive(Args)]
pub struct ReceiptsCheckArgs {
    /// Directory containing the Apple M4 receipt bundle.
    dir: PathBuf,
    /// Print the validation report as JSON.
    #[arg(long, default_value_t = false)]
    json: bool,
    /// Allow summary.json to be absent. Used internally before validate writes it.
    #[arg(long, default_value_t = false)]
    allow_missing_summary: bool,
}

pub fn run(cmd: AppleM4Cmd) -> Result<()> {
    let root = std::env::current_dir().context("resolve current directory")?;
    match cmd {
        AppleM4Cmd::Validate(args) => validate(&root, args),
        AppleM4Cmd::ReceiptsCheck(args) => receipts_check(&root, args),
    }
}

fn validate(root: &Path, args: ValidateArgs) -> Result<()> {
    let date = args.date.unwrap_or_else(|| Utc::now().date_naive().to_string());
    let out_logical =
        args.out.unwrap_or_else(|| PathBuf::from(format!("ci/hardware/{MACHINE_ID}/{date}")));
    let paths = BundlePaths::new(root, out_logical)?;
    let model_path = resolve_against(root, &args.model);
    if !model_path.exists() {
        bail!("Apple M4 validation model does not exist: {}", model_path.display());
    }

    fs::create_dir_all(&paths.out_abs)
        .with_context(|| format!("create receipt directory {}", paths.out_abs.display()))?;

    println!("Apple M4 validation bundle: {}", paths.out_abs.display());
    let profile = write_machine_profile(&paths)?;
    write_metal_probe(&paths, &profile)?;

    run_metal_test(
        root,
        &args.cargo,
        "tiny_m4_metal_add_smoke_runs_when_enabled",
        "BITNET_RUN_M4_METAL_SMOKE",
        "BITNET_M4_METAL_SMOKE_RECEIPT",
        "BITNET_M4_METAL_SMOKE_ARTIFACT_PATH",
        paths.file_abs("metal-smoke.json"),
        paths.file_logical("metal-smoke.json"),
    )?;
    run_metal_test(
        root,
        &args.cargo,
        "tiny_m4_metal_i2s_matches_cpu_neon_reference_when_enabled",
        "BITNET_RUN_M4_METAL_I2S_PARITY",
        "BITNET_M4_METAL_I2S_PARITY_RECEIPT",
        "BITNET_M4_METAL_I2S_PARITY_ARTIFACT_PATH",
        paths.file_abs("metal-i2s-parity.json"),
        paths.file_logical("metal-i2s-parity.json"),
    )?;
    run_mpsgraph_test(
        root,
        &args.cargo,
        paths.file_abs("mpsgraph-smoke.json"),
        paths.file_logical("mpsgraph-smoke.json"),
    )?;
    run_strict_cpu_proof(
        root,
        &args.cargo,
        &args.model,
        &args.prompt,
        1,
        None,
        false,
        paths.file_abs("strict-bitnet-cpu-neon-proof.json"),
    )?;
    run_strict_cpu_proof(
        root,
        &args.cargo,
        &args.model,
        &args.prompt,
        4,
        Some("smoke_4"),
        false,
        paths.file_abs("phase-profile.json"),
    )?;
    run_strict_cpu_proof(
        root,
        &args.cargo,
        &args.model,
        &args.prompt,
        4,
        Some("smoke_4"),
        true,
        paths.file_abs("allocation-audit.json"),
    )?;

    let report = check_bundle(&paths.out_abs, true)?;
    write_summary(&paths, &report)?;
    check_bundle(&paths.out_abs, false)?;
    println!(
        "Apple M4 validation receipts passed: {}",
        paths.file_logical("summary.json").display()
    );
    Ok(())
}

fn receipts_check(root: &Path, args: ReceiptsCheckArgs) -> Result<()> {
    let dir = resolve_against(root, &args.dir);
    let report = check_bundle(&dir, args.allow_missing_summary)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("Apple M4 receipt bundle passed: {}", dir.display());
        for receipt in &report.receipts {
            println!(
                "- {}: {}{}",
                receipt.file,
                receipt.status,
                receipt.artifact_kind.as_ref().map(|kind| format!(" ({kind})")).unwrap_or_default()
            );
        }
    }
    Ok(())
}

#[derive(Debug)]
struct BundlePaths {
    root: PathBuf,
    out_abs: PathBuf,
    out_logical: PathBuf,
}

impl BundlePaths {
    fn new(root: &Path, out_logical: PathBuf) -> Result<Self> {
        let out_abs = resolve_against(root, &out_logical);
        Ok(Self { root: root.to_path_buf(), out_abs, out_logical })
    }

    fn file_abs(&self, file: &str) -> PathBuf {
        self.out_abs.join(file)
    }

    fn file_logical(&self, file: &str) -> PathBuf {
        self.out_logical.join(file)
    }
}

fn resolve_against(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() { path.to_path_buf() } else { root.join(path) }
}

#[derive(Debug, Serialize, Deserialize)]
struct CommandCapture {
    command: String,
    status: Option<i32>,
    stdout: String,
    stderr: String,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct BundleCheckReport {
    machine_id: &'static str,
    checked_at: String,
    directory: String,
    receipts: Vec<ReceiptCheck>,
    status: &'static str,
}

#[derive(Debug, Serialize)]
struct ReceiptCheck {
    file: String,
    status: &'static str,
    artifact_kind: Option<String>,
    requested_backend: Option<String>,
    selected_backend: Option<String>,
    runtime_api: Option<String>,
    fallback_used: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReceiptRole {
    MachineProfile,
    MetalProbe,
    MetalSmoke,
    MetalI2sParity,
    MpsGraphSmoke,
    StrictCpuProof,
    PhaseProfile,
    AllocationAudit,
    Summary,
}

impl ReceiptRole {
    fn is_backend_receipt(self) -> bool {
        !matches!(self, Self::MachineProfile | Self::Summary)
    }

    fn is_bitnet_receipt(self) -> bool {
        matches!(
            self,
            Self::MetalI2sParity
                | Self::StrictCpuProof
                | Self::PhaseProfile
                | Self::AllocationAudit
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct ExpectedReceipt {
    file: &'static str,
    role: ReceiptRole,
}

const EXPECTED_RECEIPTS: &[ExpectedReceipt] = &[
    ExpectedReceipt { file: "machine-profile.json", role: ReceiptRole::MachineProfile },
    ExpectedReceipt { file: "metal-probe.json", role: ReceiptRole::MetalProbe },
    ExpectedReceipt { file: "metal-smoke.json", role: ReceiptRole::MetalSmoke },
    ExpectedReceipt { file: "metal-i2s-parity.json", role: ReceiptRole::MetalI2sParity },
    ExpectedReceipt { file: "mpsgraph-smoke.json", role: ReceiptRole::MpsGraphSmoke },
    ExpectedReceipt {
        file: "strict-bitnet-cpu-neon-proof.json",
        role: ReceiptRole::StrictCpuProof,
    },
    ExpectedReceipt { file: "phase-profile.json", role: ReceiptRole::PhaseProfile },
    ExpectedReceipt { file: "allocation-audit.json", role: ReceiptRole::AllocationAudit },
    ExpectedReceipt { file: "summary.json", role: ReceiptRole::Summary },
];

fn write_machine_profile(paths: &BundlePaths) -> Result<Value> {
    let commands = json!({
        "sw_vers": capture_command("sw_vers", [] as [&str; 0]),
        "uname": capture_command("uname", ["-a"]),
        "system_profiler_hardware": capture_command("system_profiler", ["SPHardwareDataType"]),
        "system_profiler_displays": capture_command("system_profiler", ["SPDisplaysDataType"]),
        "system_profiler_metal": capture_command("system_profiler", ["SPMetalDataType"]),
        "vm_stat": capture_command("vm_stat", [] as [&str; 0]),
        "sysctl_memsize": capture_command("sysctl", ["hw.memsize"]),
        "sysctl_cpu_brand": capture_command("sysctl", ["machdep.cpu.brand_string"]),
        "sysctl_neon": capture_command("sysctl", ["hw.optional.neon"]),
        "sysctl_vmm": capture_command("sysctl", ["kern.hv_vmm_present"]),
        "rustc": capture_command("rustc", ["--version"]),
        "cargo": capture_command("cargo", ["--version"]),
    });

    let hardware = command_stdout(&commands, "system_profiler_hardware");
    let metal = command_stdout(&commands, "system_profiler_metal");
    let displays = command_stdout(&commands, "system_profiler_displays");
    let memsize = command_stdout(&commands, "sysctl_memsize");
    let vmm = command_stdout(&commands, "sysctl_vmm");
    let sw_vers = command_stdout(&commands, "sw_vers");
    let rustc = command_stdout(&commands, "rustc");
    let cargo = command_stdout(&commands, "cargo");
    let chip = parse_after_colon(&hardware, "Chip").unwrap_or_else(|| "unknown".to_string());
    let cpu_core_count = parse_core_count(&hardware);
    let unified_memory_bytes = parse_sysctl_u64(&memsize, "hw.memsize");
    let metal_visible = command_success(&commands, "system_profiler_metal")
        && (!metal.trim().is_empty() || displays.contains("Metal"));
    let native_macos = parse_sysctl_u64(&vmm, "kern.hv_vmm_present").map(|value| value == 0);

    let receipt = json!({
        "schema_version": "1.0.0",
        "timestamp": Utc::now().to_rfc3339(),
        "machine_id": MACHINE_ID,
        "artifact_kind": "machine_profile",
        "artifact_path": paths.file_logical("machine-profile.json").display().to_string(),
        "requested_backend": null,
        "selected_backend": null,
        "runtime_api": "macos-system-profiler",
        "fallback_used": false,
        "claim_level": "machine_profile_only",
        "kernel_execution": false,
        "graph_execution": false,
        "bitnet_inference": false,
        "machine": {
            "chip": chip,
            "cpu_core_count": cpu_core_count,
            "gpu_core_count": null,
            "unified_memory": chip_starts_with_apple(&hardware),
            "unified_memory_size_bytes": unified_memory_bytes,
            "memory_bandwidth_class": null,
            "macos_version": parse_sw_vers(&sw_vers, "ProductVersion"),
            "macos_build": parse_sw_vers(&sw_vers, "BuildVersion"),
            "metal_visible": metal_visible,
            "native_macos": native_macos,
        },
        "rust_toolchain": {
            "rustc": first_nonempty_line(&rustc),
            "cargo": first_nonempty_line(&cargo),
        },
        "commands": commands,
    });
    write_json(paths.file_abs("machine-profile.json"), &receipt)?;
    Ok(receipt)
}

fn write_metal_probe(paths: &BundlePaths, profile: &Value) -> Result<()> {
    let machine = profile.get("machine").unwrap_or(&Value::Null);
    let chip = machine.get("chip").and_then(Value::as_str).unwrap_or("unknown").to_string();
    let metal_visible = machine.get("metal_visible").and_then(Value::as_bool).unwrap_or(false);
    let selected_backend = metal_visible.then_some(APPLE_M4_METAL);
    let receipt = json!({
        "schema_version": "1.0.0",
        "timestamp": Utc::now().to_rfc3339(),
        "machine_id": MACHINE_ID,
        "artifact_kind": "probe",
        "artifact_path": paths.file_logical("metal-probe.json").display().to_string(),
        "requested_backend": APPLE_M4_METAL,
        "selected_backend": selected_backend,
        "runtime_api": "metal",
        "fallback_used": false,
        "claim_level": if metal_visible { "runtime_detected" } else { "runtime_unavailable" },
        "metal_visible": metal_visible,
        "metal_execution": false,
        "bitnet_inference": false,
        "resolved_device": {
            "chip": chip,
            "gpu_cores": machine.get("gpu_core_count").cloned().unwrap_or(Value::Null),
            "unified_memory": machine.get("unified_memory").cloned().unwrap_or(Value::Null),
            "unified_memory_size_bytes": machine
                .get("unified_memory_size_bytes")
                .cloned()
                .unwrap_or(Value::Null),
        },
    });
    write_json(paths.file_abs("metal-probe.json"), &receipt)
}

#[allow(clippy::too_many_arguments)]
fn run_metal_test(
    root: &Path,
    cargo: &str,
    test_name: &str,
    run_env: &str,
    receipt_env: &str,
    artifact_env: &str,
    receipt_path: PathBuf,
    artifact_path: PathBuf,
) -> Result<()> {
    let mut command = Command::new(cargo);
    command
        .current_dir(root)
        .env(run_env, "1")
        .env(receipt_env, receipt_path.as_os_str())
        .env(artifact_env, artifact_path.as_os_str())
        .args([
            "test",
            "--locked",
            "-p",
            "bitnet-kernels",
            "--no-default-features",
            "--features",
            "metal",
            "--test",
            "metal_tiny_smoke",
            test_name,
            "--",
            "--nocapture",
        ]);
    run_command(command, format!("Metal proof test {test_name}"))
}

fn run_mpsgraph_test(
    root: &Path,
    cargo: &str,
    receipt_path: PathBuf,
    artifact_path: PathBuf,
) -> Result<()> {
    let mut command = Command::new(cargo);
    command
        .current_dir(root)
        .env("BITNET_RUN_M4_MPSGRAPH_SMOKE", "1")
        .env("BITNET_M4_MPSGRAPH_SMOKE_RECEIPT", receipt_path.as_os_str())
        .env("BITNET_M4_MPSGRAPH_SMOKE_ARTIFACT_PATH", artifact_path.as_os_str())
        .args([
            "test",
            "--locked",
            "-p",
            "bitnet-device-probe",
            "--no-default-features",
            "--features",
            "metal",
            "--test",
            "apple_mpsgraph_smoke",
            "tiny_m4_mpsgraph_matmul_smoke_runs_when_enabled",
            "--",
            "--nocapture",
        ]);
    run_command(command, "MPSGraph smoke proof")
}

#[allow(clippy::too_many_arguments)]
fn run_strict_cpu_proof(
    root: &Path,
    cargo: &str,
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    profile_id: Option<&str>,
    allocation_audit: bool,
    json_out: PathBuf,
) -> Result<()> {
    let mut command = Command::new(cargo);
    command
        .current_dir(root)
        .env("BITNET_DISABLE_MINIMAL_LOADER", "1")
        .env("BITNET_STRICT_MODE", "1")
        .arg("run")
        .arg("--locked")
        .arg("-p")
        .arg("bitnet-cli")
        .arg("--no-default-features")
        .arg("--features")
        .arg("cpu,full-cli")
        .arg("--")
        .arg("--device")
        .arg(APPLE_M4_CPU_NEON)
        .arg("run")
        .arg("--model")
        .arg(model.as_os_str())
        .arg("--prompt")
        .arg(prompt)
        .arg("--max-tokens")
        .arg(max_tokens.to_string())
        .arg("--temperature")
        .arg("0.0")
        .arg("--greedy")
        .arg("--deterministic")
        .arg("--strict-loader")
        .arg("--strict-tokenizer")
        .arg("--prompt-template")
        .arg("raw")
        .arg("--no-warnings");
    if let Some(profile_id) = profile_id {
        command.arg("--profile-id").arg(profile_id);
    }
    if allocation_audit {
        command.arg("--allocation-audit");
    }
    command.arg("--json-out").arg(json_out.as_os_str());
    run_command(command, "strict Apple M4 CPU/NEON BitNet proof")
}

fn run_command(mut command: Command, label: impl AsRef<str>) -> Result<()> {
    println!("running: {}", label.as_ref());
    let status = command.status().with_context(|| format!("spawn {}", label.as_ref()))?;
    if !status.success() {
        bail!("{} failed with status {}", label.as_ref(), status);
    }
    Ok(())
}

fn capture_command<I, S>(program: &str, args: I) -> CommandCapture
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let args = args.into_iter().map(|arg| arg.as_ref().to_owned()).collect::<Vec<_>>();
    let command = std::iter::once(program.to_string())
        .chain(args.iter().map(|arg| arg.to_string_lossy().to_string()))
        .collect::<Vec<_>>()
        .join(" ");
    match Command::new(program).args(&args).output() {
        Ok(output) => CommandCapture {
            command,
            status: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).trim().to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            error: None,
        },
        Err(error) => CommandCapture {
            command,
            status: None,
            stdout: String::new(),
            stderr: String::new(),
            error: Some(error.to_string()),
        },
    }
}

fn check_bundle(dir: &Path, allow_missing_summary: bool) -> Result<BundleCheckReport> {
    let mut receipts = Vec::new();
    let mut failures = Vec::new();

    for expected in EXPECTED_RECEIPTS {
        if expected.role == ReceiptRole::Summary && allow_missing_summary {
            continue;
        }
        let path = dir.join(expected.file);
        if !path.exists() {
            failures.push(format!("missing receipt {}", path.display()));
            continue;
        }
        let value = match read_json(&path) {
            Ok(value) => value,
            Err(error) => {
                failures.push(error.to_string());
                continue;
            }
        };
        if let Err(error) = validate_receipt(expected.file, expected.role, &value) {
            failures.push(error.to_string());
        }
        receipts.push(ReceiptCheck {
            file: expected.file.to_string(),
            status: "pass",
            artifact_kind: value.get("artifact_kind").and_then(Value::as_str).map(str::to_string),
            requested_backend: value
                .get("requested_backend")
                .and_then(Value::as_str)
                .map(str::to_string),
            selected_backend: value
                .get("selected_backend")
                .and_then(Value::as_str)
                .map(str::to_string),
            runtime_api: value.get("runtime_api").and_then(Value::as_str).map(str::to_string),
            fallback_used: value.get("fallback_used").and_then(Value::as_bool),
        });
    }

    if !failures.is_empty() {
        bail!("Apple M4 receipt bundle check failed:\n{}", failures.join("\n"));
    }

    Ok(BundleCheckReport {
        machine_id: MACHINE_ID,
        checked_at: Utc::now().to_rfc3339(),
        directory: dir.display().to_string(),
        receipts,
        status: "pass",
    })
}

fn validate_receipt(file: &str, role: ReceiptRole, value: &Value) -> Result<()> {
    require_string(value, "artifact_path", file)?;
    require_bool(value, "fallback_used", file)?;
    validate_fallback_fields(file, value)?;

    if role.is_backend_receipt() {
        require_string(value, "requested_backend", file)?;
        require_string_or_null(value, "selected_backend", file)?;
        require_string(value, "runtime_api", file)?;
        validate_backend_fallback(file, value)?;
    }

    match role {
        ReceiptRole::MachineProfile => {
            require_eq(value, "artifact_kind", "machine_profile", file)?;
            require_bool_path(value, &["machine", "metal_visible"], file)?;
        }
        ReceiptRole::MetalProbe => {
            require_eq(value, "artifact_kind", "probe", file)?;
            require_eq(value, "requested_backend", APPLE_M4_METAL, file)?;
            require_eq(value, "runtime_api", "metal", file)?;
        }
        ReceiptRole::MetalSmoke => {
            validate_metal_execution_receipt(file, value)?;
            require_eq(value, "artifact_kind", "smoke", file)?;
            require_string(value, "kernel_id", file)?;
            require_eq(value, "result", "pass", file)?;
        }
        ReceiptRole::MetalI2sParity => {
            validate_metal_execution_receipt(file, value)?;
            validate_bitnet_fields(file, value)?;
            require_eq(value, "artifact_kind", "parity", file)?;
            require_eq_path(value, &["bitnet", "kernel_family"], "i2_s", file)?;
            require_eq_path(value, &["bitnet", "execution_phase"], "parity", file)?;
            require_eq_path_bool(value, &["layout", "consumes_packed_i2_s_directly"], true, file)?;
            require_eq_path_bool(value, &["layout", "dequantizes_before_compute"], false, file)?;
            require_object(value, "parity", file)?;
        }
        ReceiptRole::MpsGraphSmoke => {
            require_eq(value, "requested_backend", APPLE_M4_MPSGRAPH, file)?;
            require_eq(value, "selected_backend", APPLE_M4_MPSGRAPH, file)?;
            require_eq(value, "runtime_api", "mpsgraph", file)?;
            require_eq_bool(value, "fallback_used", false, file)?;
            require_string(value, "graph_id", file)?;
            reject_mpsgraph_neural_engine_claim(file, value)?;
        }
        ReceiptRole::StrictCpuProof => {
            validate_strict_cpu_bitnet_receipt(file, value)?;
        }
        ReceiptRole::PhaseProfile => {
            validate_strict_cpu_bitnet_receipt(file, value)?;
            require_object(value, "profile", file)?;
            require_eq_path_bool(value, &["profile", "requested"], true, file)?;
        }
        ReceiptRole::AllocationAudit => {
            validate_strict_cpu_bitnet_receipt(file, value)?;
            require_eq_path_bool(value, &["profile", "allocation_audit", "enabled"], true, file)?;
        }
        ReceiptRole::Summary => {
            require_eq(value, "artifact_kind", "apple_m4_validation_summary", file)?;
            require_array(value, "proven", file)?;
            require_array(value, "not_proven", file)?;
        }
    }

    if role.is_bitnet_receipt() {
        validate_bitnet_fields(file, value)?;
    }
    if role != ReceiptRole::MachineProfile {
        reject_unsupported_claims(file, role, value)?;
    }
    Ok(())
}

fn validate_metal_execution_receipt(file: &str, value: &Value) -> Result<()> {
    require_eq(value, "requested_backend", APPLE_M4_METAL, file)?;
    require_eq(value, "selected_backend", APPLE_M4_METAL, file)?;
    require_eq(value, "runtime_api", "metal", file)?;
    require_eq_bool(value, "fallback_used", false, file)
}

fn validate_strict_cpu_bitnet_receipt(file: &str, value: &Value) -> Result<()> {
    require_eq(value, "requested_backend", APPLE_M4_CPU_NEON, file)?;
    require_eq(value, "selected_backend", APPLE_M4_CPU_NEON, file)?;
    require_eq(value, "runtime_api", "cpu", file)?;
    require_eq_bool(value, "fallback_used", false, file)?;
    require_object(value, "model", file)?;
    require_model_tokenizer(file, value)
}

fn validate_bitnet_fields(file: &str, value: &Value) -> Result<()> {
    require_object(value, "bitnet", file)?;
    require_string_path(value, &["bitnet", "kernel_family"], file)?;
    require_string_path(value, &["bitnet", "execution_phase"], file)
}

fn validate_fallback_fields(file: &str, value: &Value) -> Result<()> {
    let fallback_used = value.get("fallback_used").and_then(Value::as_bool).unwrap_or(false);
    let fallback_reason = value.get("fallback_reason");
    if fallback_used {
        let Some(reason) = fallback_reason.and_then(Value::as_str) else {
            bail!("{file}: fallback_used=true requires fallback_reason");
        };
        if reason.trim().is_empty() {
            bail!("{file}: fallback_reason must not be empty when fallback_used=true");
        }
    } else if fallback_reason.and_then(Value::as_str).is_some_and(|reason| !reason.is_empty()) {
        bail!("{file}: fallback_reason must be absent or null when fallback_used=false");
    }
    Ok(())
}

fn validate_backend_fallback(file: &str, value: &Value) -> Result<()> {
    let requested = value.get("requested_backend").and_then(Value::as_str);
    let selected = value.get("selected_backend").and_then(Value::as_str);
    if let (Some(requested), Some(selected)) = (requested, selected)
        && requested != selected
    {
        let reason = value.get("fallback_reason").and_then(Value::as_str).unwrap_or_default();
        if reason.trim().is_empty() {
            bail!(
                "{file}: selected_backend differs from requested_backend without fallback_reason"
            );
        }
    }
    Ok(())
}

fn require_model_tokenizer(file: &str, value: &Value) -> Result<()> {
    if value.get("model").and_then(|model| model.get("tokenizer")).is_some()
        || value.get("tokenizer").is_some()
    {
        Ok(())
    } else {
        bail!("{file}: BitNet receipt missing model.tokenizer or tokenizer")
    }
}

fn reject_mpsgraph_neural_engine_claim(file: &str, value: &Value) -> Result<()> {
    let resolved_target = value.get("resolved_target").and_then(Value::as_str).unwrap_or("unknown");
    if matches!(
        resolved_target.to_ascii_lowercase().as_str(),
        "ane" | "neural_engine" | "neural-engine"
    ) {
        bail!("{file}: MPSGraph resolved target claims Neural Engine without separate proof");
    }
    Ok(())
}

fn reject_unsupported_claims(file: &str, role: ReceiptRole, value: &Value) -> Result<()> {
    let mut strings = Vec::new();
    collect_strings(value, String::new(), &mut strings);
    for (path, text) in strings {
        let lower_path = path.to_ascii_lowercase();
        let lower_text = text.to_ascii_lowercase();
        let in_not_proven = is_not_proven_path(&lower_path);
        if !in_not_proven && (lower_path.contains("qk256") || lower_text.contains("qk256")) {
            bail!("{file}: unsupported Apple QK256 claim at {path}");
        }
        if !in_not_proven
            && (lower_path.contains("neural_engine")
                || lower_path.contains("neural-engine")
                || lower_text.contains("neural engine"))
        {
            bail!("{file}: unsupported Neural Engine claim at {path}");
        }
        if !in_not_proven && lower_text.contains("full apple-m4-metal model inference") {
            bail!("{file}: unsupported full apple-m4-metal inference claim at {path}");
        }
        if role == ReceiptRole::MpsGraphSmoke
            && !in_not_proven
            && lower_text.contains("native metal")
        {
            bail!("{file}: MPSGraph receipt claims native Metal at {path}");
        }
    }
    Ok(())
}

fn is_not_proven_path(path: &str) -> bool {
    path == "not_proven" || path.starts_with("not_proven[") || path.contains(".not_proven[")
}

fn collect_strings(value: &Value, path: String, out: &mut Vec<(String, String)>) {
    if !path.is_empty() {
        out.push((path.clone(), String::new()));
    }
    match value {
        Value::String(text) => out.push((path, text.clone())),
        Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                collect_strings(value, format!("{path}[{index}]"), out);
            }
        }
        Value::Object(map) => {
            for (key, value) in map {
                let next = if path.is_empty() { key.clone() } else { format!("{path}.{key}") };
                collect_strings(value, next, out);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn write_summary(paths: &BundlePaths, report: &BundleCheckReport) -> Result<()> {
    let receipts = report
        .receipts
        .iter()
        .map(|receipt| {
            json!({
                "file": receipt.file,
                "status": receipt.status,
                "artifact_kind": receipt.artifact_kind,
                "requested_backend": receipt.requested_backend,
                "selected_backend": receipt.selected_backend,
                "runtime_api": receipt.runtime_api,
                "fallback_used": receipt.fallback_used,
            })
        })
        .collect::<Vec<_>>();
    let summary = json!({
        "schema_version": "1.0.0",
        "timestamp": Utc::now().to_rfc3339(),
        "artifact_kind": "apple_m4_validation_summary",
        "machine_id": MACHINE_ID,
        "artifact_path": paths.file_logical("summary.json").display().to_string(),
        "requested_backend": null,
        "selected_backend": null,
        "runtime_api": "xtask-apple-m4",
        "fallback_used": false,
        "output_dir": paths.out_logical.display().to_string(),
        "workspace_root": paths.root.display().to_string(),
        "receipt_check": {
            "status": report.status,
            "checked_at": report.checked_at,
            "receipts": receipts,
        },
        "proven": [
            "Apple M4 machine profile was recorded.",
            "Apple M4 Metal runtime visibility was recorded.",
            "Tiny native Metal compute smoke receipt is present with fallback_used=false.",
            "I2_S-adjacent Metal parity receipt is present with CPU/NEON reference and fallback_used=false.",
            "Tiny MPSGraph reference smoke receipt is present as graph/reference evidence.",
            "Strict BitNet CPU/NEON proof receipt is present for the selected Apple CPU backend.",
            "CPU/NEON profile and allocation receipts are present for the recorded profile."
        ],
        "not_proven": [
            "Full apple-m4-metal model inference is not proven by this bundle.",
            "QK256 on Apple Silicon is not proven by this bundle.",
            "Neural Engine execution is not proven by this bundle.",
            "General M4 performance is not proven by this bundle.",
            "MPSGraph evidence is not native Metal kernel proof."
        ],
        "fallback_policy": [
            "CPU fallback must be explicit.",
            "Metal proof cannot be counted when selected_backend is CPU.",
            "MPSGraph is graph/reference evidence only unless a resolved target is separately receipt-backed."
        ]
    });
    write_json(paths.file_abs("summary.json"), &summary)
}

fn write_json(path: PathBuf, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(&path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("write {}", path.display()))
}

fn read_json(path: &Path) -> Result<Value> {
    let contents = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&contents).with_context(|| format!("parse JSON {}", path.display()))
}

fn require_string(value: &Value, field: &str, file: &str) -> Result<()> {
    if value.get(field).and_then(Value::as_str).is_some_and(|text| !text.trim().is_empty()) {
        Ok(())
    } else {
        bail!("{file}: missing string field `{field}`")
    }
}

fn require_string_or_null(value: &Value, field: &str, file: &str) -> Result<()> {
    match value.get(field) {
        Some(Value::String(text)) if !text.trim().is_empty() => Ok(()),
        Some(Value::Null) => Ok(()),
        None => bail!("{file}: missing field `{field}`"),
        _ => bail!("{file}: `{field}` must be string or null"),
    }
}

fn require_string_path(value: &Value, path: &[&str], file: &str) -> Result<()> {
    let Some(value) = value_at_path(value, path) else {
        bail!("{file}: missing `{}`", path.join("."));
    };
    if value.as_str().is_some_and(|text| !text.trim().is_empty()) {
        Ok(())
    } else {
        bail!("{file}: `{}` must be a non-empty string", path.join("."))
    }
}

fn require_bool(value: &Value, field: &str, file: &str) -> Result<()> {
    if value.get(field).and_then(Value::as_bool).is_some() {
        Ok(())
    } else {
        bail!("{file}: missing bool field `{field}`")
    }
}

fn require_bool_path(value: &Value, path: &[&str], file: &str) -> Result<()> {
    if value_at_path(value, path).and_then(Value::as_bool).is_some() {
        Ok(())
    } else {
        bail!("{file}: missing bool field `{}`", path.join("."))
    }
}

fn require_eq(value: &Value, field: &str, expected: &str, file: &str) -> Result<()> {
    match value.get(field).and_then(Value::as_str) {
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => bail!("{file}: `{field}` expected `{expected}`, got `{actual}`"),
        None => bail!("{file}: missing string field `{field}`"),
    }
}

fn require_eq_bool(value: &Value, field: &str, expected: bool, file: &str) -> Result<()> {
    match value.get(field).and_then(Value::as_bool) {
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => bail!("{file}: `{field}` expected `{expected}`, got `{actual}`"),
        None => bail!("{file}: missing bool field `{field}`"),
    }
}

fn require_eq_path(value: &Value, path: &[&str], expected: &str, file: &str) -> Result<()> {
    match value_at_path(value, path).and_then(Value::as_str) {
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => bail!("{file}: `{}` expected `{expected}`, got `{actual}`", path.join(".")),
        None => bail!("{file}: missing string field `{}`", path.join(".")),
    }
}

fn require_eq_path_bool(value: &Value, path: &[&str], expected: bool, file: &str) -> Result<()> {
    match value_at_path(value, path).and_then(Value::as_bool) {
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => bail!("{file}: `{}` expected `{expected}`, got `{actual}`", path.join(".")),
        None => bail!("{file}: missing bool field `{}`", path.join(".")),
    }
}

fn require_object(value: &Value, field: &str, file: &str) -> Result<()> {
    if value.get(field).and_then(Value::as_object).is_some() {
        Ok(())
    } else {
        bail!("{file}: missing object `{field}`")
    }
}

fn require_array(value: &Value, field: &str, file: &str) -> Result<()> {
    if value.get(field).and_then(Value::as_array).is_some_and(|array| !array.is_empty()) {
        Ok(())
    } else {
        bail!("{file}: missing non-empty array `{field}`")
    }
}

fn value_at_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for segment in path {
        current = current.get(*segment)?;
    }
    Some(current)
}

fn command_stdout(commands: &Value, key: &str) -> String {
    commands
        .get(key)
        .and_then(|value| value.get("stdout"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn command_success(commands: &Value, key: &str) -> bool {
    commands.get(key).and_then(|value| value.get("status")).and_then(Value::as_i64) == Some(0)
}

fn parse_after_colon(text: &str, label: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let trimmed = line.trim();
        let (key, value) = trimmed.split_once(':')?;
        (key.trim() == label).then(|| value.trim().to_string()).filter(|value| !value.is_empty())
    })
}

fn parse_core_count(text: &str) -> Option<u64> {
    let cores = parse_after_colon(text, "Total Number of Cores")
        .or_else(|| parse_after_colon(text, "Number of Cores"))?;
    cores.split_whitespace().next()?.parse().ok()
}

fn parse_sysctl_u64(text: &str, name: &str) -> Option<u64> {
    let value = parse_after_colon(text, name)?;
    value.split_whitespace().next()?.parse().ok()
}

fn parse_sw_vers(text: &str, name: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let mut parts = line.splitn(2, ':');
        let key = parts.next()?.trim();
        let value = parts.next()?.trim();
        (key == name).then(|| value.to_string()).filter(|value| !value.is_empty())
    })
}

fn chip_starts_with_apple(hardware: &str) -> bool {
    parse_after_colon(hardware, "Chip").is_some_and(|chip| chip.starts_with("Apple "))
}

fn first_nonempty_line(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let trimmed = line.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn apple_m4_bundle_paths_are_stable() {
        let root = Path::new("/workspace");
        let paths =
            BundlePaths::new(root, PathBuf::from("ci/hardware/apple-m4-mac-mini/2026-05-07"))
                .unwrap();
        assert_eq!(
            paths.file_abs("summary.json"),
            PathBuf::from("/workspace/ci/hardware/apple-m4-mac-mini/2026-05-07/summary.json")
        );
        assert_eq!(
            paths.file_logical("summary.json"),
            PathBuf::from("ci/hardware/apple-m4-mac-mini/2026-05-07/summary.json")
        );
    }

    #[test]
    fn receipt_checker_accepts_minimal_valid_bundle() {
        let dir = tempdir().unwrap();
        write_minimal_bundle(dir.path());
        let report = check_bundle(dir.path(), false).unwrap();
        assert_eq!(report.status, "pass");
        assert_eq!(report.receipts.len(), EXPECTED_RECEIPTS.len());
    }

    #[test]
    fn receipt_checker_rejects_qk256_claim() {
        let dir = tempdir().unwrap();
        write_minimal_bundle(dir.path());
        let path = dir.path().join("metal-i2s-parity.json");
        let mut receipt = read_json(&path).unwrap();
        receipt["bitnet"]["kernel_family"] = json!("qk256");
        write_json(path, &receipt).unwrap();
        assert_bundle_error_contains(dir.path(), "qk256");
    }

    #[test]
    fn receipt_checker_rejects_qk256_claim_hidden_in_field_name() {
        let dir = tempdir().unwrap();
        write_minimal_bundle(dir.path());
        let path = dir.path().join("strict-bitnet-cpu-neon-proof.json");
        let mut receipt = read_json(&path).unwrap();
        receipt["bitnet"]["qk256_supported"] = json!(true);
        write_json(path, &receipt).unwrap();
        assert_bundle_error_contains(dir.path(), "qk256");
    }

    #[test]
    fn receipt_checker_rejects_metal_fallback() {
        let dir = tempdir().unwrap();
        write_minimal_bundle(dir.path());
        let path = dir.path().join("metal-smoke.json");
        let mut receipt = read_json(&path).unwrap();
        receipt["selected_backend"] = json!(APPLE_M4_CPU_NEON);
        receipt["fallback_used"] = json!(true);
        receipt["fallback_reason"] = json!("Metal unavailable");
        write_json(path, &receipt).unwrap();
        assert_bundle_error_contains(dir.path(), "selected_backend");
    }

    #[test]
    fn receipt_checker_rejects_missing_fallback_used() {
        let dir = tempdir().unwrap();
        write_minimal_bundle(dir.path());
        let path = dir.path().join("metal-smoke.json");
        let mut receipt = read_json(&path).unwrap();
        receipt.as_object_mut().unwrap().remove("fallback_used");
        write_json(path, &receipt).unwrap();
        assert_bundle_error_contains(dir.path(), "fallback_used");
    }

    #[test]
    fn receipt_checker_rejects_backend_mismatch_without_fallback_reason() {
        let dir = tempdir().unwrap();
        write_minimal_bundle(dir.path());
        let path = dir.path().join("metal-probe.json");
        let mut receipt = read_json(&path).unwrap();
        receipt["selected_backend"] = json!(APPLE_M4_CPU_NEON);
        write_json(path, &receipt).unwrap();
        assert_bundle_error_contains(dir.path(), "without fallback_reason");
    }

    #[test]
    fn receipt_checker_rejects_mpsgraph_neural_engine_target() {
        let dir = tempdir().unwrap();
        write_minimal_bundle(dir.path());
        let path = dir.path().join("mpsgraph-smoke.json");
        let mut receipt = read_json(&path).unwrap();
        receipt["resolved_target"] = json!("ane");
        write_json(path, &receipt).unwrap();
        assert_bundle_error_contains(dir.path(), "Neural Engine");
    }

    #[test]
    fn receipt_checker_rejects_neural_engine_claim_hidden_in_field_name() {
        let dir = tempdir().unwrap();
        write_minimal_bundle(dir.path());
        let path = dir.path().join("mpsgraph-smoke.json");
        let mut receipt = read_json(&path).unwrap();
        receipt["neural_engine_used"] = json!(true);
        write_json(path, &receipt).unwrap();
        assert_bundle_error_contains(dir.path(), "Neural Engine");
    }

    #[test]
    fn receipt_checker_rejects_missing_bitnet_kernel_family() {
        let dir = tempdir().unwrap();
        write_minimal_bundle(dir.path());
        let path = dir.path().join("metal-i2s-parity.json");
        let mut receipt = read_json(&path).unwrap();
        receipt["bitnet"].as_object_mut().unwrap().remove("kernel_family");
        write_json(path, &receipt).unwrap();
        assert_bundle_error_contains(dir.path(), "kernel_family");
    }

    #[test]
    fn receipt_checker_rejects_missing_bitnet_execution_phase() {
        let dir = tempdir().unwrap();
        write_minimal_bundle(dir.path());
        let path = dir.path().join("strict-bitnet-cpu-neon-proof.json");
        let mut receipt = read_json(&path).unwrap();
        receipt["bitnet"].as_object_mut().unwrap().remove("execution_phase");
        write_json(path, &receipt).unwrap();
        assert_bundle_error_contains(dir.path(), "execution_phase");
    }

    #[test]
    fn receipt_checker_rejects_strict_cpu_receipt_without_tokenizer() {
        let dir = tempdir().unwrap();
        write_minimal_bundle(dir.path());
        let path = dir.path().join("strict-bitnet-cpu-neon-proof.json");
        let mut receipt = read_json(&path).unwrap();
        receipt["model"].as_object_mut().unwrap().remove("tokenizer");
        write_json(path, &receipt).unwrap();
        assert_bundle_error_contains(dir.path(), "tokenizer");
    }

    #[test]
    fn receipt_checker_allows_qk256_only_in_summary_not_proven() {
        let dir = tempdir().unwrap();
        write_minimal_bundle(dir.path());
        let path = dir.path().join("summary.json");
        let mut receipt = read_json(&path).unwrap();
        receipt["not_proven"] = json!(["QK256 on Apple Silicon is not proven by this bundle."]);
        write_json(path, &receipt).unwrap();
        check_bundle(dir.path(), false).unwrap();
    }

    fn write_minimal_bundle(dir: &Path) {
        fs::create_dir_all(dir).unwrap();
        write_json(
            dir.join("machine-profile.json"),
            &json!({
                "artifact_kind": "machine_profile",
                "artifact_path": "ci/hardware/apple-m4-mac-mini/2026-05-07/machine-profile.json",
                "requested_backend": null,
                "selected_backend": null,
                "runtime_api": "macos-system-profiler",
                "fallback_used": false,
                "machine": {"metal_visible": true}
            }),
        )
        .unwrap();
        write_json(
            dir.join("metal-probe.json"),
            &json!({
                "artifact_kind": "probe",
                "artifact_path": "ci/hardware/apple-m4-mac-mini/2026-05-07/metal-probe.json",
                "requested_backend": APPLE_M4_METAL,
                "selected_backend": APPLE_M4_METAL,
                "runtime_api": "metal",
                "fallback_used": false
            }),
        )
        .unwrap();
        write_json(dir.join("metal-smoke.json"), &metal_receipt("smoke")).unwrap();
        write_json(
            dir.join("metal-i2s-parity.json"),
            &json!({
                "artifact_kind": "parity",
                "artifact_path": "ci/hardware/apple-m4-mac-mini/2026-05-07/metal-i2s-parity.json",
                "requested_backend": APPLE_M4_METAL,
                "selected_backend": APPLE_M4_METAL,
                "runtime_api": "metal",
                "fallback_used": false,
                "kernel_id": "tiny_metal_i2s_parity",
                "bitnet": {"kernel_family": "i2_s", "execution_phase": "parity"},
                "layout": {
                    "consumes_packed_i2_s_directly": true,
                    "dequantizes_before_compute": false
                },
                "parity": {}
            }),
        )
        .unwrap();
        write_json(
            dir.join("mpsgraph-smoke.json"),
            &json!({
                "artifact_kind": "smoke",
                "artifact_path": "ci/hardware/apple-m4-mac-mini/2026-05-07/mpsgraph-smoke.json",
                "requested_backend": APPLE_M4_MPSGRAPH,
                "selected_backend": APPLE_M4_MPSGRAPH,
                "runtime_api": "mpsgraph",
                "fallback_used": false,
                "graph_id": "tiny_mpsgraph_matmul",
                "resolved_target": "unknown"
            }),
        )
        .unwrap();
        write_json(dir.join("strict-bitnet-cpu-neon-proof.json"), &strict_cpu_receipt()).unwrap();
        let mut profile = strict_cpu_receipt();
        profile["profile"] = json!({"requested": true, "allocation_audit": {"enabled": false}});
        write_json(dir.join("phase-profile.json"), &profile).unwrap();
        let mut allocation = strict_cpu_receipt();
        allocation["profile"] = json!({"requested": true, "allocation_audit": {"enabled": true}});
        write_json(dir.join("allocation-audit.json"), &allocation).unwrap();
        write_json(
            dir.join("summary.json"),
            &json!({
                "artifact_kind": "apple_m4_validation_summary",
                "artifact_path": "ci/hardware/apple-m4-mac-mini/2026-05-07/summary.json",
                "requested_backend": null,
                "selected_backend": null,
                "runtime_api": "xtask-apple-m4",
                "fallback_used": false,
                "proven": ["bundle receipts validate"],
                "not_proven": ["QK256 on Apple Silicon is not proven"]
            }),
        )
        .unwrap();
    }

    fn metal_receipt(kind: &str) -> Value {
        json!({
            "artifact_kind": kind,
            "artifact_path": "ci/hardware/apple-m4-mac-mini/2026-05-07/metal-smoke.json",
            "requested_backend": APPLE_M4_METAL,
            "selected_backend": APPLE_M4_METAL,
            "runtime_api": "metal",
            "fallback_used": false,
            "kernel_id": "tiny_metal_add_smoke",
            "result": "pass"
        })
    }

    fn strict_cpu_receipt() -> Value {
        json!({
            "artifact_kind": "strict_bitnet_cpu_reference",
            "artifact_path": "ci/hardware/apple-m4-mac-mini/2026-05-07/strict-bitnet-cpu-neon-proof.json",
            "requested_backend": APPLE_M4_CPU_NEON,
            "selected_backend": APPLE_M4_CPU_NEON,
            "runtime_api": "cpu",
            "fallback_used": false,
            "model": {
                "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
                "file": "ggml-model-i2_s.gguf",
                "tokenizer": "llama3"
            },
            "bitnet": {
                "kernel_family": "i2_s",
                "execution_phase": "decode"
            }
        })
    }

    fn assert_bundle_error_contains(dir: &Path, expected: &str) {
        let error = check_bundle(dir, false).unwrap_err().to_string();
        assert!(error.contains(expected), "expected error to contain `{expected}`, got:\n{error}");
    }
}
