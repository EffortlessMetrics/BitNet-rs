//! Interactive chat mode with REPL
//!
//! Provides a streaming chat interface with conversation history.

use anyhow::{Context, Result, bail};
use bitnet_repl_core::{
    BoundedHistory, ChatMetrics, ReplInput, copy_receipt_if_present, parse_repl_input,
};
use console::style;
use futures::StreamExt;
use humantime::format_duration;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, error};

use bitnet_inference::prompt_template::{ChatRole, ChatTurn};
use bitnet_inference::{InferenceEngine, TemplateType};
use tracing::info;

use super::dense_gguf_linear_parity::{
    DenseQwenCudaChatOptions, is_supported_dense_qwen_cuda_model_path, run_dense_qwen_cuda_chat,
};
use super::inference::InferenceCommand;
use super::receipts::{explain_receipt, print_compact_proof_summary};
use super::template_util::looks_like_llama3_chat;
use crate::config::CliConfig;
use crate::model_cache;

impl InferenceCommand {
    /// Run interactive chat mode with REPL
    pub async fn run_chat(&self, config: &CliConfig) -> Result<()> {
        // Setup environment (logging already initialized in main())
        self.setup_environment()?;

        let requested_backend_label = self.device.as_deref().unwrap_or(&config.default_device);
        if is_dense_qwen_cuda_chat_backend(requested_backend_label)
            && let Some(model) = resolve_dense_qwen_cuda_chat_model(self.model.as_deref())?
        {
            return self.run_dense_qwen_cuda_chat_path(model).await;
        }

        println!("{}", style("BitNet Interactive Chat").bold().cyan());
        println!("Loading model and tokenizer...");
        println!();

        // Load model and tokenizer
        let (mut engine, _tokenizer) = self.load_model_and_tokenizer(config).await?;

        // Resolve prompt template with Instruct as initial default, then promote to LLaMA-3 if appropriate
        let tt = self.resolve_template_type_with_default(TemplateType::Instruct)?;
        let template_type = if matches!(tt, TemplateType::Raw | TemplateType::Instruct) {
            // Try to extract metadata from model path for safer LLaMA-3 detection
            let mut tokenizer_name: Option<String> = None;
            let mut chat_template: Option<String> = None;

            if let Some(model_path) = &self.model {
                // Try to read GGUF metadata
                if let Ok(mmap) = bitnet_models::loader::MmapFile::open(model_path)
                    && let Ok(reader) = bitnet_models::GgufReader::new(mmap.as_slice())
                {
                    tokenizer_name = reader.get_string_metadata("general.name");
                    // Note: tokenizer.chat_template might not be present in all GGUFs
                    chat_template = reader.get_string_metadata("tokenizer.chat_template");
                }
            }

            if looks_like_llama3_chat(tokenizer_name.as_deref(), chat_template.as_deref()) {
                info!("auto-detect: promoting to LLaMA-3 chat template");
                TemplateType::Llama3Chat
            } else {
                tt
            }
        } else {
            tt
        };

        println!("{}", style("Chat ready!").bold().green());
        println!("Template: {}", style(format!("{:?}", template_type)).dim());
        println!("Commands: /help, /clear, /metrics, /exit");
        println!();

        // Conversation history: typed chat turns
        let mut conversation_history: BoundedHistory<ChatTurn> =
            BoundedHistory::new(self.chat_history_limit);
        let mut metrics = ChatMetrics::default();

        // Create generation config
        let gen_config = self.create_generation_config()?;

        // Detect if output is a TTY (for emoji/color support)
        let is_tty = io::stdout().is_terminal();

        loop {
            // Use fancy prompts for TTY, plain for pipes/redirects
            if is_tty {
                print!("{} ", style("you>").green().bold());
            } else {
                print!("you> ");
            }

            // Handle BrokenPipe gracefully
            if let Err(e) = io::stdout().flush() {
                if e.kind() == io::ErrorKind::BrokenPipe {
                    return Ok(());
                }
                return Err(e.into());
            }

            let mut input = String::new();
            match io::stdin().read_line(&mut input) {
                Ok(0) => break, // EOF (Ctrl+D)
                Ok(_) => {
                    let Some(parsed) = parse_repl_input(&input) else {
                        continue;
                    };

                    match parsed {
                        ReplInput::Exit => break,
                        ReplInput::Help => {
                            self.show_chat_help();
                            continue;
                        }
                        ReplInput::Clear => {
                            conversation_history.clear();
                            metrics = ChatMetrics::default();
                            println!("{}", style("Conversation cleared.").dim());
                            continue;
                        }
                        ReplInput::Metrics => {
                            self.show_chat_metrics(&metrics);
                            continue;
                        }
                        ReplInput::Message(line) => {
                            // Format prompt with conversation history using library render_chat()
                            // Build current turn history (all previous + current user input)
                            let mut current_history = conversation_history.to_vec();
                            current_history.push(ChatTurn::new(ChatRole::User, &line));

                            let formatted_prompt = template_type
                                .render_chat(&current_history, self.system_prompt.as_deref())?;

                            if self.verbose {
                                debug!("Formatted prompt:\n{}", formatted_prompt);
                            }

                            // Run streaming inference
                            let start_time = Instant::now();
                            if is_tty {
                                print!("{} ", style("assistant>").blue().bold());
                            } else {
                                print!("assistant> ");
                            }

                            // Handle BrokenPipe gracefully
                            if let Err(e) = io::stdout().flush() {
                                if e.kind() == io::ErrorKind::BrokenPipe {
                                    return Ok(());
                                }
                                return Err(e.into());
                            }

                            match self
                                .run_chat_inference(&mut engine, &formatted_prompt, &gen_config)
                                .await
                            {
                                Ok((response_text, token_count)) => {
                                    println!(); // Newline after streaming

                                    let elapsed = start_time.elapsed();
                                    let elapsed_ms = elapsed.as_millis() as u64;

                                    // Update metrics
                                    metrics.add_exchange(token_count, elapsed_ms);

                                    // Add to conversation history: user turn and assistant turn
                                    conversation_history.push(ChatTurn::new(ChatRole::User, &line));
                                    conversation_history
                                        .push(ChatTurn::new(ChatRole::Assistant, &response_text));

                                    // Copy receipt if directory specified
                                    if let Some(dir) = &self.emit_receipt_dir {
                                        let receipt_src = self.effective_receipt_path();
                                        match copy_receipt_if_present(receipt_src, dir) {
                                            Ok(Some(path)) => {
                                                debug!("Receipt saved: {}", path.display());
                                            }
                                            Ok(None) => {
                                                debug!("No receipt found to copy");
                                            }
                                            Err(e) => {
                                                debug!("Failed to copy receipt: {}", e);
                                            }
                                        }
                                    }

                                    // Show timing if metrics enabled
                                    if self.metrics {
                                        let tps = if elapsed.as_secs_f64() > 0.0 {
                                            token_count as f64 / elapsed.as_secs_f64()
                                        } else {
                                            0.0
                                        };
                                        println!(
                                            "  {} {} ({:.2} tok/s)",
                                            style("Time:").dim(),
                                            style(format_duration(elapsed)).dim(),
                                            tps
                                        );
                                    }

                                    println!(); // Extra line for readability
                                }
                                Err(e) => {
                                    println!();
                                    error!("Inference failed: {}", e);
                                    println!("{}", style(format!("Error: {}", e)).red());
                                    println!();
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to read input: {}", e);
                    break;
                }
            }
        }

        println!("\n{}", style("Goodbye!").cyan());
        Ok(())
    }

    /// Run streaming inference for a single chat turn
    async fn run_chat_inference(
        &self,
        engine: &mut InferenceEngine,
        prompt: &str,
        config: &super::inference::GenerationConfig,
    ) -> Result<(String, usize)> {
        // Clear kernel recorder before each turn to track per-turn kernels
        if let Some(recorder) = engine.kernel_recorder() {
            recorder.clear();
        }

        // Reset canonical token counter before generation
        engine.reset_decoded_tokens();

        // Get tokenizer for stop token ID resolution
        let tokenizer = engine.tokenizer();
        let engine_config = self.to_engine_config(config, Some(tokenizer.as_ref()));
        let mut stream = engine
            .generate_stream_with_config(prompt, &engine_config)
            .context("Failed to start streaming generation")?;

        let mut full_response = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Streaming chunk error")?;
            // Increment engine's canonical token counter
            engine.inc_decoded_tokens_by(chunk.token_ids.len());
            full_response.push_str(&chunk.text);
            print!("{}", chunk.text);

            // Handle BrokenPipe gracefully during streaming
            if let Err(e) = io::stdout().flush() {
                if e.kind() == io::ErrorKind::BrokenPipe {
                    debug!("BrokenPipe during streaming - client disconnected");
                    break;
                }
                return Err(e.into());
            }
        }

        // Write standard receipt to ci/inference.json using engine's canonical token count
        let tokens_generated = engine.decoded_token_count();
        if let Err(e) = self.write_receipt(engine, tokens_generated).await {
            debug!("Failed to write receipt: {}", e);
        }

        Ok((full_response, tokens_generated))
    }

    /// Show chat-specific help
    fn show_chat_help(&self) {
        println!("{}", style("Available commands:").bold());
        println!("  /help     - Show this help");
        println!("  /clear    - Clear conversation history");
        println!("  /metrics  - Show performance metrics");
        println!("  /exit     - Exit chat mode (also /quit)");
        println!();
        println!("{}", style("Keyboard shortcuts:").bold());
        println!("  Ctrl+C    - Exit chat");
        println!("  Ctrl+D    - Exit chat");
    }

    /// Show chat session metrics
    fn show_chat_metrics(&self, metrics: &ChatMetrics) {
        println!();
        println!("{}", style("Session Metrics:").bold());
        println!("  Exchanges: {}", style(metrics.num_exchanges.to_string()).cyan());
        println!("  Total tokens: {}", style(metrics.total_tokens_generated.to_string()).cyan());
        println!(
            "  Total time: {}",
            style(format_duration(std::time::Duration::from_millis(metrics.total_time_ms))).cyan()
        );
        println!(
            "  Average speed: {:.2} tok/s",
            style(format!("{:.2}", metrics.average_tps())).cyan()
        );
        println!();
    }
}

impl InferenceCommand {
    async fn run_dense_qwen_cuda_chat_path(&self, model: PathBuf) -> Result<()> {
        if self.tokenizer.is_some() {
            bail!(
                "dense Qwen CUDA chat uses contract-authoritative tokenizer resolution; do not pass --tokenizer"
            );
        }
        if self.system_prompt.as_ref().is_some_and(|value| !value.trim().is_empty()) {
            bail!(
                "dense Qwen CUDA chat is scoped to the contract deterministic prompt path; --system-prompt is not supported yet"
            );
        }
        if self.chat_template.is_some() || self.prompt_template != "auto" {
            bail!(
                "dense Qwen CUDA chat uses the contract-authoritative prompt template; do not pass --chat-template or --prompt-template"
            );
        }
        let prompts = collect_dense_qwen_cuda_chat_prompts(self)?;
        let max_new_tokens = if self.max_tokens == 512 { 8 } else { self.max_tokens };
        if !(5..=16).contains(&max_new_tokens) {
            bail!("dense Qwen CUDA chat is currently bounded to --max-tokens 5..=16");
        }
        let top_k = self.top_k.unwrap_or(10);
        if top_k == 0 {
            bail!("dense Qwen CUDA chat requires top-k evidence; use --top-k > 0");
        }
        if let Some(cuda_bin) = ensure_dense_qwen_chat_cuda_runtime_libraries_visible()? {
            debug!(
                "added CUDA Toolkit bin directory to process PATH for dense Qwen CUDA chat: {}",
                cuda_bin.display()
            );
        }

        println!("{}", style("BitNet CUDA Chat").bold().cyan());
        println!("{}", style("Running bounded dense Qwen CUDA chat proof...").dim());
        println!();

        let outcome = run_dense_qwen_cuda_chat(DenseQwenCudaChatOptions {
            model,
            prompts: prompts.clone(),
            max_new_tokens,
            top_k,
            device_index: 0,
            receipt_out: self.receipt_path.clone(),
        })
        .await?;

        for (prompt, answer) in prompts.iter().zip(outcome.answers.iter()) {
            println!("{} {}", style("you>").green().bold(), prompt);
            println!("{} {}", style("assistant>").blue().bold(), answer);
            println!();
        }

        let explanation = explain_receipt(&outcome.receipt_path, &outcome.receipt);
        print_compact_proof_summary(&explanation);
        Ok(())
    }
}

fn is_dense_qwen_cuda_chat_backend(requested_backend_label: &str) -> bool {
    matches!(
        requested_backend_label.trim().to_ascii_lowercase().as_str(),
        "cuda" | "nvidia-rtx-5070-ti-cuda"
    )
}

fn resolve_dense_qwen_cuda_chat_model(model: Option<&Path>) -> Result<Option<PathBuf>> {
    let Some(model) = model else {
        return Ok(None);
    };

    if let Some(cached) = model_cache::verified_dense_qwen_cuda_model_arg(model, None)? {
        return Ok(Some(cached.path));
    }

    if is_supported_dense_qwen_cuda_model_path(model) {
        return Ok(Some(model.to_path_buf()));
    }

    Ok(None)
}

fn collect_dense_qwen_cuda_chat_prompts(command: &InferenceCommand) -> Result<Vec<String>> {
    if command.prompt.is_some() && command.input_file.is_some() {
        bail!("dense Qwen CUDA chat accepts either --prompt or --input-file, not both");
    }

    let raw_prompts = if let Some(input_file) = &command.input_file {
        std::fs::read_to_string(input_file)
            .with_context(|| format!("failed to read chat prompts from {}", input_file.display()))?
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>()
    } else if let Some(prompt) = &command.prompt {
        prompt.lines().map(str::to_string).collect::<Vec<_>>()
    } else {
        collect_dense_qwen_cuda_chat_prompts_from_stdin()?
    };

    normalize_dense_qwen_cuda_chat_prompts(raw_prompts)
}

fn collect_dense_qwen_cuda_chat_prompts_from_stdin() -> Result<Vec<String>> {
    let is_tty = io::stdin().is_terminal();
    if is_tty {
        println!("{}", style("BitNet CUDA Chat").bold().cyan());
        println!("{}", style("Enter 2-4 turns, then /exit to run the bounded CUDA proof.").dim());
        println!();
    }

    let mut prompts = Vec::new();
    loop {
        if is_tty {
            print!("{} ", style("you>").green().bold());
            io::stdout().flush()?;
        }

        let mut input = String::new();
        match io::stdin().read_line(&mut input)? {
            0 => break,
            _ => match parse_repl_input(&input) {
                Some(ReplInput::Exit) => break,
                Some(ReplInput::Help) => {
                    if is_tty {
                        println!("Enter 2-4 user messages, then /exit to execute.");
                    }
                }
                Some(ReplInput::Clear) => {
                    prompts.clear();
                    if is_tty {
                        println!("{}", style("Pending chat turns cleared.").dim());
                    }
                }
                Some(ReplInput::Metrics) => {
                    if is_tty {
                        println!(
                            "{}",
                            style("Metrics are emitted after the bounded proof run.").dim()
                        );
                    }
                }
                Some(ReplInput::Message(message)) => {
                    prompts.push(message);
                    if prompts.len() == 4 {
                        break;
                    }
                }
                None => {}
            },
        }
    }
    Ok(prompts)
}

fn normalize_dense_qwen_cuda_chat_prompts(prompts: Vec<String>) -> Result<Vec<String>> {
    let prompts = prompts
        .into_iter()
        .map(|prompt| prompt.trim().to_string())
        .filter(|prompt| !prompt.is_empty())
        .collect::<Vec<_>>();
    if !(2..=4).contains(&prompts.len()) {
        bail!("dense Qwen CUDA chat requires 2..=4 non-empty user turns");
    }
    Ok(prompts)
}

fn ensure_dense_qwen_chat_cuda_runtime_libraries_visible() -> Result<Option<PathBuf>> {
    #[cfg(all(feature = "cuda", target_os = "windows"))]
    {
        ensure_windows_cuda_toolkit_bin_on_path()
    }

    #[cfg(not(all(feature = "cuda", target_os = "windows")))]
    {
        Ok(None)
    }
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn ensure_windows_cuda_toolkit_bin_on_path() -> Result<Option<PathBuf>> {
    if windows_cuda_runtime_libraries_visible_on_path() {
        return Ok(None);
    }

    let Some(cuda_bin) = discover_windows_cuda_toolkit_bin() else {
        return Ok(None);
    };
    prepend_process_path(&cuda_bin).with_context(|| {
        format!("failed to add CUDA Toolkit bin to PATH: {}", cuda_bin.display())
    })?;
    Ok(Some(cuda_bin))
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn discover_windows_cuda_toolkit_bin() -> Option<PathBuf> {
    discover_cuda_toolkit_bin_from_roots(windows_cuda_toolkit_search_roots())
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn discover_cuda_toolkit_bin_from_roots<I, P>(roots: I) -> Option<PathBuf>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    let mut candidates = Vec::new();
    for root in roots {
        collect_cuda_toolkit_bin_candidates(root.as_ref(), &mut candidates);
    }
    candidates.sort_by(|left, right| {
        cuda_bin_version_key(right).cmp(&cuda_bin_version_key(left)).then_with(|| left.cmp(right))
    });
    candidates.into_iter().find(|candidate| cuda_toolkit_bin_has_runtime_libraries(candidate))
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn collect_cuda_toolkit_bin_candidates(root: &Path, candidates: &mut Vec<PathBuf>) {
    candidates.push(root.to_path_buf());
    candidates.push(root.join("bin"));

    let Ok(children) = std::fs::read_dir(root) else {
        return;
    };
    for child in children.flatten() {
        let path = child.path();
        if path.is_dir()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with('v'))
        {
            candidates.push(path.join("bin"));
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn cuda_toolkit_bin_has_runtime_libraries(bin: &Path) -> bool {
    cuda_toolkit_bin_has_any(bin, WINDOWS_NVRTC_LIBRARY_NAMES)
        && cuda_toolkit_bin_has_any(bin, WINDOWS_CUDART_LIBRARY_NAMES)
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn cuda_toolkit_bin_has_any(bin: &Path, names: &[&str]) -> bool {
    names.iter().any(|name| bin.join(name).is_file())
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn windows_cuda_runtime_libraries_visible_on_path() -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|entry| cuda_toolkit_bin_has_runtime_libraries(&entry))
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn windows_cuda_toolkit_search_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for (key, value) in std::env::vars_os() {
        if key.to_string_lossy().to_ascii_uppercase().starts_with("CUDA_PATH") && !value.is_empty()
        {
            roots.push(PathBuf::from(value));
        }
    }

    for key in ["ProgramW6432", "ProgramFiles"] {
        if let Some(program_files) = std::env::var_os(key) {
            roots.push(
                PathBuf::from(program_files).join("NVIDIA GPU Computing Toolkit").join("CUDA"),
            );
        }
    }
    roots.push(PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"));

    dedupe_paths(roots)
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut deduped = Vec::<PathBuf>::new();
    for path in paths {
        if !deduped.iter().any(|existing| paths_equal_for_process_path(existing, &path)) {
            deduped.push(path);
        }
    }
    deduped
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn prepend_process_path(path: &Path) -> Result<()> {
    let current = std::env::var_os("PATH").unwrap_or_default();
    let mut entries = Vec::from([path.to_path_buf()]);
    entries.extend(
        std::env::split_paths(&current).filter(|entry| !paths_equal_for_process_path(entry, path)),
    );
    let updated_path = std::env::join_paths(entries)?;
    // SAFETY: The bounded dense Qwen CUDA chat path adjusts this process before
    // CUDA/NVRTC loading starts, so cudarc can discover Toolkit DLLs installed
    // in the standard Windows location.
    unsafe {
        std::env::set_var("PATH", updated_path);
    }
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn paths_equal_for_process_path(left: &Path, right: &Path) -> bool {
    left.to_string_lossy().eq_ignore_ascii_case(&right.to_string_lossy())
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn cuda_bin_version_key(path: &Path) -> (u32, u32, u32) {
    let version_name =
        path.parent().and_then(|parent| parent.file_name()).and_then(|name| name.to_str());
    parse_cuda_version_name(version_name.unwrap_or_default())
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn parse_cuda_version_name(name: &str) -> (u32, u32, u32) {
    let Some(rest) = name.strip_prefix('v') else {
        return (0, 0, 0);
    };
    let mut parts = rest.split('.');
    let major = parts.next().and_then(|value| value.parse().ok()).unwrap_or_default();
    let minor = parts.next().and_then(|value| value.parse().ok()).unwrap_or_default();
    let patch = parts.next().and_then(|value| value.parse().ok()).unwrap_or_default();
    (major, minor, patch)
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
const WINDOWS_NVRTC_LIBRARY_NAMES: &[&str] =
    &["nvrtc64_120_0.dll", "nvrtc64_120.dll", "nvrtc64_12.dll", "nvrtc64.dll", "nvrtc.dll"];

#[cfg(all(feature = "cuda", target_os = "windows"))]
const WINDOWS_CUDART_LIBRARY_NAMES: &[&str] =
    &["cudart64_120.dll", "cudart64_12.dll", "cudart64.dll", "cudart.dll"];

#[cfg(test)]
mod dense_cuda_chat_tests {
    use super::*;

    #[test]
    fn dense_qwen_chat_backend_accepts_cuda_aliases_only() {
        assert!(is_dense_qwen_cuda_chat_backend("cuda"));
        assert!(is_dense_qwen_cuda_chat_backend("nvidia-rtx-5070-ti-cuda"));
        assert!(!is_dense_qwen_cuda_chat_backend("cpu"));
        assert!(!is_dense_qwen_cuda_chat_backend("apple-m4-cpu-neon"));
    }

    #[test]
    fn dense_qwen_chat_prompt_normalization_requires_two_to_four_turns() {
        let prompts = normalize_dense_qwen_cuda_chat_prompts(vec![
            " What is 2+2? ".to_string(),
            "".to_string(),
            "Name the answer.".to_string(),
        ])
        .unwrap();

        assert_eq!(prompts, vec!["What is 2+2?", "Name the answer."]);
        assert!(normalize_dense_qwen_cuda_chat_prompts(vec!["one".to_string()]).is_err());
        assert!(
            normalize_dense_qwen_cuda_chat_prompts(vec![
                "one".to_string(),
                "two".to_string(),
                "three".to_string(),
                "four".to_string(),
                "five".to_string(),
            ])
            .is_err()
        );
    }
}
