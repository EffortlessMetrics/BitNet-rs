//! Interactive chat mode with REPL
//!
//! Provides a streaming chat interface with conversation history.

use anyhow::{Context, Result};
use console::style;
use futures::StreamExt;
use humantime::format_duration;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, error};

use bitnet_inference::{InferenceEngine, TemplateType};

use super::inference::InferenceCommand;
use crate::config::CliConfig;

/// Performance metrics for chat session
#[derive(Debug, Default)]
struct ChatMetrics {
    total_tokens_generated: usize,
    total_time_ms: u64,
    num_exchanges: usize,
}

/// Copy receipt from ci/inference.json to timestamped file in the specified directory
fn copy_receipt_if_present(dir: &Path) -> Result<Option<PathBuf>> {
    use std::fs;

    let src = Path::new("ci").join("inference.json");
    if !src.exists() {
        return Ok(None);
    }

    fs::create_dir_all(dir)?;
    let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let dst = dir.join(format!("chat-{}.json", ts));
    fs::copy(&src, &dst)?;
    Ok(Some(dst))
}

impl ChatMetrics {
    fn add_exchange(&mut self, tokens: usize, elapsed_ms: u64) {
        self.total_tokens_generated += tokens;
        self.total_time_ms += elapsed_ms;
        self.num_exchanges += 1;
    }

    fn average_tps(&self) -> f64 {
        if self.total_time_ms > 0 {
            (self.total_tokens_generated as f64) / (self.total_time_ms as f64 / 1000.0)
        } else {
            0.0
        }
    }
}

impl InferenceCommand {
    /// Run interactive chat mode with REPL
    pub async fn run_chat(&self, config: &CliConfig) -> Result<()> {
        // Setup environment and logging
        self.setup_environment()?;
        self.setup_logging(config)?;

        println!("{}", style("BitNet Interactive Chat").bold().cyan());
        println!("Loading model and tokenizer...");
        println!();

        // Load model and tokenizer
        let (mut engine, _tokenizer) = self.load_model_and_tokenizer(config).await?;

        // Resolve prompt template using same logic as `run` subcommand
        let template_type = self.resolve_template_type()?;

        println!("{}", style("Chat ready!").bold().green());
        println!("Template: {}", style(format!("{:?}", template_type)).dim());
        println!("Commands: /help, /clear, /metrics, /exit");
        println!();

        // Conversation history: (user_msg, assistant_msg)
        let mut conversation_history: Vec<(String, String)> = Vec::new();
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
                    let line = input.trim();

                    if line.is_empty() {
                        continue;
                    }

                    // Handle commands
                    match line {
                        "/exit" | "/quit" => break,
                        "/help" => {
                            self.show_chat_help();
                            continue;
                        }
                        "/clear" => {
                            conversation_history.clear();
                            metrics = ChatMetrics::default();
                            println!("{}", style("Conversation cleared.").dim());
                            continue;
                        }
                        "/metrics" => {
                            self.show_chat_metrics(&metrics);
                            continue;
                        }
                        _ => {}
                    }

                    // Format prompt with conversation history
                    let formatted_prompt =
                        self.format_chat_turn(&template_type, &conversation_history, line)?;

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

                    match self.run_chat_inference(&mut engine, &formatted_prompt, &gen_config).await
                    {
                        Ok((response_text, token_count)) => {
                            println!(); // Newline after streaming

                            let elapsed = start_time.elapsed();
                            let elapsed_ms = elapsed.as_millis() as u64;

                            // Update metrics
                            metrics.add_exchange(token_count, elapsed_ms);

                            // Add to conversation history
                            conversation_history.push((line.to_string(), response_text));

                            // Copy receipt if directory specified
                            if let Some(dir) = &self.emit_receipt_dir {
                                match copy_receipt_if_present(dir) {
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
        let engine_config = self.to_engine_config(config);
        let mut stream = engine
            .generate_stream_with_config(prompt, &engine_config)
            .context("Failed to start streaming generation")?;

        let mut full_response = String::new();
        let mut token_count = 0usize;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Streaming chunk error")?;
            token_count += chunk.token_ids.len();
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

        // Write standard receipt to ci/inference.json
        if let Err(e) = self.write_receipt(engine, token_count).await {
            debug!("Failed to write receipt: {}", e);
        }

        Ok((full_response, token_count))
    }

    /// Format a chat turn with conversation history using the template
    fn format_chat_turn(
        &self,
        template: &TemplateType,
        history: &[(String, String)],
        current_input: &str,
    ) -> Result<String> {
        // Build conversation context
        let mut full_context = String::new();

        // For chat templates, we need to format the entire conversation
        match template {
            TemplateType::Llama3Chat => {
                // LLaMA-3 chat format
                full_context.push_str("<|begin_of_text|>");

                // Add system prompt if provided
                if let Some(system_prompt) = &self.system_prompt {
                    full_context.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                    full_context.push_str(system_prompt);
                    full_context.push_str("<|eot_id|>");
                }

                // Add conversation history
                for (user_msg, assistant_msg) in history {
                    full_context.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
                    full_context.push_str(user_msg);
                    full_context.push_str("<|eot_id|>");
                    full_context.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                    full_context.push_str(assistant_msg);
                    full_context.push_str("<|eot_id|>");
                }

                // Add current turn
                full_context.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
                full_context.push_str(current_input);
                full_context.push_str("<|eot_id|>");
                full_context.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            TemplateType::Instruct => {
                // Instruct format
                if let Some(system_prompt) = &self.system_prompt {
                    full_context.push_str(&format!("System: {}\n\n", system_prompt));
                }

                for (user_msg, assistant_msg) in history {
                    full_context.push_str(&format!("Q: {}\nA: {}\n\n", user_msg, assistant_msg));
                }

                full_context.push_str(&format!("Q: {}\nA:", current_input));
            }
            TemplateType::Raw => {
                // Raw format - simple concatenation
                for (user_msg, assistant_msg) in history {
                    full_context.push_str(user_msg);
                    full_context.push('\n');
                    full_context.push_str(assistant_msg);
                    full_context.push('\n');
                }
                full_context.push_str(current_input);
            }
        }

        Ok(full_context)
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
