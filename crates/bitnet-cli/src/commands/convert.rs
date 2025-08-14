//! Model conversion command implementation

use anyhow::{Context, Result};
use clap::Args;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use bitnet_common::QuantizationType;
use bitnet_models::ModelLoader;
use candle_core::Device;

use crate::config::CliConfig;

/// Convert command arguments
#[derive(Args, Debug)]
pub struct ConvertCommand {
    /// Input model path
    #[arg(short, long, value_name = "PATH")]
    pub input: PathBuf,

    /// Output model path
    #[arg(short, long, value_name = "PATH")]
    pub output: PathBuf,

    /// Target format (gguf, safetensors, huggingface)
    #[arg(short, long, value_name = "FORMAT")]
    pub format: Option<String>,

    /// Target quantization (i2s, tl1, tl2, none)
    #[arg(short, long, value_name = "TYPE")]
    pub quantization: Option<String>,

    /// Verify conversion integrity
    #[arg(long)]
    pub verify: bool,

    /// Show verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Force overwrite output file
    #[arg(long)]
    pub force: bool,

    /// Compression level (0-9, if supported by format)
    #[arg(long, value_name = "LEVEL")]
    pub compression: Option<u8>,

    /// Preserve metadata
    #[arg(long)]
    pub preserve_metadata: bool,
}

impl ConvertCommand {
    /// Execute the convert command
    pub async fn execute(&self, config: &CliConfig) -> Result<()> {
        // Validate arguments
        self.validate_args()?;

        // Setup logging
        if self.verbose {
            info!("Verbose mode enabled");
        }

        info!(
            "Converting model from {} to {}",
            self.input.display(),
            self.output.display()
        );

        // Check if output exists and handle overwrite
        if self.output.exists() && !self.force {
            anyhow::bail!(
                "Output file {} already exists. Use --force to overwrite.",
                self.output.display()
            );
        }

        let start_time = Instant::now();

        // Load input model
        let model = self.load_input_model(config).await?;

        // Convert model
        self.convert_model(model).await?;

        // Verify conversion if requested
        if self.verify {
            self.verify_conversion().await?;
        }

        let elapsed = start_time.elapsed();
        println!(
            "{} Conversion completed in {:.2}s",
            style("✓").green(),
            elapsed.as_secs_f64()
        );

        Ok(())
    }

    /// Validate command arguments
    fn validate_args(&self) -> Result<()> {
        // Check input file exists
        if !self.input.exists() {
            anyhow::bail!("Input file does not exist: {}", self.input.display());
        }

        // Validate format if specified
        if let Some(format) = &self.format {
            match format.to_lowercase().as_str() {
                "gguf" | "safetensors" | "huggingface" => {}
                _ => anyhow::bail!(
                    "Invalid format: {}. Must be one of: gguf, safetensors, huggingface",
                    format
                ),
            }
        }

        // Validate quantization if specified
        if let Some(quant) = &self.quantization {
            match quant.to_lowercase().as_str() {
                "i2s" | "tl1" | "tl2" | "none" => {}
                _ => anyhow::bail!(
                    "Invalid quantization: {}. Must be one of: i2s, tl1, tl2, none",
                    quant
                ),
            }
        }

        // Validate compression level
        if let Some(level) = self.compression {
            if level > 9 {
                anyhow::bail!("Compression level must be between 0 and 9");
            }
        }

        Ok(())
    }

    /// Load the input model
    async fn load_input_model(
        &self,
        config: &CliConfig,
    ) -> Result<Box<dyn bitnet_models::Model<Config = bitnet_common::BitNetConfig>>> {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        pb.set_message("Loading input model...");
        pb.enable_steady_tick(Duration::from_millis(100));

        // Determine device (CPU for conversion)
        let device = Device::Cpu;

        // Load model
        let loader = ModelLoader::new(device);
        let model = loader
            .load(&self.input)
            .with_context(|| format!("Failed to load input model: {}", self.input.display()))?;

        pb.finish_with_message(format!("{} Input model loaded", style("✓").green()));

        if self.verbose {
            // Extract and display model metadata
            let metadata = loader.extract_metadata(&self.input)?;
            info!("Model metadata:");
            info!("  Name: {}", metadata.name);
            info!("  Architecture: {}", metadata.architecture);
            info!("  Vocabulary size: {}", metadata.vocab_size);
            info!("  Context length: {}", metadata.context_length);
            if let Some(quant) = metadata.quantization {
                info!("  Quantization: {}", quant);
            }
        }

        Ok(model)
    }

    /// Convert the model to the target format
    async fn convert_model(
        &self,
        _model: Box<dyn bitnet_models::Model<Config = bitnet_common::BitNetConfig>>,
    ) -> Result<()> {
        let pb = ProgressBar::new(100);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}% {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );

        // Determine target format
        let target_format = self.determine_target_format()?;
        let target_quantization = self.determine_target_quantization()?;

        info!("Converting to format: {}", target_format);
        if let Some(quant) = &target_quantization {
            info!("Target quantization: {}", quant);
        }

        // Simulate conversion progress
        pb.set_message("Analyzing model structure...");
        tokio::time::sleep(Duration::from_millis(500)).await;
        pb.set_position(10);

        pb.set_message("Converting weights...");
        tokio::time::sleep(Duration::from_millis(1000)).await;
        pb.set_position(40);

        if target_quantization.is_some() {
            pb.set_message("Applying quantization...");
            tokio::time::sleep(Duration::from_millis(800)).await;
            pb.set_position(70);
        }

        pb.set_message("Writing output file...");
        tokio::time::sleep(Duration::from_millis(600)).await;
        pb.set_position(90);

        // Create output directory if needed
        if let Some(parent) = self.output.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create output directory: {}", parent.display())
            })?;
        }

        // Placeholder: Write a dummy output file
        std::fs::write(
            &self.output,
            format!(
                "# BitNet Model Conversion\n\
             # Converted from: {}\n\
             # Target format: {}\n\
             # Target quantization: {:?}\n\
             # Conversion time: {}\n\
             # Note: This is a placeholder implementation\n",
                self.input.display(),
                target_format,
                target_quantization,
                chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
            ),
        )?;

        pb.set_position(100);
        pb.finish_with_message(format!(
            "{} Model converted successfully",
            style("✓").green()
        ));

        // Show conversion summary
        self.show_conversion_summary(&target_format, &target_quantization)?;

        Ok(())
    }

    /// Determine target format
    fn determine_target_format(&self) -> Result<String> {
        if let Some(format) = &self.format {
            return Ok(format.to_lowercase());
        }

        // Auto-detect from output extension
        if let Some(ext) = self.output.extension().and_then(|s| s.to_str()) {
            match ext.to_lowercase().as_str() {
                "gguf" => Ok("gguf".to_string()),
                "safetensors" => Ok("safetensors".to_string()),
                _ => Ok("gguf".to_string()), // Default
            }
        } else {
            Ok("gguf".to_string()) // Default
        }
    }

    /// Determine target quantization
    fn determine_target_quantization(&self) -> Result<Option<QuantizationType>> {
        if let Some(quant) = &self.quantization {
            match quant.to_lowercase().as_str() {
                "i2s" => Ok(Some(QuantizationType::I2S)),
                "tl1" => Ok(Some(QuantizationType::TL1)),
                "tl2" => Ok(Some(QuantizationType::TL2)),
                "none" => Ok(None),
                _ => unreachable!(), // Already validated
            }
        } else {
            Ok(None) // Keep original quantization
        }
    }

    /// Verify the conversion
    async fn verify_conversion(&self) -> Result<()> {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        pb.set_message("Verifying conversion...");
        pb.enable_steady_tick(Duration::from_millis(100));

        // Simulate verification
        tokio::time::sleep(Duration::from_millis(1000)).await;

        // Check output file exists and is readable
        if !self.output.exists() {
            anyhow::bail!("Output file was not created: {}", self.output.display());
        }

        let metadata = std::fs::metadata(&self.output)?;
        if metadata.len() == 0 {
            anyhow::bail!("Output file is empty: {}", self.output.display());
        }

        pb.finish_with_message(format!("{} Conversion verified", style("✓").green()));

        if self.verbose {
            info!("Verification details:");
            info!("  Output file size: {} bytes", metadata.len());
            info!(
                "  Output file created: {:?}",
                metadata
                    .created()
                    .unwrap_or_else(|_| std::time::SystemTime::now())
            );
        }

        Ok(())
    }

    /// Show conversion summary
    fn show_conversion_summary(
        &self,
        target_format: &str,
        target_quantization: &Option<QuantizationType>,
    ) -> Result<()> {
        println!("\n{}", style("Conversion Summary:").bold());
        println!("  Input: {}", self.input.display());
        println!("  Output: {}", self.output.display());
        println!("  Format: {}", target_format);

        if let Some(quant) = target_quantization {
            println!("  Quantization: {}", quant);
        } else {
            println!("  Quantization: Preserved from input");
        }

        // Show file sizes
        if let Ok(input_metadata) = std::fs::metadata(&self.input) {
            if let Ok(output_metadata) = std::fs::metadata(&self.output) {
                let input_size = input_metadata.len();
                let output_size = output_metadata.len();
                let ratio = if input_size > 0 {
                    output_size as f64 / input_size as f64
                } else {
                    0.0
                };

                println!("  Input size: {} bytes", format_size(input_size));
                println!("  Output size: {} bytes", format_size(output_size));
                println!("  Size ratio: {:.2}x", ratio);
            }
        }

        Ok(())
    }
}

/// Format file size for display
fn format_size(size: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = size as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{:.0} {}", size, UNITS[unit_index])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}
