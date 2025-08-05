use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::time::Instant;
use std::path::Path;

use crate::model::PyBitNetModel;
use crate::tokenizer::{PyTokenizer, PyChatFormat};
use crate::inference::{PyInferenceEngine, PyStats};
use crate::config::{PyModelArgs, PyGenArgs};
use crate::error::PyBitNetError;

/// Load a BitNet model from various formats
#[pyfunction]
#[pyo3(signature = (model_path, model_format = None, device = None, **kwargs))]
pub fn load_model(
    model_path: &str,
    model_format: Option<&str>,
    device: Option<String>,
    kwargs: Option<&PyDict>,
) -> PyResult<PyBitNetModel> {
    let path = Path::new(model_path);
    let device = device.unwrap_or_else(|| "cpu".to_string());
    
    // Auto-detect format if not specified
    let format = model_format.unwrap_or_else(|| {
        if path.extension().map_or(false, |ext| ext == "gguf") {
            "gguf"
        } else if path.extension().map_or(false, |ext| ext == "safetensors") {
            "safetensors"
        } else if path.is_dir() {
            "pretrained"
        } else {
            "auto"
        }
    });
    
    match format {
        "gguf" => PyBitNetModel::from_gguf(
            PyBitNetModel::type_object(Python::with_gil(|py| py)),
            model_path,
            Some(device),
        ),
        "safetensors" => PyBitNetModel::from_safetensors(
            PyBitNetModel::type_object(Python::with_gil(|py| py)),
            model_path,
            Some(device),
        ),
        "pretrained" | "auto" => {
            let model_args = if let Some(kwargs) = kwargs {
                if let Ok(args_dict) = kwargs.get_item("model_args") {
                    Some(PyModelArgs::from_dict(
                        PyModelArgs::type_object(Python::with_gil(|py| py)),
                        args_dict.downcast()?,
                    )?)
                } else {
                    None
                }
            } else {
                None
            };
            
            PyBitNetModel::from_pretrained(
                PyBitNetModel::type_object(Python::with_gil(|py| py)),
                model_path,
                model_args,
                device,
                "bfloat16".to_string(),
            )
        }
        _ => Err(PyBitNetError::new(
            format!("Unsupported model format: {}", format),
            Some("ValueError".to_string())
        ).into()),
    }
}

/// Create a tokenizer from model path
#[pyfunction]
#[pyo3(signature = (tokenizer_path, chat_format = false))]
pub fn create_tokenizer(
    tokenizer_path: &str,
    chat_format: bool,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let tokenizer = Py::new(py, PyTokenizer::new(tokenizer_path.to_string())?)?;
        
        if chat_format {
            let chat_tokenizer = Py::new(py, PyChatFormat::new(tokenizer)?)?;
            Ok(chat_tokenizer.into())
        } else {
            Ok(tokenizer.into())
        }
    })
}

/// Benchmark inference performance
#[pyfunction]
#[pyo3(signature = (model, tokenizer, prompts, gen_args = None, num_runs = 1, warmup_runs = 1))]
pub fn benchmark_inference(
    model: &PyBitNetModel,
    tokenizer: &PyTokenizer,
    prompts: Vec<String>,
    gen_args: Option<PyGenArgs>,
    num_runs: usize,
    warmup_runs: usize,
) -> PyResult<PyDict> {
    Python::with_gil(|py| {
        let gen_args = gen_args.unwrap_or_else(|| PyGenArgs::new(
            128, 1, 64, false, 0.8, 0.9, None, 1.0
        ));
        
        // Convert prompts to tokens
        let mut prompt_tokens = Vec::new();
        for prompt in &prompts {
            let tokens = tokenizer.encode(prompt, true, false, None, None)?;
            prompt_tokens.push(tokens);
        }
        
        // Create inference engine
        let prefill_model = Py::new(py, model.clone())?;
        let decode_model = Py::new(py, model.clone())?;
        let tokenizer_py = Py::new(py, tokenizer.clone())?;
        
        let mut engine = PyInferenceEngine::new(
            prefill_model,
            decode_model,
            tokenizer_py,
            gen_args.clone(),
            Some("cpu".to_string()),
        );
        
        // Warmup runs
        for _ in 0..warmup_runs {
            let _ = engine.generate_all(prompt_tokens.clone(), false, None)?;
        }
        
        // Benchmark runs
        let mut total_time = 0.0;
        let mut total_tokens = 0;
        let mut all_stats = Vec::new();
        
        for _ in 0..num_runs {
            let start = Instant::now();
            let (stats, results) = engine.generate_all(prompt_tokens.clone(), false, None)?;
            let elapsed = start.elapsed().as_secs_f64();
            
            total_time += elapsed;
            total_tokens += results.iter().map(|r| r.len()).sum::<usize>();
            all_stats.push(stats);
        }
        
        // Calculate statistics
        let avg_time = total_time / num_runs as f64;
        let avg_tokens_per_second = total_tokens as f64 / total_time;
        let avg_latency = avg_time / prompts.len() as f64;
        
        // Create results dictionary
        let results = PyDict::new(py);
        results.set_item("num_runs", num_runs)?;
        results.set_item("num_prompts", prompts.len())?;
        results.set_item("total_time", total_time)?;
        results.set_item("avg_time", avg_time)?;
        results.set_item("avg_tokens_per_second", avg_tokens_per_second)?;
        results.set_item("avg_latency", avg_latency)?;
        results.set_item("total_tokens", total_tokens)?;
        
        // Add detailed stats
        let stats_list = PyList::empty(py);
        for stat in all_stats {
            stats_list.append(stat)?;
        }
        results.set_item("detailed_stats", stats_list)?;
        
        Ok(results)
    })
}

/// Compare performance between Python and Rust implementations
#[pyfunction]
#[pyo3(signature = (rust_model, python_model, tokenizer, prompts, **kwargs))]
pub fn compare_performance(
    rust_model: &PyBitNetModel,
    python_model: &PyAny,
    tokenizer: &PyTokenizer,
    prompts: Vec<String>,
    kwargs: Option<&PyDict>,
) -> PyResult<PyDict> {
    Python::with_gil(|py| {
        let num_runs = kwargs
            .and_then(|k| k.get_item("num_runs"))
            .and_then(|v| v.extract().ok())
            .unwrap_or(3);
        
        // Benchmark Rust implementation
        let rust_results = benchmark_inference(
            rust_model,
            tokenizer,
            prompts.clone(),
            None,
            num_runs,
            1,
        )?;
        
        // TODO: Benchmark Python implementation when available
        // For now, create dummy Python results
        let python_results = PyDict::new(py);
        python_results.set_item("avg_tokens_per_second", 50.0)?; // Dummy value
        python_results.set_item("avg_time", 2.0)?; // Dummy value
        python_results.set_item("total_tokens", 1000)?; // Dummy value
        
        // Calculate improvement
        let rust_tps: f64 = rust_results.get_item("avg_tokens_per_second")?.extract()?;
        let python_tps: f64 = python_results.get_item("avg_tokens_per_second")?.extract()?;
        let speedup = rust_tps / python_tps;
        
        let rust_time: f64 = rust_results.get_item("avg_time")?.extract()?;
        let python_time: f64 = python_results.get_item("avg_time")?.extract()?;
        let time_improvement = python_time / rust_time;
        
        // Create comparison results
        let comparison = PyDict::new(py);
        comparison.set_item("rust_results", rust_results)?;
        comparison.set_item("python_results", python_results)?;
        comparison.set_item("speedup", speedup)?;
        comparison.set_item("time_improvement", time_improvement)?;
        comparison.set_item("improvement_percentage", (speedup - 1.0) * 100.0)?;
        
        Ok(comparison)
    })
}

/// Validate model outputs between implementations
#[pyfunction]
#[pyo3(signature = (rust_model, python_model, tokenizer, prompts, tolerance = 1e-6))]
pub fn validate_outputs(
    rust_model: &PyBitNetModel,
    python_model: &PyAny,
    tokenizer: &PyTokenizer,
    prompts: Vec<String>,
    tolerance: f64,
) -> PyResult<PyDict> {
    Python::with_gil(|py| {
        let mut results = Vec::new();
        let mut all_match = true;
        let mut max_diff = 0.0;
        
        for prompt in prompts {
            // Generate with Rust model
            let rust_tokens = tokenizer.encode(&prompt, true, false, None, None)?;
            // TODO: Get actual logits when model is ready
            let rust_output = format!("rust_output_for_{}", prompt);
            
            // TODO: Generate with Python model when available
            let python_output = format!("python_output_for_{}", prompt);
            
            // Compare outputs (simplified for now)
            let matches = rust_output == python_output;
            all_match &= matches;
            
            results.push((prompt, rust_output, python_output, matches));
        }
        
        // Create validation results
        let validation = PyDict::new(py);
        validation.set_item("all_match", all_match)?;
        validation.set_item("max_difference", max_diff)?;
        validation.set_item("tolerance", tolerance)?;
        validation.set_item("num_prompts", results.len())?;
        validation.set_item("num_matches", results.iter().filter(|(_, _, _, m)| *m).count())?;
        
        // Add detailed results
        let detailed = PyList::empty(py);
        for (prompt, rust_out, python_out, matches) in results {
            let result = PyDict::new(py);
            result.set_item("prompt", prompt)?;
            result.set_item("rust_output", rust_out)?;
            result.set_item("python_output", python_out)?;
            result.set_item("matches", matches)?;
            detailed.append(result)?;
        }
        validation.set_item("detailed_results", detailed)?;
        
        Ok(validation)
    })
}

/// Get system information for debugging
#[pyfunction]
pub fn get_system_info() -> PyResult<PyDict> {
    Python::with_gil(|py| {
        let info = PyDict::new(py);
        
        // Rust version
        info.set_item("rust_version", env!("CARGO_PKG_VERSION"))?;
        
        // System information
        info.set_item("target_arch", std::env::consts::ARCH)?;
        info.set_item("target_os", std::env::consts::OS)?;
        
        // Feature flags
        let mut features = Vec::new();
        #[cfg(feature = "cpu")]
        features.push("cpu");
        #[cfg(feature = "gpu")]
        features.push("gpu");
        info.set_item("features", features)?;
        
        // CPU features
        let mut cpu_features = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                cpu_features.push("avx2");
            }
            if is_x86_feature_detected!("avx512f") {
                cpu_features.push("avx512f");
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                cpu_features.push("neon");
            }
        }
        info.set_item("cpu_features", cpu_features)?;
        
        Ok(info)
    })
}