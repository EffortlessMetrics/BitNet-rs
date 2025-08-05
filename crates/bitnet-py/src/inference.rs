use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_asyncio::tokio::future_into_py;
use numpy::{PyArray1, PyArray2};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::model::PyBitNetModel;
use crate::tokenizer::PyTokenizer;
use crate::config::{PyGenArgs, PyInferenceConfig};
use crate::error::PyBitNetError;

/// Statistics tracking for inference performance
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyStats {
    #[pyo3(get)]
    pub total_tokens: usize,
    #[pyo3(get)]
    pub total_time: f64,
    #[pyo3(get)]
    pub tokens_per_second: f64,
    #[pyo3(get)]
    pub prefill_time: f64,
    #[pyo3(get)]
    pub decode_time: f64,
    #[pyo3(get)]
    pub memory_used: f64,
}

#[pymethods]
impl PyStats {
    #[new]
    fn new() -> Self {
        Self {
            total_tokens: 0,
            total_time: 0.0,
            tokens_per_second: 0.0,
            prefill_time: 0.0,
            decode_time: 0.0,
            memory_used: 0.0,
        }
    }
    
    fn show(&self) -> String {
        format!(
            "Stats(tokens={}, time={:.2}s, tokens/s={:.2}, prefill={:.2}s, decode={:.2}s, memory={:.2}GB)",
            self.total_tokens, self.total_time, self.tokens_per_second,
            self.prefill_time, self.decode_time, self.memory_used
        )
    }
    
    fn __repr__(&self) -> String {
        self.show()
    }
}

/// Inference engine matching the existing FastGen API
#[pyclass]
pub struct PyInferenceEngine {
    prefill_model: Py<PyBitNetModel>,
    decode_model: Py<PyBitNetModel>,
    tokenizer: Py<PyTokenizer>,
    gen_args: PyGenArgs,
    device: String,
    cache: Option<Py<PyList>>,
}

#[pymethods]
impl PyInferenceEngine {
    #[new]
    fn new(
        prefill_model: Py<PyBitNetModel>,
        decode_model: Py<PyBitNetModel>,
        tokenizer: Py<PyTokenizer>,
        gen_args: PyGenArgs,
        device: Option<String>,
    ) -> Self {
        let device = device.unwrap_or_else(|| "cpu".to_string());
        
        Self {
            prefill_model,
            decode_model,
            tokenizer,
            gen_args,
            device,
            cache: None,
        }
    }
    
    /// Build inference engine from checkpoint directory (matching existing API)
    #[classmethod]
    #[pyo3(signature = (ckpt_dir, gen_args, device, tokenizer_path = None, num_layers = 13, use_full_vocab = false))]
    fn build(
        _cls: &PyType,
        ckpt_dir: String,
        gen_args: PyGenArgs,
        device: String,
        tokenizer_path: Option<String>,
        num_layers: usize,
        use_full_vocab: bool,
    ) -> PyResult<Self> {
        Python::with_gil(|py| {
            // Create model args for prefill and decode
            let prefill_args = crate::config::PyModelArgs::new(
                2560, num_layers, 20, Some(5), 128256, 6912, 1e-5, 500000.0, false
            );
            let decode_args = crate::config::PyModelArgs::new(
                2560, num_layers, 20, Some(5), 128256, 6912, 1e-5, 500000.0, true
            );
            
            // Create models
            let prefill_model = Py::new(py, PyBitNetModel::from_pretrained(
                PyBitNetModel::type_object(py),
                &ckpt_dir,
                Some(prefill_args),
                device.clone(),
                "bfloat16".to_string(),
            )?)?;
            
            let decode_model = Py::new(py, PyBitNetModel::from_pretrained(
                PyBitNetModel::type_object(py),
                &ckpt_dir,
                Some(decode_args),
                device.clone(),
                "bfloat16".to_string(),
            )?)?;
            
            // Create tokenizer
            let tokenizer_path = tokenizer_path.unwrap_or_else(|| "./tokenizer.model".to_string());
            let tokenizer = Py::new(py, PyTokenizer::new(tokenizer_path)?)?;
            
            Ok(Self::new(prefill_model, decode_model, tokenizer, gen_args, Some(device)))
        })
    }
    
    /// Generate text from prompts (matching existing API)
    #[pyo3(signature = (prompts, use_cuda_graphs = true, use_sampling = None))]
    fn generate_all(
        &mut self,
        prompts: Vec<Vec<u32>>,
        use_cuda_graphs: bool,
        use_sampling: Option<bool>,
    ) -> PyResult<(PyStats, Vec<Vec<u32>>)> {
        let use_sampling = use_sampling.unwrap_or(self.gen_args.use_sampling);
        let batch_size = prompts.len();
        
        // Initialize stats
        let mut stats = PyStats::new();
        let start_time = std::time::Instant::now();
        
        // TODO: Implement actual generation when inference engine is ready
        // For now, generate dummy responses
        let mut results = Vec::new();
        
        for prompt in prompts {
            let mut generated = prompt.clone();
            
            // Generate dummy tokens
            for i in 0..self.gen_args.gen_length {
                let next_token = if use_sampling {
                    // Simple random sampling
                    1000 + (i % 1000) as u32
                } else {
                    // Greedy decoding
                    1000 + (prompt.len() + i) as u32 % 1000
                };
                
                generated.push(next_token);
                
                // Check for EOS
                Python::with_gil(|py| {
                    let tokenizer = self.tokenizer.borrow(py);
                    if next_token == tokenizer.eos_id {
                        return;
                    }
                });
            }
            
            // Return only the generated part
            results.push(generated[prompt.len()..].to_vec());
        }
        
        // Update stats
        let elapsed = start_time.elapsed().as_secs_f64();
        stats.total_time = elapsed;
        stats.total_tokens = results.iter().map(|r| r.len()).sum();
        stats.tokens_per_second = stats.total_tokens as f64 / elapsed;
        stats.prefill_time = elapsed * 0.1; // Dummy values
        stats.decode_time = elapsed * 0.9;
        stats.memory_used = 2.5; // Dummy memory usage in GB
        
        Ok((stats, results))
    }
    
    /// Generate text from a single prompt
    fn generate(&mut self, prompt: &str) -> PyResult<String> {
        Python::with_gil(|py| {
            let tokenizer = self.tokenizer.borrow(py);
            let tokens = tokenizer.encode(prompt, true, false, None, None)?;
            
            let (_, results) = self.generate_all(vec![tokens], true, None)?;
            let generated_tokens = &results[0];
            
            tokenizer.decode(generated_tokens.clone())
        })
    }
    
    /// Generate text with streaming (async)
    fn generate_stream<'py>(&mut self, py: Python<'py>, prompt: &str) -> PyResult<&'py PyAny> {
        let prompt = prompt.to_string();
        let tokenizer = self.tokenizer.clone();
        let gen_args = self.gen_args.clone();
        
        future_into_py(py, async move {
            // TODO: Implement actual streaming when inference engine is ready
            Python::with_gil(|py| {
                let tokenizer = tokenizer.borrow(py);
                let tokens = tokenizer.encode(&prompt, true, false, None, None)?;
                
                // For now, just return the full generation as a single chunk
                let mut generated = tokens.clone();
                for i in 0..gen_args.gen_length {
                    generated.push(1000 + i as u32);
                }
                
                let result = tokenizer.decode(generated[tokens.len()..].to_vec())?;
                Ok(result)
            })
        })
    }
    
    /// Compile prefill model (matching existing API)
    fn compile_prefill(&mut self) -> PyResult<()> {
        // TODO: Implement model compilation when inference engine is ready
        Ok(())
    }
    
    /// Compile generation model (matching existing API)
    fn compile_generate(&mut self) -> PyResult<()> {
        // TODO: Implement model compilation when inference engine is ready
        Ok(())
    }
    
    /// Get generation arguments
    #[getter]
    fn gen_args(&self) -> PyGenArgs {
        self.gen_args.clone()
    }
    
    /// Set generation arguments
    #[setter]
    fn set_gen_args(&mut self, gen_args: PyGenArgs) {
        self.gen_args = gen_args;
    }
    
    /// Get device
    #[getter]
    fn device(&self) -> String {
        self.device.clone()
    }
    
    /// Get tokenizer
    #[getter]
    fn tokenizer(&self) -> Py<PyTokenizer> {
        self.tokenizer.clone()
    }
    
    fn __repr__(&self) -> String {
        format!(
            "InferenceEngine(device='{}', gen_length={}, batch_size={})",
            self.device, self.gen_args.gen_length, self.gen_args.gen_bsz
        )
    }
}

/// Simple inference engine for basic use cases
#[pyclass]
pub struct PySimpleInference {
    model: Py<PyBitNetModel>,
    tokenizer: Py<PyTokenizer>,
    config: PyInferenceConfig,
}

#[pymethods]
impl PySimpleInference {
    #[new]
    fn new(
        model: Py<PyBitNetModel>,
        tokenizer: Py<PyTokenizer>,
        config: Option<PyInferenceConfig>,
    ) -> Self {
        let config = config.unwrap_or_else(|| PyInferenceConfig::new(
            2048, 128, 0.8, 0.9, None, 1.0, true, None, None, None
        ));
        
        Self {
            model,
            tokenizer,
            config,
        }
    }
    
    /// Generate text from prompt
    fn generate(&self, prompt: &str) -> PyResult<String> {
        Python::with_gil(|py| {
            let tokenizer = self.tokenizer.borrow(py);
            let tokens = tokenizer.encode(prompt, true, false, None, None)?;
            
            // TODO: Implement actual generation when inference engine is ready
            // For now, generate dummy response
            let mut generated = tokens.clone();
            for i in 0..self.config.max_new_tokens {
                generated.push(1000 + i as u32);
            }
            
            tokenizer.decode(generated[tokens.len()..].to_vec())
        })
    }
    
    /// Generate text with streaming
    fn generate_stream<'py>(&self, py: Python<'py>, prompt: &str) -> PyResult<&'py PyAny> {
        let prompt = prompt.to_string();
        let tokenizer = self.tokenizer.clone();
        let config = self.config.clone();
        
        future_into_py(py, async move {
            // TODO: Implement actual streaming generation
            Python::with_gil(|py| {
                let tokenizer = tokenizer.borrow(py);
                let tokens = tokenizer.encode(&prompt, true, false, None, None)?;
                
                let mut generated = tokens.clone();
                for i in 0..config.max_new_tokens {
                    generated.push(1000 + i as u32);
                }
                
                tokenizer.decode(generated[tokens.len()..].to_vec())
            })
        })
    }
    
    /// Get inference configuration
    #[getter]
    fn config(&self) -> PyInferenceConfig {
        self.config.clone()
    }
    
    /// Set inference configuration
    #[setter]
    fn set_config(&mut self, config: PyInferenceConfig) {
        self.config = config;
    }
}