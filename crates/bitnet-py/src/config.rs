use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Model configuration matching the existing Python API
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyModelArgs {
    #[pyo3(get, set)]
    pub dim: usize,
    #[pyo3(get, set)]
    pub n_layers: usize,
    #[pyo3(get, set)]
    pub n_heads: usize,
    #[pyo3(get, set)]
    pub n_kv_heads: Option<usize>,
    #[pyo3(get, set)]
    pub vocab_size: usize,
    #[pyo3(get, set)]
    pub ffn_dim: usize,
    #[pyo3(get, set)]
    pub norm_eps: f64,
    #[pyo3(get, set)]
    pub rope_theta: f64,
    #[pyo3(get, set)]
    pub use_kernel: bool,
}

#[pymethods]
impl PyModelArgs {
    #[new]
    #[pyo3(signature = (
        dim = 2560,
        n_layers = 30,
        n_heads = 20,
        n_kv_heads = None,
        vocab_size = 128256,
        ffn_dim = 6912,
        norm_eps = 1e-5,
        rope_theta = 500000.0,
        use_kernel = false
    ))]
    fn new(
        dim: usize,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: Option<usize>,
        vocab_size: usize,
        ffn_dim: usize,
        norm_eps: f64,
        rope_theta: f64,
        use_kernel: bool,
    ) -> Self {
        Self {
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            ffn_dim,
            norm_eps,
            rope_theta,
            use_kernel,
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "ModelArgs(dim={}, n_layers={}, n_heads={}, n_kv_heads={:?}, vocab_size={}, ffn_dim={}, norm_eps={}, rope_theta={}, use_kernel={})",
            self.dim, self.n_layers, self.n_heads, self.n_kv_heads, self.vocab_size, 
            self.ffn_dim, self.norm_eps, self.rope_theta, self.use_kernel
        )
    }
    
    /// Create ModelArgs from a dictionary
    #[classmethod]
    fn from_dict(_cls: &PyType, dict: &PyDict) -> PyResult<Self> {
        let json_str = serde_json::to_string(&dict.extract::<serde_json::Value>()?)?;
        let args: PyModelArgs = serde_json::from_str(&json_str)?;
        Ok(args)
    }
    
    /// Convert ModelArgs to a dictionary
    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("dim", self.dim)?;
            dict.set_item("n_layers", self.n_layers)?;
            dict.set_item("n_heads", self.n_heads)?;
            dict.set_item("n_kv_heads", self.n_kv_heads)?;
            dict.set_item("vocab_size", self.vocab_size)?;
            dict.set_item("ffn_dim", self.ffn_dim)?;
            dict.set_item("norm_eps", self.norm_eps)?;
            dict.set_item("rope_theta", self.rope_theta)?;
            dict.set_item("use_kernel", self.use_kernel)?;
            Ok(dict.into())
        })
    }
}

/// Generation configuration matching the existing Python API
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyGenArgs {
    #[pyo3(get, set)]
    pub gen_length: usize,
    #[pyo3(get, set)]
    pub gen_bsz: usize,
    #[pyo3(get, set)]
    pub prompt_length: usize,
    #[pyo3(get, set)]
    pub use_sampling: bool,
    #[pyo3(get, set)]
    pub temperature: f64,
    #[pyo3(get, set)]
    pub top_p: f64,
    #[pyo3(get, set)]
    pub top_k: Option<usize>,
    #[pyo3(get, set)]
    pub repetition_penalty: f64,
}

#[pymethods]
impl PyGenArgs {
    #[new]
    #[pyo3(signature = (
        gen_length = 32,
        gen_bsz = 1,
        prompt_length = 64,
        use_sampling = false,
        temperature = 0.8,
        top_p = 0.9,
        top_k = None,
        repetition_penalty = 1.0
    ))]
    fn new(
        gen_length: usize,
        gen_bsz: usize,
        prompt_length: usize,
        use_sampling: bool,
        temperature: f64,
        top_p: f64,
        top_k: Option<usize>,
        repetition_penalty: f64,
    ) -> Self {
        Self {
            gen_length,
            gen_bsz,
            prompt_length,
            use_sampling,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "GenArgs(gen_length={}, gen_bsz={}, prompt_length={}, use_sampling={}, temperature={}, top_p={}, top_k={:?}, repetition_penalty={})",
            self.gen_length, self.gen_bsz, self.prompt_length, self.use_sampling, 
            self.temperature, self.top_p, self.top_k, self.repetition_penalty
        )
    }
}

/// Inference configuration for the Rust engine
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyInferenceConfig {
    #[pyo3(get, set)]
    pub max_length: usize,
    #[pyo3(get, set)]
    pub max_new_tokens: usize,
    #[pyo3(get, set)]
    pub temperature: f64,
    #[pyo3(get, set)]
    pub top_p: f64,
    #[pyo3(get, set)]
    pub top_k: Option<usize>,
    #[pyo3(get, set)]
    pub repetition_penalty: f64,
    #[pyo3(get, set)]
    pub do_sample: bool,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    #[pyo3(get, set)]
    pub pad_token_id: Option<u32>,
    #[pyo3(get, set)]
    pub eos_token_id: Option<u32>,
}

#[pymethods]
impl PyInferenceConfig {
    #[new]
    #[pyo3(signature = (
        max_length = 2048,
        max_new_tokens = 128,
        temperature = 0.8,
        top_p = 0.9,
        top_k = None,
        repetition_penalty = 1.0,
        do_sample = true,
        seed = None,
        pad_token_id = None,
        eos_token_id = None
    ))]
    fn new(
        max_length: usize,
        max_new_tokens: usize,
        temperature: f64,
        top_p: f64,
        top_k: Option<usize>,
        repetition_penalty: f64,
        do_sample: bool,
        seed: Option<u64>,
        pad_token_id: Option<u32>,
        eos_token_id: Option<u32>,
    ) -> Self {
        Self {
            max_length,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            do_sample,
            seed,
            pad_token_id,
            eos_token_id,
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "InferenceConfig(max_length={}, max_new_tokens={}, temperature={}, top_p={}, top_k={:?}, repetition_penalty={}, do_sample={}, seed={:?})",
            self.max_length, self.max_new_tokens, self.temperature, self.top_p, 
            self.top_k, self.repetition_penalty, self.do_sample, self.seed
        )
    }
}