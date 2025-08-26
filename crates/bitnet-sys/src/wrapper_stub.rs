use std::ptr;

/// Error type for C++ FFI operations when backend is unavailable
#[derive(Debug, thiserror::Error)]
pub enum CppError {
    #[error("C++ backend unavailable")]
    Unavailable,
}

pub type Result<T> = std::result::Result<T, CppError>;

pub fn init_backend() {}

pub fn free_backend() {}

pub fn get_version() -> String {
    "unavailable".to_string()
}

/// Placeholder model when C++ bridge is missing
#[derive(Debug)]
pub struct Model;

impl Model {
    pub fn load(_path: &str) -> Result<Self> {
        Err(CppError::Unavailable)
    }
    pub fn n_vocab(&self) -> i32 {
        0
    }
    pub fn n_ctx_train(&self) -> i32 {
        0
    }
    pub fn n_embd(&self) -> i32 {
        0
    }
    #[allow(dead_code)]
    pub(crate) fn as_ptr(&self) -> *mut std::ffi::c_void {
        ptr::null_mut()
    }
}

/// Placeholder context when C++ bridge is missing
#[derive(Debug)]
pub struct Context;

impl Context {
    pub fn new(_model: &Model, _n_ctx: u32, _n_batch: u32, _n_threads: i32) -> Result<Self> {
        Err(CppError::Unavailable)
    }

    pub fn tokenize(&self, _text: &str, _add_special: bool) -> Result<Vec<i32>> {
        Err(CppError::Unavailable)
    }

    pub fn decode(&self, _tokens: &[i32]) -> Result<String> {
        Err(CppError::Unavailable)
    }

    pub fn eval(&mut self, _tokens: &[i32], _n_past: i32) -> Result<()> {
        Err(CppError::Unavailable)
    }

    pub fn get_logits(&self) -> Result<Vec<f32>> {
        Err(CppError::Unavailable)
    }

    pub fn get_logits_ith(&self, _i: i32) -> Result<Vec<f32>> {
        Err(CppError::Unavailable)
    }

    pub fn get_all_logits(&self, _n_tokens: usize) -> Result<Vec<Vec<f32>>> {
        Err(CppError::Unavailable)
    }

    pub fn sample_greedy(&self, _logits: &[f32]) -> i32 {
        0
    }
}

/// Placeholder session when C++ bridge is missing
#[derive(Debug)]
pub struct Session {
    pub model: Model,
    pub context: Context,
}

impl Session {
    pub fn load(_model_path: &str, _n_ctx: u32, _n_batch: u32, _n_threads: i32) -> Result<Self> {
        Err(CppError::Unavailable)
    }

    pub fn load_deterministic(_model_path: &str) -> Result<Self> {
        Err(CppError::Unavailable)
    }

    pub fn tokenize(&self, _text: &str) -> Result<Vec<i32>> {
        Err(CppError::Unavailable)
    }

    pub fn decode(&self, _tokens: &[i32]) -> Result<String> {
        Err(CppError::Unavailable)
    }

    pub fn eval_and_get_logits(&mut self, _tokens: &[i32], _n_past: i32) -> Result<Vec<f32>> {
        Err(CppError::Unavailable)
    }

    pub fn generate_greedy(&mut self, _prompt: &str, _max_tokens: usize) -> Result<Vec<i32>> {
        Err(CppError::Unavailable)
    }
}
