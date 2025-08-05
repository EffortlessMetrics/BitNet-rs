use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::PyBitNetError;

/// Message type for chat formatting
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyMessage {
    #[pyo3(get, set)]
    pub role: String,
    #[pyo3(get, set)]
    pub content: String,
}

#[pymethods]
impl PyMessage {
    #[new]
    fn new(role: String, content: String) -> Self {
        Self { role, content }
    }
    
    fn __repr__(&self) -> String {
        format!("Message(role='{}', content='{}')", self.role, self.content)
    }
}

/// Tokenizer wrapper matching the existing Python API
#[pyclass]
pub struct PyTokenizer {
    model_path: PathBuf,
    vocab_size: usize,
    special_tokens: HashMap<String, u32>,
    bos_id: u32,
    eos_id: u32,
    eot_id: u32,
    pad_id: u32,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    fn new(model_path: String) -> PyResult<Self> {
        let path = PathBuf::from(&model_path);
        if !path.exists() {
            return Err(PyBitNetError::new(
                format!("Tokenizer model not found: {}", model_path),
                Some("TokenizerError".to_string())
            ).into());
        }
        
        // Initialize with default values - will be loaded from actual tokenizer
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<|begin_of_text|>".to_string(), 128000);
        special_tokens.insert("<|end_of_text|>".to_string(), 128001);
        special_tokens.insert("<|eot_id|>".to_string(), 128009);
        
        Ok(Self {
            model_path: path,
            vocab_size: 128256,
            special_tokens: special_tokens.clone(),
            bos_id: 128000,
            eos_id: 128001,
            eot_id: 128009,
            pad_id: 128255,
        })
    }
    
    /// Encode text to token IDs
    #[pyo3(signature = (text, bos = true, eos = false, allowed_special = None, disallowed_special = None))]
    fn encode(
        &self,
        text: &str,
        bos: bool,
        eos: bool,
        allowed_special: Option<&PyAny>,
        disallowed_special: Option<&PyAny>,
    ) -> PyResult<Vec<u32>> {
        // TODO: Implement actual tokenization when tokenizer is ready
        // For now, return dummy tokens based on text length
        let mut tokens = Vec::new();
        
        if bos {
            tokens.push(self.bos_id);
        }
        
        // Simple word-based tokenization for demo
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, _word) in words.iter().enumerate() {
            // Use a simple hash-based token ID
            tokens.push(1000 + (i % 1000) as u32);
        }
        
        if eos {
            tokens.push(self.eos_id);
        }
        
        Ok(tokens)
    }
    
    /// Decode token IDs to text
    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        // TODO: Implement actual detokenization when tokenizer is ready
        // For now, return dummy text based on token count
        let mut text = String::new();
        
        for (i, token) in tokens.iter().enumerate() {
            if *token == self.bos_id {
                continue;
            } else if *token == self.eos_id || *token == self.eot_id {
                break;
            } else {
                if i > 0 && tokens[i-1] != self.bos_id {
                    text.push(' ');
                }
                text.push_str(&format!("token_{}", token));
            }
        }
        
        Ok(text)
    }
    
    /// Get vocabulary size
    #[getter]
    fn n_words(&self) -> usize {
        self.vocab_size
    }
    
    /// Get BOS token ID
    #[getter]
    fn bos_id(&self) -> u32 {
        self.bos_id
    }
    
    /// Get EOS token ID
    #[getter]
    fn eos_id(&self) -> u32 {
        self.eos_id
    }
    
    /// Get EOT token ID
    #[getter]
    fn eot_id(&self) -> u32 {
        self.eot_id
    }
    
    /// Get PAD token ID
    #[getter]
    fn pad_id(&self) -> u32 {
        self.pad_id
    }
    
    /// Get special tokens dictionary
    #[getter]
    fn special_tokens(&self) -> HashMap<String, u32> {
        self.special_tokens.clone()
    }
    
    fn __repr__(&self) -> String {
        format!(
            "Tokenizer(model_path='{}', vocab_size={}, bos_id={}, eos_id={})",
            self.model_path.to_string_lossy(),
            self.vocab_size,
            self.bos_id,
            self.eos_id
        )
    }
}

/// Chat format wrapper for dialog-based interactions
#[pyclass]
pub struct PyChatFormat {
    tokenizer: Py<PyTokenizer>,
    eot_id: u32,
}

#[pymethods]
impl PyChatFormat {
    #[new]
    fn new(tokenizer: Py<PyTokenizer>) -> PyResult<Self> {
        let eot_id = Python::with_gil(|py| {
            tokenizer.borrow(py).eot_id
        });
        
        Ok(Self { tokenizer, eot_id })
    }
    
    /// Decode tokens with chat formatting
    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        Python::with_gil(|py| {
            let tokenizer = self.tokenizer.borrow(py);
            let mut decoded = tokenizer.decode(tokens)?;
            
            // Remove special tokens from decoded string
            decoded = decoded.replace("<|eot_id|>", "");
            
            Ok(decoded)
        })
    }
    
    /// Encode message header
    fn encode_header(&self, message: &PyMessage) -> PyResult<Vec<u32>> {
        Python::with_gil(|py| {
            let tokenizer = self.tokenizer.borrow(py);
            let header_text = match message.role.as_str() {
                "system" => "System: ",
                "user" => "User: ",
                "assistant" => "Assistant: ",
                _ => return Err(PyBitNetError::new(
                    format!("Unknown role: {}", message.role),
                    Some("TokenizerError".to_string())
                ).into()),
            };
            
            tokenizer.encode(header_text, false, false, None, None)
        })
    }
    
    /// Encode a single message
    #[pyo3(signature = (message, return_target = false))]
    fn encode_message(&self, message: &PyMessage, return_target: bool) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let tokenizer = self.tokenizer.borrow(py);
            
            let header_tokens = self.encode_header(message)?;
            let content_tokens = tokenizer.encode(&message.content.trim(), false, false, None, None)?;
            let mut tokens = header_tokens;
            tokens.extend(content_tokens);
            tokens.push(self.eot_id);
            
            if return_target {
                let targets = if message.role == "assistant" {
                    // For assistant messages, target is the content + eot
                    let mut targets = vec![-1i32; header_tokens.len()];
                    targets.extend(vec![1i32; tokens.len() - header_tokens.len()]);
                    targets
                } else {
                    // For other messages, no target
                    vec![-1i32; tokens.len()]
                };
                
                let result = PyList::new(py, &[tokens, targets]);
                Ok(result.into())
            } else {
                Ok(tokens.into())
            }
        })
    }
    
    /// Encode dialog prompt
    #[pyo3(signature = (dialog, completion = false, return_target = false))]
    fn encode_dialog_prompt(
        &self,
        dialog: Vec<PyMessage>,
        completion: bool,
        return_target: bool,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let tokenizer = self.tokenizer.borrow(py);
            
            let mut tokens = vec![tokenizer.bos_id];
            let mut targets = vec![-1i32];
            
            for message in dialog {
                let result = self.encode_message(&message, return_target)?;
                
                if return_target {
                    let result_list = result.downcast::<PyList>(py)?;
                    let msg_tokens: Vec<u32> = result_list.get_item(0)?.extract()?;
                    let msg_targets: Vec<i32> = result_list.get_item(1)?.extract()?;
                    
                    tokens.extend(msg_tokens);
                    targets.extend(msg_targets);
                } else {
                    let msg_tokens: Vec<u32> = result.extract(py)?;
                    tokens.extend(msg_tokens);
                }
            }
            
            // Add assistant header for completion
            if completion {
                let assistant_msg = PyMessage::new("assistant".to_string(), "".to_string());
                let header_tokens = self.encode_header(&assistant_msg)?;
                tokens.extend(header_tokens);
                
                if return_target {
                    targets.extend(vec![-1i32; header_tokens.len()]);
                }
            }
            
            if return_target {
                let result = PyList::new(py, &[tokens, targets]);
                Ok(result.into())
            } else {
                Ok(tokens.into())
            }
        })
    }
    
    #[getter]
    fn eot_id(&self) -> u32 {
        self.eot_id
    }
}