//! Streaming generation support for WebAssembly

use wasm_bindgen::prelude::*;
use js_sys::{Promise, AsyncIterator, Object, Reflect};
use web_sys::{ReadableStream, ReadableStreamDefaultController, console};
use wasm_bindgen_futures::spawn_local;
use std::rc::Rc;
use std::cell::RefCell;

use crate::inference::WasmGenerationConfig;
use crate::utils::{JsError, to_js_error};

/// WebAssembly-compatible streaming generation
#[wasm_bindgen]
pub struct WasmGenerationStream {
    prompt: String,
    config: WasmGenerationConfig,
    position: usize,
    finished: bool,
    tokens: Vec<String>,
}

#[wasm_bindgen]
impl WasmGenerationStream {
    /// Create a new generation stream
    pub fn new(prompt: String, config: WasmGenerationConfig) -> Result<WasmGenerationStream, JsError> {
        console::log_1(&format!("Creating generation stream for: {}", prompt).into());

        // Pre-generate tokens for demonstration
        // In a real implementation, this would be done incrementally
        let tokens = Self::simulate_token_generation(&prompt, &config);

        Ok(WasmGenerationStream {
            prompt,
            config,
            position: 0,
            finished: false,
            tokens,
        })
    }

    /// Get the next token (async)
    #[wasm_bindgen]
    pub fn next(&mut self) -> Promise {
        let position = self.position;
        let tokens = self.tokens.clone();
        let finished = self.finished;
        let max_tokens = self.config.max_new_tokens;

        wasm_bindgen_futures::future_to_promise(async move {
            if finished || position >= tokens.len() || position >= max_tokens {
                return Ok(Self::create_iterator_result(None, true));
            }

            // Simulate processing delay
            Self::sleep(50).await;

            let token = tokens.get(position).cloned().unwrap_or_default();
            Ok(Self::create_iterator_result(Some(token), false))
        })
    }

    /// Check if the stream is finished
    #[wasm_bindgen]
    pub fn is_finished(&self) -> bool {
        self.finished || self.position >= self.tokens.len() || self.position >= self.config.max_new_tokens
    }

    /// Get current position in the stream
    #[wasm_bindgen]
    pub fn get_position(&self) -> usize {
        self.position
    }

    /// Get total number of tokens that will be generated
    #[wasm_bindgen]
    pub fn get_total_tokens(&self) -> usize {
        self.tokens.len().min(self.config.max_new_tokens)
    }

    /// Cancel the stream
    #[wasm_bindgen]
    pub fn cancel(&mut self) {
        self.finished = true;
        console::log_1(&"Generation stream cancelled".into());
    }

    /// Convert to JavaScript async iterator
    #[wasm_bindgen]
    pub fn to_async_iterator(&self) -> Result<JsValue, JsError> {
        let stream_data = Rc::new(RefCell::new(StreamData {
            tokens: self.tokens.clone(),
            position: 0,
            max_tokens: self.config.max_new_tokens,
            finished: false,
        }));

        let iterator = Object::new();
        
        // Create the next() method
        let stream_data_clone = stream_data.clone();
        let next_fn = Closure::wrap(Box::new(move || -> Promise {
            let stream_data = stream_data_clone.clone();
            wasm_bindgen_futures::future_to_promise(async move {
                let mut data = stream_data.borrow_mut();
                
                if data.finished || data.position >= data.tokens.len() || data.position >= data.max_tokens {
                    return Ok(Self::create_iterator_result(None, true));
                }

                // Simulate processing delay
                Self::sleep(50).await;

                let token = data.tokens.get(data.position).cloned().unwrap_or_default();
                data.position += 1;

                if data.position >= data.tokens.len() || data.position >= data.max_tokens {
                    data.finished = true;
                }

                Ok(Self::create_iterator_result(Some(token), data.finished))
            })
        }) as Box<dyn Fn() -> Promise>);

        Reflect::set(&iterator, &"next".into(), next_fn.as_ref().unchecked_ref())?;
        next_fn.forget();

        // Make it an async iterator
        let symbol_async_iterator = js_sys::Symbol::async_iterator();
        let self_fn = Closure::wrap(Box::new(move || iterator.clone()) as Box<dyn Fn() -> Object>);
        Reflect::set(&iterator, &symbol_async_iterator, self_fn.as_ref().unchecked_ref())?;
        self_fn.forget();

        Ok(iterator.into())
    }

    /// Convert to ReadableStream for use with Streams API
    #[wasm_bindgen]
    pub fn to_readable_stream(&self) -> Result<ReadableStream, JsError> {
        let tokens = self.tokens.clone();
        let max_tokens = self.config.max_new_tokens;
        
        let start_fn = Closure::wrap(Box::new(move |controller: ReadableStreamDefaultController| {
            let tokens = tokens.clone();
            spawn_local(async move {
                for (i, token) in tokens.iter().enumerate() {
                    if i >= max_tokens {
                        break;
                    }

                    // Simulate processing delay
                    Self::sleep(50).await;

                    let chunk = js_sys::Uint8Array::from(token.as_bytes());
                    if controller.enqueue_with_chunk(&chunk).is_err() {
                        break;
                    }
                }
                let _ = controller.close();
            });
        }) as Box<dyn FnMut(ReadableStreamDefaultController)>);

        let underlying_source = Object::new();
        Reflect::set(&underlying_source, &"start".into(), start_fn.as_ref().unchecked_ref())?;
        start_fn.forget();

        Ok(ReadableStream::new_with_underlying_source(&underlying_source)?)
    }
}

impl WasmGenerationStream {
    /// Simulate token generation for demonstration
    fn simulate_token_generation(prompt: &str, config: &WasmGenerationConfig) -> Vec<String> {
        let base_tokens = vec![
            " Hello", " there", "!", " This", " is", " a", " generated", " response", " for", " your", " prompt", ":",
            " \"", prompt, "\".", " The", " temperature", " is", " set", " to", &format!(" {:.1}", config.temperature),
            " and", " we", " will", " generate", " up", " to", &format!(" {}", config.max_new_tokens), " tokens", "."
        ];

        let mut tokens = Vec::new();
        let mut count = 0;

        for token in base_tokens.iter().cycle() {
            if count >= config.max_new_tokens {
                break;
            }
            tokens.push(token.to_string());
            count += 1;

            // Check for stop tokens
            if config.stop_tokens.iter().any(|stop| token.contains(stop)) {
                break;
            }
        }

        tokens
    }

    /// Create an iterator result object
    fn create_iterator_result(value: Option<String>, done: bool) -> JsValue {
        let result = Object::new();
        
        if let Some(val) = value {
            let _ = Reflect::set(&result, &"value".into(), &JsValue::from_str(&val));
        } else {
            let _ = Reflect::set(&result, &"value".into(), &JsValue::UNDEFINED);
        }
        
        let _ = Reflect::set(&result, &"done".into(), &JsValue::from_bool(done));
        result.into()
    }

    /// Async sleep utility
    async fn sleep(ms: i32) {
        let promise = Promise::new(&mut |resolve, _reject| {
            web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(
                    &resolve,
                    ms,
                )
                .unwrap();
        });
        let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    }
}

/// Internal stream data structure
struct StreamData {
    tokens: Vec<String>,
    position: usize,
    max_tokens: usize,
    finished: bool,
}

/// JavaScript-compatible stream token
#[wasm_bindgen]
pub struct StreamToken {
    text: String,
    position: usize,
    is_final: bool,
    timestamp: f64,
}

#[wasm_bindgen]
impl StreamToken {
    #[wasm_bindgen(constructor)]
    pub fn new(text: String, position: usize, is_final: bool) -> StreamToken {
        let timestamp = js_sys::Date::now();
        StreamToken {
            text,
            position,
            is_final,
            timestamp,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn position(&self) -> usize {
        self.position
    }

    #[wasm_bindgen(getter)]
    pub fn is_final(&self) -> bool {
        self.is_final
    }

    #[wasm_bindgen(getter)]
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }
}

/// Stream statistics
#[wasm_bindgen]
pub struct StreamStats {
    tokens_generated: usize,
    time_elapsed_ms: f64,
    tokens_per_second: f64,
    bytes_processed: usize,
}

#[wasm_bindgen]
impl StreamStats {
    #[wasm_bindgen(getter)]
    pub fn tokens_generated(&self) -> usize {
        self.tokens_generated
    }

    #[wasm_bindgen(getter)]
    pub fn time_elapsed_ms(&self) -> f64 {
        self.time_elapsed_ms
    }

    #[wasm_bindgen(getter)]
    pub fn tokens_per_second(&self) -> f64 {
        self.tokens_per_second
    }

    #[wasm_bindgen(getter)]
    pub fn bytes_processed(&self) -> usize {
        self.bytes_processed
    }
}