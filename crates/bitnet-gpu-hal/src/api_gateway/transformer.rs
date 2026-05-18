use bitnet_minimal_json_core::MinimalJson;

// ── Request transformer ─────────────────────────────────────────────────────

/// Transforms between OpenAI-compatible request/response format and the
/// internal `BitNet` inference representation.
#[derive(Debug, Clone)]
pub struct RequestTransformer {
    /// Default model name injected when the request omits one.
    pub default_model: String,
    /// Maximum tokens cap enforced on transformed requests.
    pub max_tokens_limit: u32,
}

impl Default for RequestTransformer {
    fn default() -> Self {
        Self { default_model: "bitnet-b1.58-2B-4T".to_string(), max_tokens_limit: 4096 }
    }
}

/// Simplified internal inference request produced by transformation.
#[derive(Debug, Clone, PartialEq)]
pub struct InternalInferenceRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub stream: bool,
}

/// Simplified internal inference response before transformation back to `OpenAI`
/// format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternalInferenceResponse {
    pub text: String,
    pub tokens_used: u32,
    pub finish_reason: String,
    pub model: String,
}

impl RequestTransformer {
    /// Transform an OpenAI-style JSON body into an internal request.
    ///
    /// Expects a JSON object with optional fields: `model`, `prompt` or
    /// `messages`, `max_tokens`, `temperature`, `top_p`, `stream`.
    pub fn to_internal(&self, body: &[u8]) -> Result<InternalInferenceRequest, String> {
        let text = std::str::from_utf8(body).map_err(|e| format!("invalid UTF-8: {e}"))?;
        let parsed = MinimalJson::parse(text)?;

        let model = parsed.get_str("model").unwrap_or_else(|| self.default_model.clone());

        // Accept either `prompt` (completions) or first message content (chat).
        let prompt = if let Some(p) = parsed.get_str("prompt") {
            p
        } else if let Some(m) = parsed.get_str("messages") {
            m
        } else {
            return Err("missing 'prompt' or 'messages'".to_string());
        };

        let max_tokens = parsed.get_u32("max_tokens").unwrap_or(256).min(self.max_tokens_limit);

        let temperature = parsed.get_f32("temperature").unwrap_or(1.0);
        let top_p = parsed.get_f32("top_p").unwrap_or(1.0);
        let stream = parsed.get_bool("stream").unwrap_or(false);

        Ok(InternalInferenceRequest { model, prompt, max_tokens, temperature, top_p, stream })
    }

    /// Transform an internal response back into an OpenAI-compatible JSON body.
    pub fn to_openai_response(&self, resp: &InternalInferenceResponse) -> Vec<u8> {
        let json = format!(
            r#"{{"id":"chatcmpl-bitnet","object":"chat.completion","model":"{}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{}"}},"finish_reason":"{}"}}],"usage":{{"prompt_tokens":0,"completion_tokens":{},"total_tokens":{}}}}}"#,
            resp.model,
            resp.text.replace('\\', "\\\\").replace('"', "\\\""),
            resp.finish_reason,
            resp.tokens_used,
            resp.tokens_used,
        );
        json.into_bytes()
    }
}
