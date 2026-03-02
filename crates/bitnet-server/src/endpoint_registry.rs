//! Server endpoint registry.
//!
//! Tracks available API endpoints and their configuration.

use std::collections::HashMap;

/// HTTP method type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
}

impl HttpMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
        }
    }
}

/// API endpoint definition.
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub method: HttpMethod,
    pub path: String,
    pub description: String,
    pub auth_required: bool,
    pub streaming: bool,
    pub deprecated: bool,
}

/// Endpoint registry.
#[derive(Debug, Clone)]
pub struct EndpointRegistry {
    endpoints: Vec<Endpoint>,
    tags: HashMap<String, Vec<usize>>, // tag → endpoint indices
}

impl Default for EndpointRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl EndpointRegistry {
    pub fn new() -> Self {
        Self { endpoints: Vec::new(), tags: HashMap::new() }
    }

    pub fn register(&mut self, endpoint: Endpoint) -> usize {
        let idx = self.endpoints.len();
        self.endpoints.push(endpoint);
        idx
    }

    pub fn tag(&mut self, idx: usize, tag: &str) {
        self.tags.entry(tag.to_string()).or_default().push(idx);
    }

    pub fn by_tag(&self, tag: &str) -> Vec<&Endpoint> {
        self.tags
            .get(tag)
            .map(|indices| indices.iter().filter_map(|&i| self.endpoints.get(i)).collect())
            .unwrap_or_default()
    }

    pub fn by_method(&self, method: HttpMethod) -> Vec<&Endpoint> {
        self.endpoints.iter().filter(|e| e.method == method).collect()
    }

    pub fn by_path(&self, path: &str) -> Option<&Endpoint> {
        self.endpoints.iter().find(|e| e.path == path)
    }

    pub fn all(&self) -> &[Endpoint] {
        &self.endpoints
    }

    pub fn count(&self) -> usize {
        self.endpoints.len()
    }

    pub fn streaming_endpoints(&self) -> Vec<&Endpoint> {
        self.endpoints.iter().filter(|e| e.streaming).collect()
    }

    pub fn deprecated_endpoints(&self) -> Vec<&Endpoint> {
        self.endpoints.iter().filter(|e| e.deprecated).collect()
    }

    /// Build the standard BitNet inference endpoints.
    pub fn standard() -> Self {
        let mut reg = Self::new();

        let completions = reg.register(Endpoint {
            method: HttpMethod::Post,
            path: "/v1/completions".into(),
            description: "Generate text completions".into(),
            auth_required: false,
            streaming: true,
            deprecated: false,
        });
        reg.tag(completions, "inference");

        let chat = reg.register(Endpoint {
            method: HttpMethod::Post,
            path: "/v1/chat/completions".into(),
            description: "Chat-style completions".into(),
            auth_required: false,
            streaming: true,
            deprecated: false,
        });
        reg.tag(chat, "inference");
        reg.tag(chat, "chat");

        let models = reg.register(Endpoint {
            method: HttpMethod::Get,
            path: "/v1/models".into(),
            description: "List loaded models".into(),
            auth_required: false,
            streaming: false,
            deprecated: false,
        });
        reg.tag(models, "info");

        let health = reg.register(Endpoint {
            method: HttpMethod::Get,
            path: "/health".into(),
            description: "Health check".into(),
            auth_required: false,
            streaming: false,
            deprecated: false,
        });
        reg.tag(health, "info");

        let embed = reg.register(Endpoint {
            method: HttpMethod::Post,
            path: "/v1/embeddings".into(),
            description: "Generate embeddings".into(),
            auth_required: false,
            streaming: false,
            deprecated: false,
        });
        reg.tag(embed, "inference");

        reg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_registry() {
        let r = EndpointRegistry::new();
        assert_eq!(r.count(), 0);
    }

    #[test]
    fn test_register() {
        let mut r = EndpointRegistry::new();
        r.register(Endpoint {
            method: HttpMethod::Get,
            path: "/test".into(),
            description: "test".into(),
            auth_required: false,
            streaming: false,
            deprecated: false,
        });
        assert_eq!(r.count(), 1);
    }

    #[test]
    fn test_by_path() {
        let mut r = EndpointRegistry::new();
        r.register(Endpoint {
            method: HttpMethod::Post,
            path: "/api/gen".into(),
            description: "gen".into(),
            auth_required: false,
            streaming: false,
            deprecated: false,
        });
        assert!(r.by_path("/api/gen").is_some());
        assert!(r.by_path("/nonexistent").is_none());
    }

    #[test]
    fn test_by_method() {
        let r = EndpointRegistry::standard();
        let gets = r.by_method(HttpMethod::Get);
        assert!(gets.len() >= 2); // /v1/models and /health
    }

    #[test]
    fn test_by_tag() {
        let r = EndpointRegistry::standard();
        let inference = r.by_tag("inference");
        assert!(inference.len() >= 2);
    }

    #[test]
    fn test_streaming() {
        let r = EndpointRegistry::standard();
        let streaming = r.streaming_endpoints();
        assert!(streaming.len() >= 2); // completions and chat
    }

    #[test]
    fn test_standard_count() {
        let r = EndpointRegistry::standard();
        assert_eq!(r.count(), 5);
    }

    #[test]
    fn test_http_method_str() {
        assert_eq!(HttpMethod::Get.as_str(), "GET");
        assert_eq!(HttpMethod::Post.as_str(), "POST");
    }

    #[test]
    fn test_deprecated() {
        let r = EndpointRegistry::standard();
        assert!(r.deprecated_endpoints().is_empty());
    }

    #[test]
    fn test_default() {
        let r = EndpointRegistry::default();
        assert_eq!(r.count(), 0);
    }

    #[test]
    fn test_chat_tag() {
        let r = EndpointRegistry::standard();
        let chat = r.by_tag("chat");
        assert_eq!(chat.len(), 1);
        assert_eq!(chat[0].path, "/v1/chat/completions");
    }

    #[test]
    fn test_info_tag() {
        let r = EndpointRegistry::standard();
        let info = r.by_tag("info");
        assert_eq!(info.len(), 2);
    }
}
