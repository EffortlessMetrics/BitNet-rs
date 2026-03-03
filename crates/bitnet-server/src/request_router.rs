//! Request routing for the inference server.
//!
//! Route incoming requests to appropriate model backends.

use std::collections::HashMap;

/// HTTP method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Method {
    Get,
    Post,
    Put,
    Delete,
}

impl Method {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
        }
    }
}

/// Route definition.
#[derive(Debug, Clone)]
pub struct Route {
    pub method: Method,
    pub path: String,
    pub handler: String,
    pub requires_model: bool,
}

/// Router that maps paths to handlers.
#[derive(Debug, Clone)]
pub struct RequestRouter {
    routes: Vec<Route>,
    prefix: String,
}

impl Default for RequestRouter {
    fn default() -> Self {
        Self::new("")
    }
}

impl RequestRouter {
    pub fn new(prefix: &str) -> Self {
        Self { routes: Vec::new(), prefix: prefix.to_string() }
    }

    /// Build with standard OpenAI-compatible routes.
    pub fn openai_compatible() -> Self {
        let mut router = Self::new("/v1");
        router.add_route(Method::Post, "/completions", "completions", true);
        router.add_route(Method::Post, "/chat/completions", "chat_completions", true);
        router.add_route(Method::Get, "/models", "list_models", false);
        router.add_route(Method::Get, "/models/:id", "get_model", false);
        router.add_route(Method::Post, "/embeddings", "embeddings", true);
        // Health endpoints (no prefix)
        router.routes.push(Route {
            method: Method::Get,
            path: "/health".into(),
            handler: "health".into(),
            requires_model: false,
        });
        router.routes.push(Route {
            method: Method::Get,
            path: "/ready".into(),
            handler: "readiness".into(),
            requires_model: false,
        });
        router
    }

    pub fn add_route(&mut self, method: Method, path: &str, handler: &str, requires_model: bool) {
        let full_path = format!("{}{}", self.prefix, path);
        self.routes.push(Route {
            method,
            path: full_path,
            handler: handler.to_string(),
            requires_model,
        });
    }

    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    /// Find matching route.
    pub fn resolve(&self, method: Method, path: &str) -> Option<&Route> {
        self.routes.iter().find(|r| r.method == method && path_matches(&r.path, path))
    }

    /// All routes for a method.
    pub fn by_method(&self, method: Method) -> Vec<&Route> {
        self.routes.iter().filter(|r| r.method == method).collect()
    }

    /// Routes that require a loaded model.
    pub fn model_routes(&self) -> Vec<&Route> {
        self.routes.iter().filter(|r| r.requires_model).collect()
    }

    /// Generate a route table for display.
    pub fn table(&self) -> Vec<(String, String, String)> {
        self.routes
            .iter()
            .map(|r| (r.method.as_str().to_string(), r.path.clone(), r.handler.clone()))
            .collect()
    }

    /// Grouped by path prefix.
    pub fn by_prefix(&self) -> HashMap<String, Vec<&Route>> {
        let mut map: HashMap<String, Vec<&Route>> = HashMap::new();
        for r in &self.routes {
            let prefix = r.path.split('/').take(3).collect::<Vec<_>>().join("/");
            map.entry(prefix).or_default().push(r);
        }
        map
    }
}

/// Simple path matching (supports :param wildcards).
fn path_matches(pattern: &str, path: &str) -> bool {
    let pat_parts: Vec<&str> = pattern.split('/').collect();
    let path_parts: Vec<&str> = path.split('/').collect();
    if pat_parts.len() != path_parts.len() {
        return false;
    }
    pat_parts.iter().zip(path_parts.iter()).all(|(p, a)| p.starts_with(':') || *p == *a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_routes() {
        let router = RequestRouter::openai_compatible();
        assert!(router.route_count() >= 7);
    }

    #[test]
    fn test_resolve_completions() {
        let router = RequestRouter::openai_compatible();
        let route = router.resolve(Method::Post, "/v1/completions").unwrap();
        assert_eq!(route.handler, "completions");
    }

    #[test]
    fn test_resolve_chat() {
        let router = RequestRouter::openai_compatible();
        let route = router.resolve(Method::Post, "/v1/chat/completions").unwrap();
        assert_eq!(route.handler, "chat_completions");
    }

    #[test]
    fn test_resolve_health() {
        let router = RequestRouter::openai_compatible();
        let route = router.resolve(Method::Get, "/health").unwrap();
        assert_eq!(route.handler, "health");
    }

    #[test]
    fn test_resolve_missing() {
        let router = RequestRouter::openai_compatible();
        assert!(router.resolve(Method::Delete, "/v1/completions").is_none());
    }

    #[test]
    fn test_wildcard() {
        let router = RequestRouter::openai_compatible();
        let route = router.resolve(Method::Get, "/v1/models/phi-4").unwrap();
        assert_eq!(route.handler, "get_model");
    }

    #[test]
    fn test_by_method() {
        let router = RequestRouter::openai_compatible();
        let gets = router.by_method(Method::Get);
        assert!(gets.len() >= 3);
    }

    #[test]
    fn test_model_routes() {
        let router = RequestRouter::openai_compatible();
        let model = router.model_routes();
        assert!(model.len() >= 3);
    }

    #[test]
    fn test_table() {
        let router = RequestRouter::openai_compatible();
        let table = router.table();
        assert!(!table.is_empty());
        assert!(table.iter().any(|(m, _, _)| m == "POST"));
    }

    #[test]
    fn test_add_custom_route() {
        let mut router = RequestRouter::new("/api");
        router.add_route(Method::Get, "/status", "status", false);
        assert_eq!(router.route_count(), 1);
        assert!(router.resolve(Method::Get, "/api/status").is_some());
    }

    #[test]
    fn test_method_str() {
        assert_eq!(Method::Get.as_str(), "GET");
        assert_eq!(Method::Post.as_str(), "POST");
    }

    #[test]
    fn test_path_matches() {
        assert!(path_matches("/v1/models/:id", "/v1/models/phi-4"));
        assert!(!path_matches("/v1/models/:id", "/v1/models"));
    }
}
