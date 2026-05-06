//! Reusable request routing helpers.
//!
//! Provides lightweight route definitions and pattern matching utilities that
//! can be shared across server crates.

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
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
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
    #[must_use]
    pub fn new(prefix: &str) -> Self {
        Self { routes: Vec::new(), prefix: prefix.to_string() }
    }

    /// Build with standard OpenAI-compatible routes.
    #[must_use]
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

    #[must_use]
    pub const fn route_count(&self) -> usize {
        self.routes.len()
    }

    /// Find matching route.
    #[must_use]
    pub fn resolve(&self, method: Method, path: &str) -> Option<&Route> {
        self.routes.iter().find(|r| r.method == method && path_matches(&r.path, path))
    }

    /// All routes for a method.
    #[must_use]
    pub fn by_method(&self, method: Method) -> Vec<&Route> {
        self.routes.iter().filter(|r| r.method == method).collect()
    }

    /// Routes that require a loaded model.
    #[must_use]
    pub fn model_routes(&self) -> Vec<&Route> {
        self.routes.iter().filter(|r| r.requires_model).collect()
    }

    /// Generate a route table for display.
    #[must_use]
    pub fn table(&self) -> Vec<(String, String, String)> {
        self.routes
            .iter()
            .map(|r| (r.method.as_str().to_string(), r.path.clone(), r.handler.clone()))
            .collect()
    }

    /// Group routes by a shallow path prefix.
    #[must_use]
    pub fn by_prefix(&self) -> HashMap<String, Vec<&Route>> {
        let mut map: HashMap<String, Vec<&Route>> = HashMap::new();
        for route in &self.routes {
            let prefix = route.path.split('/').take(3).collect::<Vec<_>>().join("/");
            map.entry(prefix).or_default().push(route);
        }
        map
    }
}

/// Simple path matching (supports `:param` wildcards).
#[must_use]
pub fn path_matches(pattern: &str, path: &str) -> bool {
    let pattern_parts: Vec<&str> = pattern.split('/').collect();
    let route_parts: Vec<&str> = path.split('/').collect();
    if pattern_parts.len() != route_parts.len() {
        return false;
    }
    pattern_parts.iter().zip(route_parts.iter()).all(|(pattern_part, route_part)| {
        pattern_part.starts_with(':') || *pattern_part == *route_part
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_routes_include_expected_entries() {
        let router = RequestRouter::openai_compatible();
        assert!(router.route_count() >= 7);
    }

    #[test]
    fn resolve_known_routes() {
        let router = RequestRouter::openai_compatible();
        assert_eq!(
            router.resolve(Method::Post, "/v1/completions").map(|r| r.handler.as_str()),
            Some("completions")
        );
        assert_eq!(
            router.resolve(Method::Post, "/v1/chat/completions").map(|r| r.handler.as_str()),
            Some("chat_completions")
        );
        assert_eq!(
            router.resolve(Method::Get, "/health").map(|r| r.handler.as_str()),
            Some("health")
        );
    }

    #[test]
    fn wildcard_path_resolution_works() {
        let router = RequestRouter::openai_compatible();
        assert_eq!(
            router.resolve(Method::Get, "/v1/models/phi-4").map(|r| r.handler.as_str()),
            Some("get_model")
        );
    }

    #[test]
    fn route_filters_and_table_work() {
        let router = RequestRouter::openai_compatible();
        assert!(router.by_method(Method::Get).len() >= 3);
        assert!(router.model_routes().len() >= 3);
        assert!(router.table().iter().any(|(m, _, _)| m == "POST"));
    }

    #[test]
    fn add_custom_route_and_method_display() {
        let mut router = RequestRouter::new("/api");
        router.add_route(Method::Get, "/status", "status", false);
        assert_eq!(router.route_count(), 1);
        assert!(router.resolve(Method::Get, "/api/status").is_some());
        assert_eq!(Method::Get.as_str(), "GET");
        assert_eq!(Method::Post.as_str(), "POST");
    }

    #[test]
    fn path_matching_validates_shape_and_wildcards() {
        assert!(path_matches("/v1/models/:id", "/v1/models/phi-4"));
        assert!(!path_matches("/v1/models/:id", "/v1/models"));
    }
}
