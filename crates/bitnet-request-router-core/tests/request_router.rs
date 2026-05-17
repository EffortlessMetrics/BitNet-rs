//! Integration tests for `bitnet-request-router-core`.
//!
//! These tests pin the documented behaviour of `Method`, `path_matches`,
//! and `RequestRouter` so that future refactors cannot silently change
//! routing semantics.

use bitnet_request_router_core::{Method, RequestRouter, Route, path_matches};

fn route_has(
    route: Option<&Route>,
    expected_path: &str,
    expected_handler: &str,
    expected_requires_model: bool,
    expected_method: Method,
) -> bool {
    route.is_some_and(|route| {
        route.path == expected_path
            && route.handler == expected_handler
            && route.requires_model == expected_requires_model
            && route.method == expected_method
    })
}

// ---------------------------------------------------------------------------
// Method
// ---------------------------------------------------------------------------

#[test]
fn method_as_str_matches_uppercase_http_verbs() {
    assert_eq!(Method::Get.as_str(), "GET");
    assert_eq!(Method::Post.as_str(), "POST");
    assert_eq!(Method::Put.as_str(), "PUT");
    assert_eq!(Method::Delete.as_str(), "DELETE");
}

#[test]
fn method_equality_and_copy_semantics() {
    let get = Method::Get;
    let get2 = get; // Copy
    assert_eq!(get, get2);
    assert_eq!(Method::Get, Method::Get);
    assert_ne!(Method::Get, Method::Post);
    assert_ne!(Method::Put, Method::Delete);
}

#[test]
fn method_debug_output_is_non_empty() {
    // We don't pin the exact format (derive(Debug) may evolve), but it
    // must contain the variant name so debug logs stay grep-able.
    let dbg = format!("{:?}", Method::Get);
    assert!(dbg.contains("Get"), "debug repr should mention variant: {dbg}");
    let dbg_post = format!("{:?}", Method::Post);
    assert!(dbg_post.contains("Post"), "debug repr should mention variant: {dbg_post}");
}

// ---------------------------------------------------------------------------
// path_matches
// ---------------------------------------------------------------------------

#[test]
fn path_matches_literal_paths() {
    assert!(path_matches("/v1/models", "/v1/models"));
    assert!(path_matches("/health", "/health"));
    assert!(path_matches("/", "/"));
}

#[test]
fn path_matches_rejects_different_segment_counts() {
    assert!(!path_matches("/v1/models", "/v1/models/phi-4"));
    assert!(!path_matches("/v1/models/:id", "/v1/models"));
    assert!(!path_matches("/a/b/c", "/a/b"));
    assert!(!path_matches("/a", "/a/b"));
}

#[test]
fn path_matches_supports_single_param_wildcard() {
    assert!(path_matches("/v1/models/:id", "/v1/models/phi-4"));
    assert!(path_matches("/v1/models/:id", "/v1/models/anything-goes_42"));
    // Characterization: wildcard segments currently match an empty string
    // because matching only checks segment count and the `:` prefix on the
    // pattern side.
    assert!(path_matches("/v1/models/:id", "/v1/models/"));
}

#[test]
fn path_matches_supports_multiple_param_wildcards() {
    assert!(path_matches("/users/:uid/posts/:pid", "/users/42/posts/hello"));
    assert!(!path_matches("/users/:uid/posts/:pid", "/users/42/posts"));
    assert!(!path_matches("/users/:uid/posts/:pid", "/users/42/comments/hello"));
}

#[test]
fn path_matches_is_case_sensitive_and_strict_on_trailing_slash() {
    // Trailing slash produces an extra empty segment, so the lengths differ.
    assert!(!path_matches("/v1/models", "/v1/models/"));
    assert!(!path_matches("/v1/models/", "/v1/models"));
    // Case sensitivity — the matcher does a byte-equal compare per segment.
    assert!(!path_matches("/v1/Models", "/v1/models"));
}

#[test]
fn path_matches_does_not_strip_query_strings() {
    // Precondition: callers pass a URI path with any query string already
    // stripped. The matcher treats query strings as part of the final segment.
    assert!(!path_matches("/v1/models", "/v1/models?foo=bar"));
    // But if the pattern uses a `:param` in that slot, it will absorb the
    // query string because any segment starting with `:` matches anything.
    assert!(path_matches("/v1/:wild", "/v1/models?foo=bar"));
}

#[test]
fn path_matches_handles_empty_strings() {
    // Both empty: split('/') -> [""], so segment counts match and the
    // single empty segment equals itself.
    assert!(path_matches("", ""));
    // Empty vs root: split('/') of "" -> [""], of "/" -> ["", ""].
    assert!(!path_matches("", "/"));
}

// ---------------------------------------------------------------------------
// RequestRouter::new + add_route
// ---------------------------------------------------------------------------

#[test]
fn new_router_starts_empty() {
    let router = RequestRouter::new("/api");
    assert_eq!(router.route_count(), 0);
    assert!(router.by_method(Method::Get).is_empty());
    assert!(router.model_routes().is_empty());
    assert!(router.table().is_empty());
    assert!(router.by_prefix().is_empty());
}

#[test]
fn default_router_has_empty_prefix() {
    let mut router = RequestRouter::default();
    router.add_route(Method::Get, "/ping", "ping", false);
    // With empty prefix, full path is exactly "/ping".
    let resolved = router.resolve(Method::Get, "/ping");
    assert!(route_has(resolved, "/ping", "ping", false, Method::Get));
}

#[test]
fn add_route_concatenates_prefix_verbatim() {
    let mut router = RequestRouter::new("/api");
    router.add_route(Method::Get, "/status", "status", false);
    router.add_route(Method::Post, "/jobs", "create_job", true);
    assert_eq!(router.route_count(), 2);

    assert!(route_has(
        router.resolve(Method::Get, "/api/status"),
        "/api/status",
        "status",
        false,
        Method::Get,
    ));
    assert!(route_has(
        router.resolve(Method::Post, "/api/jobs"),
        "/api/jobs",
        "create_job",
        true,
        Method::Post,
    ));
}

#[test]
fn prefix_with_trailing_slash_produces_double_slash() {
    // Documenting current behaviour: prefix is concatenated verbatim with no
    // normalization. A trailing slash on the prefix combined with a leading
    // slash on the path produces "//path". Callers should pick one form.
    let mut router = RequestRouter::new("/api/");
    router.add_route(Method::Get, "/status", "status", false);
    // The stored path is literally "/api//status".
    assert_eq!(router.table()[0].1, "/api//status");
    // It resolves only via the double-slash form, not the natural single-slash.
    assert!(router.resolve(Method::Get, "/api//status").is_some());
    assert!(router.resolve(Method::Get, "/api/status").is_none());
}

#[test]
fn empty_prefix_keeps_path_verbatim() {
    let mut router = RequestRouter::new("");
    router.add_route(Method::Delete, "/items/:id", "delete_item", false);
    assert!(route_has(
        router.resolve(Method::Delete, "/items/42"),
        "/items/:id",
        "delete_item",
        false,
        Method::Delete,
    ));
}

#[test]
fn duplicate_routes_are_appended_and_first_match_wins() {
    // Current behaviour: add_route does not de-dupe. resolve() iterates and
    // returns the first match, so a later duplicate is unreachable.
    let mut router = RequestRouter::new("");
    router.add_route(Method::Get, "/dup", "first", false);
    router.add_route(Method::Get, "/dup", "second", true);
    assert_eq!(router.route_count(), 2);
    assert!(route_has(router.resolve(Method::Get, "/dup"), "/dup", "first", false, Method::Get,));
    // The duplicate still shows up in the table and by_method.
    assert_eq!(router.by_method(Method::Get).len(), 2);
    assert_eq!(router.table().len(), 2);
}

#[test]
fn wildcard_routes_can_shadow_later_literal_routes() {
    // Characterization: resolve() is ordered, so a broad wildcard registered
    // before a literal route wins even when the literal route is more specific.
    let mut router = RequestRouter::new("");
    router.add_route(Method::Get, "/v1/:id", "wildcard", false);
    router.add_route(Method::Get, "/v1/models", "models", false);

    assert!(route_has(
        router.resolve(Method::Get, "/v1/models"),
        "/v1/:id",
        "wildcard",
        false,
        Method::Get,
    ));
}

// ---------------------------------------------------------------------------
// openai_compatible
// ---------------------------------------------------------------------------

#[test]
fn openai_compatible_has_at_least_canonical_endpoints() {
    let router = RequestRouter::openai_compatible();
    // Snapshot the headline count without enumerating every route, so this
    // remains stable as the route table grows.
    assert!(
        router.route_count() >= 7,
        "expected >= 7 OpenAI-compatible routes, got {}",
        router.route_count()
    );

    // Marker endpoints we rely on. If any of these break, downstream
    // server crates almost certainly break too.
    assert!(route_has(
        router.resolve(Method::Post, "/v1/chat/completions"),
        "/v1/chat/completions",
        "chat_completions",
        true,
        Method::Post,
    ));
    assert!(route_has(
        router.resolve(Method::Post, "/v1/completions"),
        "/v1/completions",
        "completions",
        true,
        Method::Post,
    ));
    assert!(route_has(
        router.resolve(Method::Get, "/v1/models"),
        "/v1/models",
        "list_models",
        false,
        Method::Get,
    ));
    assert!(route_has(
        router.resolve(Method::Post, "/v1/embeddings"),
        "/v1/embeddings",
        "embeddings",
        true,
        Method::Post,
    ));
}

#[test]
fn openai_compatible_exposes_unprefixed_health_endpoints() {
    let router = RequestRouter::openai_compatible();
    // Health endpoints live OUTSIDE the /v1 prefix.
    assert!(route_has(
        router.resolve(Method::Get, "/health"),
        "/health",
        "health",
        false,
        Method::Get,
    ));
    assert!(route_has(
        router.resolve(Method::Get, "/ready"),
        "/ready",
        "readiness",
        false,
        Method::Get,
    ));
}

#[test]
fn openai_compatible_resolves_model_id_wildcard() {
    let router = RequestRouter::openai_compatible();
    // Wildcard route should NOT require a model (only generation does).
    assert!(route_has(
        router.resolve(Method::Get, "/v1/models/microsoft-bitnet-b1.58-2B-4T"),
        "/v1/models/:id",
        "get_model",
        false,
        Method::Get,
    ));
}

// ---------------------------------------------------------------------------
// resolve
// ---------------------------------------------------------------------------

#[test]
fn resolve_returns_none_for_unknown_path() {
    let router = RequestRouter::openai_compatible();
    assert!(router.resolve(Method::Get, "/v1/nope").is_none());
    assert!(router.resolve(Method::Post, "/totally/made/up").is_none());
}

#[test]
fn resolve_distinguishes_method_for_same_path() {
    let mut router = RequestRouter::new("");
    router.add_route(Method::Get, "/thing", "get_thing", false);
    router.add_route(Method::Delete, "/thing", "delete_thing", false);

    assert!(route_has(
        router.resolve(Method::Get, "/thing"),
        "/thing",
        "get_thing",
        false,
        Method::Get,
    ));
    assert!(route_has(
        router.resolve(Method::Delete, "/thing"),
        "/thing",
        "delete_thing",
        false,
        Method::Delete,
    ));
    // No PUT handler was added, so it must not resolve.
    assert!(router.resolve(Method::Put, "/thing").is_none());
    assert!(router.resolve(Method::Post, "/thing").is_none());
}

#[test]
fn resolve_requires_full_prefixed_path() {
    let router = RequestRouter::openai_compatible();
    // Hitting the route without the /v1 prefix must miss.
    assert!(router.resolve(Method::Post, "/chat/completions").is_none());
    assert!(router.resolve(Method::Get, "/models").is_none());
    // Hitting it with the prefix works.
    assert!(router.resolve(Method::Post, "/v1/chat/completions").is_some());
}

// ---------------------------------------------------------------------------
// by_method / model_routes / table
// ---------------------------------------------------------------------------

#[test]
fn by_method_filters_routes_by_verb() {
    let router = RequestRouter::openai_compatible();
    let gets = router.by_method(Method::Get);
    let posts = router.by_method(Method::Post);

    // Every returned route must have the requested method.
    assert!(gets.iter().all(|r| r.method == Method::Get));
    assert!(posts.iter().all(|r| r.method == Method::Post));
    // GET set must include /v1/models and /health; POST set must include chat.
    assert!(gets.iter().any(|r| r.path == "/v1/models"));
    assert!(gets.iter().any(|r| r.path == "/health"));
    assert!(posts.iter().any(|r| r.path == "/v1/chat/completions"));
    // PUT was never added by the OpenAI builder.
    assert!(router.by_method(Method::Put).is_empty());
    assert!(router.by_method(Method::Delete).is_empty());
    // Partition must cover every route.
    assert_eq!(gets.len() + posts.len(), router.route_count());
}

#[test]
fn model_routes_only_returns_routes_that_require_a_model() {
    let router = RequestRouter::openai_compatible();
    let model_routes = router.model_routes();
    assert!(!model_routes.is_empty());
    assert!(model_routes.iter().all(|r| r.requires_model));
    // The current openai_compatible() builder marks exactly 3 routes as
    // requires_model: completions, chat/completions, embeddings.
    let handlers: Vec<&str> = model_routes.iter().map(|r| r.handler.as_str()).collect();
    assert!(handlers.contains(&"completions"));
    assert!(handlers.contains(&"chat_completions"));
    assert!(handlers.contains(&"embeddings"));
    // The non-model routes must NOT appear.
    assert!(!handlers.contains(&"health"));
    assert!(!handlers.contains(&"readiness"));
    assert!(!handlers.contains(&"list_models"));
    assert!(!handlers.contains(&"get_model"));
}

#[test]
fn table_row_count_matches_route_count_and_preserves_method_strings() {
    let router = RequestRouter::openai_compatible();
    let table = router.table();
    assert_eq!(table.len(), router.route_count());
    // Methods must be the uppercase HTTP verbs from Method::as_str.
    for (m, _, _) in &table {
        assert!(
            matches!(m.as_str(), "GET" | "POST" | "PUT" | "DELETE"),
            "unexpected method string in table row: {m}"
        );
    }
    // Health row should be exactly GET /health -> health.
    assert!(table.iter().any(|(m, p, h)| m == "GET" && p == "/health" && h == "health"));
    // Chat completions row should be POST /v1/chat/completions -> chat_completions.
    assert!(
        table
            .iter()
            .any(|(m, p, h)| m == "POST" && p == "/v1/chat/completions" && h == "chat_completions")
    );
}

// ---------------------------------------------------------------------------
// by_prefix
// ---------------------------------------------------------------------------

#[test]
fn by_prefix_groups_routes_by_first_two_path_segments() {
    // by_prefix takes the first three slash-separated tokens (which, for a
    // leading-slash path, gives "", "<seg1>", "<seg2>" -> joined "/seg1/seg2").
    let router = RequestRouter::openai_compatible();
    let groups = router.by_prefix();

    // /v1/* routes (completions, chat/completions, models, models/:id,
    // embeddings) split into groups by their second segment.
    assert!(
        groups
            .get("/v1/completions")
            .is_some_and(|routes| routes.iter().any(|r| r.handler == "completions"))
    );

    assert!(
        groups
            .get("/v1/chat")
            .is_some_and(|routes| routes.iter().any(|r| r.handler == "chat_completions"))
    );

    // Should contain both list_models AND get_model (because :id is the 3rd
    // segment and we only keep the first 3 tokens).
    assert!(groups.get("/v1/models").is_some_and(|routes| {
        let model_handlers: Vec<&str> = routes.iter().map(|r| r.handler.as_str()).collect();
        model_handlers.contains(&"list_models") && model_handlers.contains(&"get_model")
    }));

    assert!(
        groups
            .get("/v1/embeddings")
            .is_some_and(|routes| routes.iter().any(|r| r.handler == "embeddings"))
    );

    // Unprefixed health endpoints live under their own "/health" / "/ready"
    // keys because the path has only 2 tokens once split.
    assert!(
        groups.get("/health").is_some_and(|routes| routes.iter().any(|r| r.handler == "health"))
    );
    assert!(
        groups.get("/ready").is_some_and(|routes| routes.iter().any(|r| r.handler == "readiness"))
    );

    // Every route must end up in exactly one group, so the total count
    // across all groups equals the route count.
    let total: usize = groups.values().map(Vec::len).sum();
    assert_eq!(total, router.route_count());
}

#[test]
fn by_prefix_on_empty_router_returns_empty_map() {
    let router = RequestRouter::new("/api");
    assert!(router.by_prefix().is_empty());
}

#[test]
fn by_prefix_separates_multiple_top_level_prefixes() {
    let mut router = RequestRouter::new("");
    router.add_route(Method::Get, "/api/v1/users", "users", false);
    router.add_route(Method::Get, "/api/v2/users", "users_v2", false);
    router.add_route(Method::Get, "/admin/dashboard", "dashboard", false);
    let groups = router.by_prefix();
    // First 3 split tokens of "/api/v1/users" -> ["", "api", "v1"] -> "/api/v1"
    assert!(groups.contains_key("/api/v1"));
    assert!(groups.contains_key("/api/v2"));
    assert!(groups.contains_key("/admin/dashboard"));
    assert_eq!(groups.len(), 3);
}
