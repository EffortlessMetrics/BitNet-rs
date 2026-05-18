use super::status::{is_claimable_status, is_known_status, status_requires_receipts};
use super::types::{CRITICAL_NOT_CLAIMS, RouteEntry, RouteQuery, RouteResolveReport, RouteTable};
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

pub(super) fn build_route_resolve_report(
    routing_table: &Path,
    query: RouteQuery,
) -> Result<RouteResolveReport> {
    let raw = fs::read_to_string(routing_table)
        .with_context(|| format!("reading {}", routing_table.display()))?;
    let table: RouteTable =
        toml::from_str(&raw).with_context(|| format!("parsing {}", routing_table.display()))?;

    let mut failures = validate_route_table(&table.route);
    let table_verified = failures.is_empty();
    let route = table
        .route
        .iter()
        .find(|route| {
            route.device_slug == query.device_slug
                && route.selected_backend == query.selected_backend
                && route.backend_family == query.backend_family
                && route.model_family == query.model_family
                && route.quantization == query.quantization
                && route.op == query.op
        })
        .cloned();

    let mut not_claims = CRITICAL_NOT_CLAIMS.iter().map(|value| (*value).to_string()).collect();
    let mut route_verified = false;
    let mut claimable = false;
    let classification;

    if let Some(route) = &route {
        claimable = is_claimable_status(&route.claim_level);
        route_verified = table_verified && validate_route_entry(route).is_empty();
        if !route.not_claims.is_empty() {
            not_claims = route.not_claims.clone();
        }
        classification = if claimable {
            "claimable_route"
        } else if route.claim_level == "unsupported" {
            "unsupported_route"
        } else {
            "diagnostic_route"
        }
        .to_string();
    } else {
        failures.push("no matching route".to_string());
        classification = "route_missing".to_string();
    }

    Ok(RouteResolveReport {
        diagnostic: "kernel_route_resolve",
        producer: "cargo xtask hardware route resolve",
        routing_table: routing_table.display().to_string(),
        query,
        passed: failures.is_empty(),
        route_found: route.is_some(),
        route_verified,
        claimable,
        classification,
        route,
        failures,
        not_claims,
    })
}

fn validate_route_table(routes: &[RouteEntry]) -> Vec<String> {
    let mut failures = Vec::new();
    if routes.is_empty() {
        failures.push("routing table has no routes".to_string());
    }
    for route in routes {
        failures.extend(validate_route_entry(route));
    }
    failures
}

fn validate_route_entry(route: &RouteEntry) -> Vec<String> {
    let mut failures = Vec::new();
    if route.route_id.trim().is_empty() {
        failures.push("route entry has empty route_id".to_string());
    }
    if route.device_slug == "*" {
        failures.push(format!("{} uses wildcard device_slug", route.route_id));
    }
    if route.device_models.is_empty() {
        failures.push(format!("{} has no device_models", route.route_id));
    }
    if route.device_models.iter().any(|model| model == "*") {
        failures.push(format!("{} uses wildcard device_models", route.route_id));
    }
    if route.kernel_variant.trim().is_empty() {
        failures.push(format!("{} has empty kernel_variant", route.route_id));
    }
    if route.kernel_variant == "missing" && route.claim_level != "unsupported" {
        failures.push(format!(
            "{} uses missing kernel_variant but claim_level={}",
            route.route_id, route.claim_level
        ));
    }
    if !is_known_status(&route.claim_level) {
        failures.push(format!("{} has unknown claim_level {}", route.route_id, route.claim_level));
    }
    let claimable = is_claimable_status(&route.claim_level);
    if claimable {
        failures.push(format!(
            "{} is claimable; A770 route rail requires diagnostic or unsupported routes",
            route.route_id
        ));
    }
    if claimable && route.fallback_allowed {
        failures.push(format!("{} is claimable but fallback_allowed=true", route.route_id));
    }
    if status_requires_receipts(&route.claim_level) && route.proof_receipts.is_empty() {
        failures.push(format!(
            "{} claim_level {} requires proof_receipts",
            route.route_id, route.claim_level
        ));
    }
    failures
}
