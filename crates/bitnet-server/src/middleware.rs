//! Production middleware for BitNet server.
//!
//! - **Correlation IDs** – generates or propagates `X-Request-ID` headers.
//! - **Metrics recording** – records request count, latency, and error rate
//!   into [`MetricsCollector`].
//! - **Connection limit** – semaphore-based cap on concurrent connections.

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::{HeaderName, HeaderValue, StatusCode},
    middleware::Next,
    response::Response,
};
use tokio::sync::Semaphore;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::monitoring::metrics::MetricsCollector;

/// Header used for request correlation.
static X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

/// Generates a `X-Request-ID` if the caller did not supply one, and copies it
/// to the response.
pub async fn correlation_id_middleware(mut request: Request, next: Next) -> Response {
    let id = request
        .headers()
        .get(&X_REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .map(String::from)
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    if let Ok(val) = HeaderValue::from_str(&id) {
        request.headers_mut().insert(X_REQUEST_ID.clone(), val);
    }

    let mut response = next.run(request).await;

    if let Ok(val) = HeaderValue::from_str(&id) {
        response.headers_mut().insert(X_REQUEST_ID.clone(), val);
    }

    response
}

/// Records request count, latency, and error rate into `MetricsCollector`.
pub async fn metrics_recording_middleware(
    State(metrics): State<Arc<MetricsCollector>>,
    request: Request,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let start = std::time::Instant::now();

    let response = next.run(request).await;

    let duration = start.elapsed();
    let status = response.status();

    metrics.record_request(method.as_ref(), &path, status.as_u16(), duration);

    if status.is_server_error() {
        debug!(
            method = %method,
            path = %path,
            status = %status,
            duration_ms = duration.as_millis(),
            "Server error recorded in metrics"
        );
    }

    response
}

/// Rejects requests with 503 when the connection semaphore is exhausted.
pub async fn connection_limit_middleware(
    State(semaphore): State<Arc<Semaphore>>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let _permit = semaphore.try_acquire().map_err(|_| {
        warn!(
            path = %request.uri().path(),
            "Connection limit reached – rejecting request"
        );
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use axum::routing::get;
    use axum::{Router, middleware as axum_mw};
    use tower::ServiceExt;

    async fn ok_handler() -> &'static str {
        "ok"
    }

    #[tokio::test]
    async fn correlation_id_generated_when_absent() {
        let app = Router::new()
            .route("/", get(ok_handler))
            .layer(axum_mw::from_fn(correlation_id_middleware));

        let resp = app.oneshot(HttpRequest::get("/").body(Body::empty()).unwrap()).await.unwrap();

        let id = resp.headers().get("x-request-id").expect("should have x-request-id");
        // Valid UUID v4
        assert!(Uuid::parse_str(id.to_str().unwrap()).is_ok());
    }

    #[tokio::test]
    async fn correlation_id_propagated_when_present() {
        let app = Router::new()
            .route("/", get(ok_handler))
            .layer(axum_mw::from_fn(correlation_id_middleware));

        let resp = app
            .oneshot(
                HttpRequest::get("/")
                    .header("x-request-id", "custom-123")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.headers().get("x-request-id").unwrap(), "custom-123");
    }

    #[tokio::test]
    async fn metrics_recording_runs_without_panic() {
        let config = crate::monitoring::MonitoringConfig::default();
        let metrics = Arc::new(MetricsCollector::new(&config).expect("metrics"));
        let app = Router::new()
            .route("/", get(ok_handler))
            .layer(axum_mw::from_fn_with_state(metrics, metrics_recording_middleware));

        let resp = app.oneshot(HttpRequest::get("/").body(Body::empty()).unwrap()).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn connection_limit_rejects_when_exhausted() {
        let sem = Arc::new(Semaphore::new(1));
        // Acquire the only permit so the middleware will fail
        let _hold = sem.clone().acquire_owned().await.expect("acquire");

        let app = Router::new()
            .route("/", get(ok_handler))
            .layer(axum_mw::from_fn_with_state(sem, connection_limit_middleware));

        let resp = app.oneshot(HttpRequest::get("/").body(Body::empty()).unwrap()).await.unwrap();

        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn connection_limit_allows_when_available() {
        let sem = Arc::new(Semaphore::new(10));

        let app = Router::new()
            .route("/", get(ok_handler))
            .layer(axum_mw::from_fn_with_state(sem, connection_limit_middleware));

        let resp = app.oneshot(HttpRequest::get("/").body(Body::empty()).unwrap()).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }
}
