use axum::{
    Json, Router,
    body::Body,
    extract::Extension,
    http::{Request, StatusCode},
    middleware,
    routing::get,
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use bitnet_server::security::{AuthState, Claims, SecurityConfig, auth_middleware};
use http_body_util::BodyExt;
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use tower::ServiceExt;
use uselesskey::{Factory, HmacFactoryExt, HmacSpec, Seed};
use uselesskey_jsonwebtoken::JwtKeyExt;

const USELESSKEY_TEST_SEED: &str = "bitnet-server/tests/uselesskey-jwt-auth";
const BASE64_SECRET_PREFIX: &str = "base64:";

fn deterministic_hs256_secret(label: &str) -> uselesskey::HmacSecret {
    let seed = Seed::from_env_value(USELESSKEY_TEST_SEED).expect("stable uselesskey seed");
    let factory = Factory::deterministic(seed);
    factory.hmac(label, HmacSpec::hs256())
}

fn test_claims() -> Claims {
    Claims {
        sub: "fixture-user".to_string(),
        exp: 4_102_444_800,
        iat: 1_700_000_000,
        role: Some("admin".to_string()),
        rate_limit: Some(128),
    }
}

fn auth_state(secret: String) -> AuthState {
    AuthState {
        config: SecurityConfig {
            require_authentication: true,
            jwt_secret: Some(secret.clone()),
            ..SecurityConfig::default()
        },
        jwt_secret: Some(secret),
    }
}

fn protected_app(secret: String) -> Router {
    Router::new()
        .route(
            "/protected",
            get(|Extension(claims): Extension<Claims>| async move { Json(claims) }),
        )
        .layer(middleware::from_fn_with_state(auth_state(secret), auth_middleware))
}

fn base64_prefixed_secret(label: &str) -> String {
    let secret = deterministic_hs256_secret(label);
    format!("{BASE64_SECRET_PREFIX}{}", STANDARD.encode(secret.secret_bytes()))
}

fn sign_uselesskey_token(label: &str, claims: &Claims) -> String {
    let secret = deterministic_hs256_secret(label);
    encode(&Header::new(Algorithm::HS256), claims, &secret.encoding_key())
        .expect("HS256 token fixture")
}

async fn authorized_request(app: Router, token: &str) -> axum::response::Response {
    app.oneshot(
        Request::builder()
            .uri("/protected")
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .expect("request fixture"),
    )
    .await
    .expect("router response")
}

#[tokio::test]
async fn auth_middleware_accepts_base64_prefixed_uselesskey_secret() {
    let claims = test_claims();
    let token = sign_uselesskey_token("auth-success", &claims);
    let response =
        authorized_request(protected_app(base64_prefixed_secret("auth-success")), &token).await;

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.expect("claims body").to_bytes();
    let round_tripped: Claims = serde_json::from_slice(&body).expect("claims json");
    assert_eq!(round_tripped.sub, claims.sub);
    assert_eq!(round_tripped.role, claims.role);
    assert_eq!(round_tripped.rate_limit, claims.rate_limit);
}

#[tokio::test]
async fn auth_middleware_rejects_token_signed_with_different_uselesskey_secret() {
    let claims = test_claims();
    let token = sign_uselesskey_token("issuer-a", &claims);
    let response =
        authorized_request(protected_app(base64_prefixed_secret("issuer-b")), &token).await;

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn auth_middleware_keeps_plaintext_secret_support() {
    let claims = test_claims();
    let secret = "legacy-plain-text-secret";
    let token = encode(
        &Header::new(Algorithm::HS256),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .expect("legacy plaintext token");

    let response = authorized_request(protected_app(secret.to_string()), &token).await;

    assert_eq!(response.status(), StatusCode::OK);
}
