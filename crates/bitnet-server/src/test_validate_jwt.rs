use jsonwebtoken::{Algorithm, DecodingKey, Validation, encode, Header, EncodingKey};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub exp: usize,
    pub iat: usize,
    pub role: Option<String>,
    pub rate_limit: Option<u64>,
}

fn main() {
    let claims = Claims {
        sub: "user".to_string(),
        exp: 10000000000,
        iat: 0,
        role: None,
        rate_limit: None,
    };
    let token = encode(&Header::default(), &claims, &EncodingKey::from_secret(b"secret")).unwrap();
    println!("Token: {}", token);
}
