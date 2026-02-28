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
    let mut validation = Validation::new(Algorithm::HS256);
    // validation.required_spec_claims.insert("exp".to_string());
    println!("Validation: {:?}", validation);
}
