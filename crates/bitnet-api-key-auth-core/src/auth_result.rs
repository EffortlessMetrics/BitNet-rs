/// Authentication result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthResult {
    /// Request is allowed.
    Allowed,
    /// Request was denied with a reason.
    Denied(String),
    /// No credential was supplied.
    NoCredentials,
}

impl AuthResult {
    /// Returns true if this result allows request processing.
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}
