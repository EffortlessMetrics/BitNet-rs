use super::TokenRing;

impl std::fmt::Display for TokenRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TokenRing({}/{})", self.len, self.capacity)
    }
}
