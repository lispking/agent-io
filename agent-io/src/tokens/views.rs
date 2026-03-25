//! Token views and types

pub use super::service::*;

/// Token count in a message
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenCount {
    pub tokens: u64,
    pub cached: bool,
}

impl TokenCount {
    pub fn new(tokens: u64) -> Self {
        Self {
            tokens,
            cached: false,
        }
    }

    pub fn cached(tokens: u64) -> Self {
        Self {
            tokens,
            cached: true,
        }
    }
}
