//! Cache control types for prompt caching

use serde::{Deserialize, Serialize};

/// Cache control for prompt caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub control_type: CacheControlType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheControlType {
    Ephemeral,
}

impl CacheControl {
    pub fn ephemeral() -> Self {
        Self {
            control_type: CacheControlType::Ephemeral,
        }
    }
}
