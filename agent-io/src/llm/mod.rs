//! LLM abstraction layer for multiple providers

mod anthropic;
mod base;
mod deepseek;
mod google;
mod groq;
mod mistral;
mod ollama;
mod openai;
mod openai_compatible;
mod openrouter;
mod schema;
mod types;

pub use anthropic::ChatAnthropic;
pub use base::*;
pub use deepseek::ChatDeepSeek;
pub use google::ChatGoogle;
pub use groq::ChatGroq;
pub use mistral::ChatMistral;
pub use ollama::ChatOllama;
pub use openai::ChatOpenAI;
pub use openai_compatible::ChatOpenAICompatible;
pub use openrouter::ChatOpenRouter;
pub use schema::SchemaOptimizer;
pub use types::*;

/// Error types for LLM operations
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("API error: {0}")]
    Api(String),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Configuration error: {0}")]
    Config(String),
}
