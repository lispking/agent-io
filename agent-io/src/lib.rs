//! # BU Agent SDK
//!
//! A Rust SDK for building AI agents with multi-provider LLM support.
//!
//! ## Features
//!
//! - Multi-provider LLM support (OpenAI, Anthropic, Google Gemini)
//! - Tool/function calling with dependency injection
//! - Streaming responses with event-based architecture
//! - Context compaction for long-running conversations
//! - Token usage tracking and cost calculation
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use agent_io::{Agent, llm::ChatOpenAI, tools::FunctionTool};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let llm = ChatOpenAI::new("gpt-4o")?;
//!     let agent = Agent::builder()
//!         .with_llm(Arc::new(llm))
//!         .build()?;
//!     
//!     let response = agent.query("Hello!").await?;
//!     println!("{}", response);
//!     Ok(())
//! }
//! ```

pub mod agent;
pub mod llm;
pub mod memory;
pub mod observability;
pub mod tokens;
pub mod tools;

/// Re-exports needed by the `#[tool]` proc-macro. Not part of the public API.
#[doc(hidden)]
pub mod __macro_support {
    pub use async_trait::async_trait;
    pub use serde;
    pub use serde_json;
}

// Re-export the `#[tool]` attribute macro
pub use agent_io_macros::tool;

pub use agent::{Agent, AgentEvent};
pub use llm::BaseChatModel;
pub use memory::{EmbeddingProvider, InMemoryStore, MemoryManager, MemoryStore};
pub use observability::*;
pub use tokens::TokenCost;
pub use tools::Tool;

/// Result type alias for SDK operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for the SDK
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("LLM error: {0}")]
    Llm(#[from] llm::LlmError),

    #[error("Tool error: {0}")]
    Tool(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Agent error: {0}")]
    Agent(String),

    #[error("Max iterations exceeded")]
    MaxIterationsExceeded,
}
