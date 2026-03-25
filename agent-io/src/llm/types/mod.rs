//! Core types for LLM interactions
//!
//! This module provides the fundamental types used across all LLM providers:
//! - Tool definitions and function calls
//! - Content parts (text, images, documents, thinking)
//! - Message types (user, assistant, system, tool)
//! - Response types (completion, usage, stop reason)
//! - Cache control

mod cache;
mod content;
mod message;
mod response;
mod tool;

// Re-export all public types
pub use cache::{CacheControl, CacheControlType};
pub use content::{
    ContentPart, ContentPartDocument, ContentPartImage, ContentPartRedactedThinking,
    ContentPartRefusal, ContentPartText, ContentPartThinking, DocumentSource, ImageUrl, JsonSchema,
};
pub use message::{
    AssistantMessage, DeveloperMessage, Message, SystemMessage, ToolMessage, UserMessage,
};
pub use response::{ChatCompletion, StopReason, Usage};
pub use tool::{Function, ToolCall, ToolChoice, ToolDefinition};
