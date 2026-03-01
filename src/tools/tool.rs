//! Core tool types and trait

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::Result;
use crate::llm::ToolDefinition;

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Tool call ID
    pub tool_call_id: String,
    /// Result content
    pub content: String,
    /// Whether this result should be ephemeral (removed after use)
    #[serde(default)]
    pub ephemeral: bool,
}

impl ToolResult {
    pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            ephemeral: false,
        }
    }

    pub fn with_ephemeral(mut self, ephemeral: bool) -> Self {
        self.ephemeral = ephemeral;
        self
    }
}

/// Trait for defining tools that can be called by an LLM
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get the tool name
    fn name(&self) -> &str;

    /// Get the tool description
    fn description(&self) -> &str;

    /// Get the tool definition (JSON Schema)
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with given arguments
    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult>;

    /// Whether tool outputs should be ephemeral (removed from context after use)
    fn ephemeral(&self) -> EphemeralConfig {
        EphemeralConfig::None
    }
}

/// Configuration for ephemeral tool outputs
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum EphemeralConfig {
    /// Not ephemeral
    #[default]
    None,
    /// Ephemeral, removed after one use
    Single,
    /// Keep last N outputs in context
    Count(usize),
}
