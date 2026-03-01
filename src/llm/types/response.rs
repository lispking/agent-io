//! Response types for LLM completions

use serde::{Deserialize, Serialize};

use super::tool::ToolCall;

/// Token usage information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cached_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_creation_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_image_tokens: Option<u64>,
}

impl Usage {
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            ..Default::default()
        }
    }
}

/// Stop reason for completion
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    StopSequence,
    ToolUse,
    MaxTokens,
    #[serde(other)]
    Unknown,
}

/// Chat completion response
#[derive(Debug, Clone)]
pub struct ChatCompletion {
    /// Text content (if any)
    pub content: Option<String>,
    /// Thinking content (for extended thinking models)
    pub thinking: Option<String>,
    /// Redacted thinking content
    pub redacted_thinking: Option<String>,
    /// Tool calls (if any)
    pub tool_calls: Vec<ToolCall>,
    /// Token usage
    pub usage: Option<Usage>,
    /// Why the completion stopped
    pub stop_reason: Option<StopReason>,
}

impl ChatCompletion {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            thinking: None,
            redacted_thinking: None,
            tool_calls: Vec::new(),
            usage: None,
            stop_reason: None,
        }
    }

    pub fn with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            content: None,
            thinking: None,
            redacted_thinking: None,
            tool_calls,
            usage: None,
            stop_reason: Some(StopReason::ToolUse),
        }
    }

    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    pub fn has_content(&self) -> bool {
        self.content.is_some() && self.content.as_ref().is_some_and(|c| !c.is_empty())
    }
}
