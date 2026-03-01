//! Anthropic API types

use serde::{Deserialize, Serialize};

/// Anthropic API request
#[derive(Serialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub max_tokens: u64,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

#[derive(Serialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u64>,
}

#[derive(Serialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Vec<AnthropicContent>,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text {
        text: String,
        #[serde(rename = "type")]
        content_type: String,
    },
    Image {
        source: AnthropicImageSource,
        #[serde(rename = "type")]
        content_type: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(rename = "type")]
        content_type: String,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(rename = "type")]
        content_type: String,
    },
}

#[derive(Serialize)]
pub struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

#[derive(Serialize)]
pub struct AnthropicTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Map<String, serde_json::Value>,
}

/// Anthropic API response
#[derive(Deserialize)]
pub struct AnthropicResponse {
    pub content: Vec<AnthropicResponseContent>,
    pub usage: AnthropicUsage,
    pub stop_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum AnthropicResponseContent {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: u64,
}
