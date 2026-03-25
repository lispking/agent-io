//! OpenAI-compatible API types

use serde::{Deserialize, Serialize};

use crate::llm::ToolCall;

/// OpenAI-compatible API request
#[derive(Serialize)]
pub struct OpenAICompatibleRequest {
    pub model: String,
    pub messages: Vec<OpenAICompatibleMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAICompatibleTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Serialize)]
pub struct OpenAICompatibleMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Serialize)]
pub struct OpenAICompatibleTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAICompatibleFunction,
}

#[derive(Serialize)]
pub struct OpenAICompatibleFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Map<String, serde_json::Value>,
}

/// OpenAI-compatible API response
#[derive(Deserialize)]
pub struct OpenAICompatibleResponse {
    pub choices: Vec<OpenAICompatibleChoice>,
    #[serde(default)]
    pub usage: Option<OpenAICompatibleUsage>,
}

#[derive(Deserialize)]
pub struct OpenAICompatibleChoice {
    pub message: OpenAICompatibleMessageResponse,
    pub finish_reason: Option<String>,
}

#[derive(Deserialize)]
pub struct OpenAICompatibleMessageResponse {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize)]
pub struct OpenAICompatibleUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}
