//! Google Gemini API types

use serde::{Deserialize, Serialize};

/// Gemini API request
#[derive(Serialize)]
pub struct GeminiRequest {
    pub contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<GeminiTools>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Serialize)]
pub struct GeminiContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum GeminiPart {
    Text {
        text: String,
    },
    InlineData {
        inline_data: GeminiInlineData,
    },
    FunctionCall {
        function_call: GeminiFunctionCall,
    },
    FunctionResponse {
        function_response: GeminiFunctionResponse,
    },
    Thought {
        thought: String,
    },
}

#[derive(Serialize)]
pub struct GeminiInlineData {
    pub mime_type: String,
    pub data: String,
}

#[derive(Serialize)]
pub struct GeminiFunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Serialize)]
pub struct GeminiFunctionResponse {
    pub name: String,
    pub response: GeminiToolResult,
}

#[derive(Serialize)]
pub struct GeminiToolResult {
    pub name: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct GeminiTools {
    pub function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Serialize)]
pub struct GeminiFunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Map<String, serde_json::Value>,
}

#[derive(Serialize)]
pub struct GeminiGenerationConfig {
    pub temperature: f32,
    pub max_output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<GeminiThinkingConfig>,
}

#[derive(Serialize)]
pub struct GeminiThinkingConfig {
    pub thinking_budget: u64,
}

/// Gemini API response
#[derive(Deserialize)]
pub struct GeminiResponse {
    pub candidates: Vec<GeminiCandidate>,
    pub usage_metadata: Option<GeminiUsage>,
}

#[derive(Deserialize)]
pub struct GeminiCandidate {
    pub content: GeminiResponseContent,
    pub finish_reason: Option<String>,
}

#[derive(Deserialize)]
pub struct GeminiResponseContent {
    pub parts: Vec<GeminiResponsePart>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum GeminiResponsePart {
    Text {
        text: String,
    },
    Thought {
        thought: String,
    },
    FunctionCall {
        function_call: GeminiFunctionCallResponse,
    },
}

#[derive(Deserialize)]
pub struct GeminiFunctionCallResponse {
    pub name: String,
    pub args: serde_json::Value,
    #[serde(default)]
    pub id: Option<String>,
}

#[derive(Deserialize)]
pub struct GeminiUsage {
    pub prompt_token_count: u64,
    pub candidates_token_count: u64,
    pub total_token_count: u64,
    #[serde(default)]
    pub cached_content_token_count: u64,
}
