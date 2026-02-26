//! Google Gemini Chat Model implementation

use async_trait::async_trait;
use derive_builder::Builder;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::llm::{
    BaseChatModel, ChatCompletion, ChatStream, ContentPart, LlmError, Message, StopReason,
    ToolChoice, ToolDefinition, Usage,
};

const GOOGLE_API_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Google Gemini Chat Model
#[derive(Builder, Clone)]
#[builder(pattern = "owned", build_fn(skip))]
pub struct ChatGoogle {
    /// Model identifier
    #[builder(setter(into))]
    model: String,
    /// API key
    api_key: String,
    /// Base URL
    #[builder(setter(into, strip_option), default = "None")]
    base_url: Option<String>,
    /// Maximum output tokens
    #[builder(default = "8192")]
    max_tokens: u64,
    /// Temperature for sampling
    #[builder(default = "0.2")]
    temperature: f32,
    /// Thinking budget (for thinking models)
    #[builder(default = "None")]
    thinking_budget: Option<u64>,
    /// HTTP client
    #[builder(setter(skip))]
    client: Client,
    /// Context window
    #[builder(setter(skip))]
    context_window: u64,
}

impl ChatGoogle {
    /// Create a new Google chat model
    pub fn new(model: impl Into<String>) -> Result<Self, LlmError> {
        let api_key = std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GEMINI_API_KEY"))
            .map_err(|_| LlmError::Config("GOOGLE_API_KEY or GEMINI_API_KEY not set".into()))?;

        Self::builder().model(model).api_key(api_key).build()
    }

    /// Create a builder for configuration
    pub fn builder() -> ChatGoogleBuilder {
        ChatGoogleBuilder::default()
    }

    /// Get the API URL for the model
    fn api_url(&self, stream: bool) -> String {
        let base = self.base_url.as_deref().unwrap_or(GOOGLE_API_URL);
        let method = if stream {
            "streamGenerateContent"
        } else {
            "generateContent"
        };
        format!("{}/{}:{}?key={}", base, self.model, method, self.api_key)
    }

    /// Build the HTTP client
    fn build_client() -> Client {
        Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client")
    }

    /// Get context window for model
    fn get_context_window(model: &str) -> u64 {
        let model_lower = model.to_lowercase();

        if model_lower.contains("gemini-1.5-pro") {
            2_097_152 // 2M tokens
        } else {
            1_048_576 // 1M tokens - default for most Gemini models
        }
    }

    /// Check if this is a thinking model
    fn is_thinking_model(&self) -> bool {
        let model_lower = self.model.to_lowercase();
        model_lower.contains("gemini-2.5")
            || model_lower.contains("thinking")
            || model_lower.contains("gemini-exp")
    }
}

impl ChatGoogleBuilder {
    pub fn build(&self) -> Result<ChatGoogle, LlmError> {
        let model = self
            .model
            .clone()
            .ok_or_else(|| LlmError::Config("model is required".into()))?;
        let api_key = self
            .api_key
            .clone()
            .ok_or_else(|| LlmError::Config("api_key is required".into()))?;

        Ok(ChatGoogle {
            context_window: ChatGoogle::get_context_window(&model),
            client: ChatGoogle::build_client(),
            model,
            api_key,
            base_url: self.base_url.clone().flatten(),
            max_tokens: self.max_tokens.unwrap_or(8192),
            temperature: self.temperature.unwrap_or(0.2),
            thinking_budget: self.thinking_budget.flatten(),
        })
    }
}

#[async_trait]
impl BaseChatModel for ChatGoogle {
    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &str {
        "google"
    }

    fn context_window(&self) -> Option<u64> {
        Some(self.context_window)
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
    ) -> Result<ChatCompletion, LlmError> {
        let request = self.build_request(messages, tools, tool_choice)?;

        let response = self
            .client
            .post(self.api_url(false))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::Api(format!(
                "Google API error ({}): {}",
                status, body
            )));
        }

        let completion: GeminiResponse = response.json().await?;
        Ok(self.parse_response(completion))
    }

    async fn invoke_stream(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
    ) -> Result<ChatStream, LlmError> {
        let request = self.build_request(messages, tools, tool_choice)?;

        let response = self
            .client
            .post(self.api_url(true))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::Api(format!(
                "Google API error ({}): {}",
                status, body
            )));
        }

        // Google returns JSON lines for streaming
        let stream = response.bytes_stream().filter_map(|result| async move {
            match result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    Self::parse_stream_chunk(&text)
                }
                Err(e) => Some(Err(LlmError::Stream(e.to_string()))),
            }
        });

        Ok(Box::pin(stream))
    }

    fn supports_vision(&self) -> bool {
        // All Gemini models support vision
        true
    }
}

// =============================================================================
// Request/Response Types
// =============================================================================

#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<GeminiTools>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum GeminiPart {
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
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Serialize)]
struct GeminiFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Serialize)]
struct GeminiFunctionResponse {
    name: String,
    response: GeminiToolResult,
}

#[derive(Serialize)]
struct GeminiToolResult {
    name: String,
    content: String,
}

#[derive(Serialize)]
struct GeminiTools {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Map<String, serde_json::Value>,
}

#[derive(Serialize)]
struct GeminiGenerationConfig {
    temperature: f32,
    max_output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
}

#[derive(Serialize)]
struct GeminiThinkingConfig {
    thinking_budget: u64,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiResponseContent,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct GeminiResponseContent {
    parts: Vec<GeminiResponsePart>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum GeminiResponsePart {
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
struct GeminiFunctionCallResponse {
    name: String,
    args: serde_json::Value,
    #[serde(default)]
    id: Option<String>,
}

#[derive(Deserialize)]
struct GeminiUsage {
    prompt_token_count: u64,
    candidates_token_count: u64,
    total_token_count: u64,
    #[serde(default)]
    cached_content_token_count: u64,
}

impl ChatGoogle {
    fn build_request(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        _tool_choice: Option<ToolChoice>,
    ) -> Result<GeminiRequest, LlmError> {
        let mut system_instruction: Option<GeminiContent> = None;
        let mut contents: Vec<GeminiContent> = Vec::new();

        for message in messages {
            match message {
                Message::System(s) => {
                    system_instruction = Some(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart::Text { text: s.content }],
                    });
                }
                Message::User(u) => {
                    let parts: Vec<GeminiPart> = u
                        .content
                        .into_iter()
                        .map(|c| match c {
                            ContentPart::Text(t) => GeminiPart::Text { text: t.text },
                            ContentPart::Image(img) => {
                                let (mime_type, data) = if img.image_url.url.starts_with("data:") {
                                    let parts: Vec<&str> =
                                        img.image_url.url.splitn(2, ',').collect();
                                    let mime = parts[0]
                                        .strip_prefix("data:")
                                        .and_then(|s| s.strip_suffix(";base64"))
                                        .unwrap_or("image/png");
                                    (mime.to_string(), parts.get(1).unwrap_or(&"").to_string())
                                } else {
                                    ("image/png".to_string(), img.image_url.url.clone())
                                };
                                GeminiPart::InlineData {
                                    inline_data: GeminiInlineData { mime_type, data },
                                }
                            }
                            _ => GeminiPart::Text {
                                text: "[Unsupported content]".to_string(),
                            },
                        })
                        .collect();

                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts,
                    });
                }
                Message::Assistant(a) => {
                    let mut parts = Vec::new();

                    if let Some(t) = a.thinking {
                        parts.push(GeminiPart::Thought { thought: t });
                    }

                    if let Some(c) = a.content {
                        parts.push(GeminiPart::Text { text: c });
                    }

                    for tc in a.tool_calls {
                        let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::json!({}));
                        parts.push(GeminiPart::FunctionCall {
                            function_call: GeminiFunctionCall {
                                name: tc.function.name,
                                args,
                            },
                        });
                    }

                    contents.push(GeminiContent {
                        role: "model".to_string(),
                        parts,
                    });
                }
                Message::Tool(t) => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart::FunctionResponse {
                            function_response: GeminiFunctionResponse {
                                name: "function_result".to_string(),
                                response: GeminiToolResult {
                                    name: "result".to_string(),
                                    content: t.content,
                                },
                            },
                        }],
                    });
                }
                Message::Developer(d) => {
                    system_instruction = Some(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart::Text { text: d.content }],
                    });
                }
            }
        }

        let gemini_tools = tools.map(|ts| GeminiTools {
            function_declarations: ts
                .into_iter()
                .map(|t| GeminiFunctionDeclaration {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                })
                .collect(),
        });

        let thinking_config = if self.is_thinking_model() {
            self.thinking_budget.map(|budget| GeminiThinkingConfig {
                thinking_budget: budget,
            })
        } else {
            None
        };

        Ok(GeminiRequest {
            contents,
            system_instruction,
            tools: gemini_tools,
            generation_config: Some(GeminiGenerationConfig {
                temperature: self.temperature,
                max_output_tokens: self.max_tokens,
                thinking_config,
            }),
        })
    }

    fn parse_response(&self, response: GeminiResponse) -> ChatCompletion {
        let stop_reason = response
            .candidates
            .first()
            .and_then(|c| c.finish_reason.as_ref())
            .and_then(|r| match r.as_str() {
                "STOP" => Some(StopReason::EndTurn),
                "MAX_TOKENS" => Some(StopReason::MaxTokens),
                "TOOL_CODE" => Some(StopReason::ToolUse),
                _ => None,
            });

        let candidate = response.candidates.into_iter().next();

        let (content, thinking, tool_calls) = candidate
            .map(|c| {
                let mut text: Option<String> = None;
                let mut think: Option<String> = None;
                let mut calls = Vec::new();

                for part in c.content.parts {
                    match part {
                        GeminiResponsePart::Text { text: t } => {
                            text = Some(t);
                        }
                        GeminiResponsePart::Thought { thought: t } => {
                            think = Some(t);
                        }
                        GeminiResponsePart::FunctionCall { function_call: fc } => {
                            let id = fc.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                            calls.push(crate::llm::ToolCall::new(
                                id,
                                fc.name,
                                serde_json::to_string(&fc.args).unwrap_or_default(),
                            ));
                        }
                    }
                }

                (text, think, calls)
            })
            .unwrap_or((None, None, Vec::new()));

        let usage = response.usage_metadata.map(|u| Usage {
            prompt_tokens: u.prompt_token_count,
            completion_tokens: u.candidates_token_count,
            total_tokens: u.total_token_count,
            prompt_cached_tokens: Some(u.cached_content_token_count),
            ..Default::default()
        });

        ChatCompletion {
            content,
            thinking,
            redacted_thinking: None,
            tool_calls,
            usage,
            stop_reason,
        }
    }

    fn parse_stream_chunk(text: &str) -> Option<Result<ChatCompletion, LlmError>> {
        // Google returns JSON array chunks
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Handle array wrapping
            let line = line.trim_start_matches('[').trim_end_matches(']');
            if line.is_empty() {
                continue;
            }

            // Handle comma-separated chunks
            for chunk_str in line.split("},") {
                let chunk_str = if !chunk_str.ends_with('}') {
                    format!("{}{}", chunk_str, "}")
                } else {
                    chunk_str.to_string()
                };

                let chunk: serde_json::Value = match serde_json::from_str(&chunk_str) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let parts = chunk
                    .get("candidates")?
                    .as_array()?
                    .first()?
                    .get("content")?
                    .get("parts")?
                    .as_array()?;

                for part in parts {
                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        return Some(Ok(ChatCompletion::text(text)));
                    }

                    if let Some(thought) = part.get("thought").and_then(|t| t.as_str()) {
                        let mut completion = ChatCompletion::text("");
                        completion.thinking = Some(thought.to_string());
                        return Some(Ok(completion));
                    }

                    if let Some(fc) = part.get("function_call") {
                        let name = fc.get("name")?.as_str()?.to_string();
                        let args = fc.get("args").cloned().unwrap_or(serde_json::json!({}));
                        let id = fc.get("id").and_then(|i| i.as_str()).unwrap_or("pending");

                        return Some(Ok(ChatCompletion {
                            content: None,
                            thinking: None,
                            redacted_thinking: None,
                            tool_calls: vec![crate::llm::ToolCall::new(
                                id,
                                name,
                                serde_json::to_string(&args).unwrap_or_default(),
                            )],
                            usage: None,
                            stop_reason: Some(StopReason::ToolUse),
                        }));
                    }
                }
            }
        }

        None
    }
}
