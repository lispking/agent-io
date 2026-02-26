//! Anthropic Claude Chat Model implementation

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

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";

/// Anthropic Chat Model
#[derive(Builder, Clone)]
#[builder(pattern = "owned", build_fn(skip))]
pub struct ChatAnthropic {
    /// Model identifier
    #[builder(setter(into))]
    model: String,
    /// API key
    api_key: String,
    /// Base URL (for proxies)
    #[builder(setter(into, strip_option), default = "None")]
    base_url: Option<String>,
    /// Maximum output tokens
    #[builder(default = "8192")]
    max_tokens: u64,
    /// Temperature for sampling
    #[builder(default = "0.2")]
    temperature: f32,
    /// Prompt cache beta header
    #[builder(default = r#"Some("prompt-caching-2024-07-31".to_string())"#)]
    prompt_cache_beta: Option<String>,
    /// Enable extended thinking
    #[builder(default = "false")]
    thinking: bool,
    /// Thinking budget (tokens)
    #[builder(default = "Some(1024)")]
    thinking_budget: Option<u64>,
    /// HTTP client
    #[builder(setter(skip))]
    client: Client,
    /// Context window
    #[builder(setter(skip))]
    context_window: u64,
}

impl ChatAnthropic {
    /// Create a new Anthropic chat model
    pub fn new(model: impl Into<String>) -> Result<Self, LlmError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| LlmError::Config("ANTHROPIC_API_KEY not set".into()))?;

        Self::builder().model(model).api_key(api_key).build()
    }

    /// Create a builder for configuration
    pub fn builder() -> ChatAnthropicBuilder {
        ChatAnthropicBuilder::default()
    }

    /// Get the API URL
    fn api_url(&self) -> &str {
        self.base_url.as_deref().unwrap_or(ANTHROPIC_API_URL)
    }

    /// Build the HTTP client
    fn build_client() -> Client {
        Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client")
    }

    /// Get context window for model
    fn get_context_window(_model: &str) -> u64 {
        // All Claude models have 200k context window
        200_000
    }

    /// Check if model supports extended thinking
    fn supports_thinking(&self) -> bool {
        let model_lower = self.model.to_lowercase();
        model_lower.contains("claude-3-7-sonnet")
            || model_lower.contains("claude-3.7")
            || model_lower.contains("claude-4")
    }
}

impl ChatAnthropicBuilder {
    pub fn build(&self) -> Result<ChatAnthropic, LlmError> {
        let model = self
            .model
            .clone()
            .ok_or_else(|| LlmError::Config("model is required".into()))?;
        let api_key = self
            .api_key
            .clone()
            .ok_or_else(|| LlmError::Config("api_key is required".into()))?;

        Ok(ChatAnthropic {
            context_window: ChatAnthropic::get_context_window(&model),
            client: ChatAnthropic::build_client(),
            model,
            api_key,
            base_url: self.base_url.clone().flatten(),
            max_tokens: self.max_tokens.unwrap_or(8192),
            temperature: self.temperature.unwrap_or(0.2),
            prompt_cache_beta: self.prompt_cache_beta.clone().flatten(),
            thinking: self.thinking.unwrap_or(false),
            thinking_budget: self.thinking_budget.flatten(),
        })
    }
}

#[async_trait]
impl BaseChatModel for ChatAnthropic {
    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &str {
        "anthropic"
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
        let request = self.build_request(messages, tools, tool_choice, false)?;

        let mut req = self
            .client
            .post(self.api_url())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json");

        if let Some(ref beta) = self.prompt_cache_beta {
            req = req.header("anthropic-beta", beta.as_str());
        }

        let response = req.json(&request).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::Api(format!(
                "Anthropic API error ({}): {}",
                status, body
            )));
        }

        let completion: AnthropicResponse = response.json().await?;
        Ok(self.parse_response(completion))
    }

    async fn invoke_stream(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
    ) -> Result<ChatStream, LlmError> {
        let request = self.build_request(messages, tools, tool_choice, true)?;

        let mut req = self
            .client
            .post(self.api_url())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json");

        if let Some(ref beta) = self.prompt_cache_beta {
            req = req.header("anthropic-beta", beta.as_str());
        }

        let response = req.json(&request).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::Api(format!(
                "Anthropic API error ({}): {}",
                status, body
            )));
        }

        // Parse SSE stream
        let stream = response.bytes_stream().filter_map(|result| async move {
            match result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    Self::parse_sse_event(&text)
                }
                Err(e) => Some(Err(LlmError::Stream(e.to_string()))),
            }
        });

        Ok(Box::pin(stream))
    }

    fn supports_vision(&self) -> bool {
        // All Claude 3+ models support vision
        true
    }
}

// =============================================================================
// Request/Response Types
// =============================================================================

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u64,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
}

#[derive(Serialize)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    budget_tokens: Option<u64>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum AnthropicContent {
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
struct AnthropicImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Map<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicResponseContent>,
    usage: AnthropicUsage,
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum AnthropicResponseContent {
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
struct AnthropicUsage {
    input_tokens: u64,
    output_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
}

impl ChatAnthropic {
    fn build_request(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
        stream: bool,
    ) -> Result<AnthropicRequest, LlmError> {
        // Separate system message from other messages
        let mut system: Option<String> = None;
        let mut anthropic_messages: Vec<AnthropicMessage> = Vec::new();

        for message in messages {
            match message {
                Message::System(s) => {
                    system = Some(s.content);
                }
                Message::User(u) => {
                    let content: Vec<AnthropicContent> = u
                        .content
                        .into_iter()
                        .map(|c| match c {
                            ContentPart::Text(t) => AnthropicContent::Text {
                                text: t.text,
                                content_type: "text".to_string(),
                            },
                            ContentPart::Image(img) => {
                                let (media_type, data) = if img.image_url.url.starts_with("data:") {
                                    // Parse data URL
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
                                AnthropicContent::Image {
                                    source: AnthropicImageSource {
                                        source_type: "base64".to_string(),
                                        media_type,
                                        data,
                                    },
                                    content_type: "image".to_string(),
                                }
                            }
                            _ => AnthropicContent::Text {
                                text: "[Unsupported content type]".to_string(),
                                content_type: "text".to_string(),
                            },
                        })
                        .collect();

                    anthropic_messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content,
                    });
                }
                Message::Assistant(a) => {
                    let mut content = Vec::new();

                    if let Some(t) = a.thinking {
                        content.push(AnthropicContent::Text {
                            text: t,
                            content_type: "thinking".to_string(),
                        });
                    }

                    if let Some(c) = a.content {
                        content.push(AnthropicContent::Text {
                            text: c,
                            content_type: "text".to_string(),
                        });
                    }

                    for tc in a.tool_calls {
                        let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::json!({}));
                        content.push(AnthropicContent::ToolUse {
                            id: tc.id,
                            name: tc.function.name,
                            input,
                            content_type: "tool_use".to_string(),
                        });
                    }

                    anthropic_messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content,
                    });
                }
                Message::Tool(t) => {
                    anthropic_messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: vec![AnthropicContent::ToolResult {
                            tool_use_id: t.tool_call_id,
                            content: t.content,
                            content_type: "tool_result".to_string(),
                        }],
                    });
                }
                Message::Developer(d) => {
                    system = Some(d.content);
                }
            }
        }

        let anthropic_tools = tools.map(|ts| {
            ts.into_iter()
                .map(|t| AnthropicTool {
                    name: t.name,
                    description: t.description,
                    input_schema: t.parameters,
                })
                .collect()
        });

        let tool_choice_value = tool_choice.map(|tc| match tc {
            ToolChoice::Auto => serde_json::json!({"type": "auto"}),
            ToolChoice::Required => serde_json::json!({"type": "any"}),
            ToolChoice::None => serde_json::json!({"type": "none"}),
            ToolChoice::Named(name) => serde_json::json!({"type": "tool", "name": name}),
        });

        let thinking_config = if self.thinking && self.supports_thinking() {
            Some(ThinkingConfig {
                thinking_type: "enabled".to_string(),
                budget_tokens: self.thinking_budget,
            })
        } else {
            None
        };

        Ok(AnthropicRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            messages: anthropic_messages,
            system,
            tools: anthropic_tools,
            tool_choice: tool_choice_value,
            temperature: Some(self.temperature),
            stream: if stream { Some(true) } else { None },
            thinking: thinking_config,
        })
    }

    fn parse_response(&self, response: AnthropicResponse) -> ChatCompletion {
        let mut content: Option<String> = None;
        let mut thinking: Option<String> = None;
        let mut redacted_thinking: Option<String> = None;
        let mut tool_calls = Vec::new();

        for c in response.content {
            match c {
                AnthropicResponseContent::Text { text } => {
                    content = Some(text);
                }
                AnthropicResponseContent::Thinking { thinking: t } => {
                    thinking = Some(t);
                }
                AnthropicResponseContent::RedactedThinking { data } => {
                    redacted_thinking = Some(data);
                }
                AnthropicResponseContent::ToolUse { id, name, input } => {
                    tool_calls.push(crate::llm::ToolCall::new(
                        id,
                        name,
                        serde_json::to_string(&input).unwrap_or_default(),
                    ));
                }
            }
        }

        let stop_reason = response.stop_reason.and_then(|r| match r.as_str() {
            "end_turn" => Some(StopReason::EndTurn),
            "tool_use" => Some(StopReason::ToolUse),
            "max_tokens" => Some(StopReason::MaxTokens),
            _ => None,
        });

        let usage = Usage {
            prompt_tokens: response.usage.input_tokens,
            completion_tokens: response.usage.output_tokens,
            total_tokens: response.usage.input_tokens + response.usage.output_tokens,
            prompt_cached_tokens: Some(response.usage.cache_read_input_tokens),
            prompt_cache_creation_tokens: Some(response.usage.cache_creation_input_tokens),
            ..Default::default()
        };

        ChatCompletion {
            content,
            thinking,
            redacted_thinking,
            tool_calls,
            usage: Some(usage),
            stop_reason,
        }
    }

    fn parse_sse_event(text: &str) -> Option<Result<ChatCompletion, LlmError>> {
        for line in text.lines() {
            let line = line.trim();

            if !line.starts_with("data:") && !line.starts_with("data: ") {
                continue;
            }

            let data = line.strip_prefix("data:").unwrap().trim();

            let event: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let event_type = event.get("type")?.as_str()?;

            match event_type {
                "content_block_delta" => {
                    let delta = event.get("delta")?;
                    let delta_type = delta.get("type")?.as_str()?;

                    match delta_type {
                        "text_delta" => {
                            let text = delta.get("text")?.as_str()?;
                            return Some(Ok(ChatCompletion::text(text)));
                        }
                        "thinking_delta" => {
                            let thinking = delta.get("thinking")?.as_str()?;
                            let mut completion = ChatCompletion::text("");
                            completion.thinking = Some(thinking.to_string());
                            return Some(Ok(completion));
                        }
                        "input_json_delta" => {
                            let partial = delta.get("partial_json")?.as_str()?;
                            let index = event.get("index")?.as_u64()? as usize;
                            return Some(Ok(ChatCompletion {
                                content: None,
                                thinking: None,
                                redacted_thinking: None,
                                tool_calls: vec![crate::llm::ToolCall::new(
                                    format!("pending_{}", index),
                                    "pending",
                                    partial.to_string(),
                                )],
                                usage: None,
                                stop_reason: None,
                            }));
                        }
                        _ => {}
                    }
                }
                "message_start" | "message_stop" | "content_block_start" | "content_block_stop" => {
                    return None;
                }
                _ => {}
            }
        }

        None
    }
}
