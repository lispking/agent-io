//! OpenAI-compatible API base implementation
//!
//! This module provides a base implementation for any LLM provider that uses
//! the OpenAI-compatible API format (Ollama, OpenRouter, DeepSeek, Groq, etc.)

use async_trait::async_trait;
use derive_builder::Builder;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::llm::{
    BaseChatModel, ChatCompletion, ChatStream, LlmError, Message, StopReason, ToolChoice,
    ToolDefinition, Usage,
};

/// OpenAI-compatible Chat Model base implementation
#[derive(Builder, Clone)]
#[builder(pattern = "owned", build_fn(skip))]
pub struct ChatOpenAICompatible {
    /// Model identifier
    #[builder(setter(into))]
    model: String,
    /// API key (optional for some providers like Ollama)
    #[builder(setter(into), default = "None")]
    api_key: Option<String>,
    /// Base URL for the API
    #[builder(setter(into))]
    base_url: String,
    /// Provider name for identification
    #[builder(setter(into))]
    provider: String,
    /// Temperature for sampling
    #[builder(default = "0.2")]
    temperature: f32,
    /// Maximum completion tokens
    #[builder(default = "Some(4096)")]
    max_completion_tokens: Option<u64>,
    /// HTTP client
    #[builder(setter(skip))]
    client: Client,
    /// Context window size
    #[builder(setter(skip))]
    context_window: u64,
    /// Whether to include Bearer prefix in auth header
    #[builder(default = "true")]
    use_bearer_auth: bool,
}

impl ChatOpenAICompatible {
    /// Create a builder for configuration
    pub fn builder() -> ChatOpenAICompatibleBuilder {
        ChatOpenAICompatibleBuilder::default()
    }

    /// Build the HTTP client
    fn build_client() -> Client {
        Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client")
    }

    /// Default context window
    fn default_context_window() -> u64 {
        128_000
    }

    /// Get the API URL
    fn api_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    /// Build request
    fn build_request(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
        stream: bool,
    ) -> Result<OpenAICompatibleRequest, LlmError> {
        let openai_messages: Vec<OpenAICompatibleMessage> =
            messages.into_iter().map(Self::convert_message).collect();

        let openai_tools = tools.map(|ts| {
            ts.into_iter()
                .map(|t| OpenAICompatibleTool {
                    tool_type: "function".to_string(),
                    function: OpenAICompatibleFunction {
                        name: t.name,
                        description: t.description,
                        parameters: t.parameters,
                    },
                })
                .collect()
        });

        let tool_choice_value = tool_choice.map(|tc| match tc {
            ToolChoice::Auto => serde_json::json!("auto"),
            ToolChoice::Required => serde_json::json!("required"),
            ToolChoice::None => serde_json::json!("none"),
            ToolChoice::Named(name) => {
                serde_json::json!({"type": "function", "function": {"name": name}})
            }
        });

        Ok(OpenAICompatibleRequest {
            model: self.model.clone(),
            messages: openai_messages,
            tools: openai_tools,
            tool_choice: tool_choice_value,
            temperature: Some(self.temperature),
            max_tokens: self.max_completion_tokens,
            stream: if stream { Some(true) } else { None },
        })
    }

    fn convert_message(message: Message) -> OpenAICompatibleMessage {
        match message {
            Message::User(u) => {
                let content = if u.content.len() == 1 && u.content[0].is_text() {
                    serde_json::json!(u.content[0].as_text().unwrap())
                } else {
                    serde_json::json!(u.content)
                };
                OpenAICompatibleMessage {
                    role: "user".to_string(),
                    content: Some(content),
                    name: u.name,
                    tool_calls: None,
                    tool_call_id: None,
                }
            }
            Message::Assistant(a) => OpenAICompatibleMessage {
                role: "assistant".to_string(),
                content: a.content.map(|c| serde_json::json!(c)),
                name: None,
                tool_calls: if a.tool_calls.is_empty() {
                    None
                } else {
                    Some(a.tool_calls)
                },
                tool_call_id: None,
            },
            Message::System(s) => OpenAICompatibleMessage {
                role: "system".to_string(),
                content: Some(serde_json::json!(s.content)),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message::Developer(d) => OpenAICompatibleMessage {
                role: "developer".to_string(),
                content: Some(serde_json::json!(d.content)),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message::Tool(t) => OpenAICompatibleMessage {
                role: "tool".to_string(),
                content: Some(serde_json::json!(t.content)),
                name: None,
                tool_calls: None,
                tool_call_id: Some(t.tool_call_id),
            },
        }
    }

    fn parse_response(response: OpenAICompatibleResponse) -> ChatCompletion {
        let stop_reason = response
            .choices
            .first()
            .and_then(|c| c.finish_reason.as_ref())
            .and_then(|r| match r.as_str() {
                "stop" => Some(StopReason::EndTurn),
                "tool_calls" => Some(StopReason::ToolUse),
                "length" => Some(StopReason::MaxTokens),
                _ => None,
            });

        let choice = response.choices.into_iter().next();

        let (content, tool_calls) = choice
            .map(|c| (c.message.content, c.message.tool_calls.unwrap_or_default()))
            .unwrap_or((None, Vec::new()));

        let usage = response.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            ..Default::default()
        });

        ChatCompletion {
            content,
            thinking: None,
            redacted_thinking: None,
            tool_calls,
            usage,
            stop_reason,
        }
    }

    fn parse_stream_chunk(text: &str) -> Option<Result<ChatCompletion, LlmError>> {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || !line.starts_with("data:") {
                continue;
            }

            let data = line.strip_prefix("data:").unwrap().trim();
            if data == "[DONE]" {
                return None;
            }

            let chunk: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let delta = chunk
                .get("choices")
                .and_then(|c| c.as_array())
                .and_then(|a| a.first())
                .and_then(|c| c.get("delta"));

            if let Some(delta) = delta {
                let content = delta
                    .get("content")
                    .and_then(|c| c.as_str())
                    .map(|s| s.to_string());

                let tool_calls: Vec<crate::llm::ToolCall> = delta
                    .get("tool_calls")
                    .and_then(|tc| tc.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|tc| {
                                let id = tc.get("id")?.as_str()?.to_string();
                                let func = tc.get("function")?;
                                let name = func.get("name")?.as_str()?.to_string();
                                let arguments = func.get("arguments")?.as_str()?.to_string();
                                Some(crate::llm::ToolCall::new(id, name, arguments))
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                if content.is_some() || !tool_calls.is_empty() {
                    return Some(Ok(ChatCompletion {
                        content,
                        thinking: None,
                        redacted_thinking: None,
                        tool_calls,
                        usage: None,
                        stop_reason: None,
                    }));
                }
            }
        }

        None
    }
}

impl ChatOpenAICompatibleBuilder {
    pub fn build(&self) -> Result<ChatOpenAICompatible, LlmError> {
        let model = self
            .model
            .clone()
            .ok_or_else(|| LlmError::Config("model is required".into()))?;
        let base_url = self
            .base_url
            .clone()
            .ok_or_else(|| LlmError::Config("base_url is required".into()))?;
        let provider = self
            .provider
            .clone()
            .ok_or_else(|| LlmError::Config("provider is required".into()))?;

        Ok(ChatOpenAICompatible {
            client: ChatOpenAICompatible::build_client(),
            context_window: ChatOpenAICompatible::default_context_window(),
            model,
            api_key: self.api_key.clone().flatten(),
            base_url,
            provider,
            temperature: self.temperature.unwrap_or(0.2),
            max_completion_tokens: self.max_completion_tokens.flatten(),
            use_bearer_auth: self.use_bearer_auth.unwrap_or(true),
        })
    }
}

#[async_trait]
impl BaseChatModel for ChatOpenAICompatible {
    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &str {
        &self.provider
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
            .header("Content-Type", "application/json");

        if let Some(ref api_key) = self.api_key {
            if self.use_bearer_auth {
                req = req.header("Authorization", format!("Bearer {}", api_key));
            } else {
                req = req.header("Authorization", api_key.clone());
            }
        }

        let response = req.json(&request).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::Api(format!(
                "{} API error ({}): {}",
                self.provider, status, body
            )));
        }

        let completion: OpenAICompatibleResponse = response.json().await?;
        Ok(Self::parse_response(completion))
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
            .header("Content-Type", "application/json");

        if let Some(ref api_key) = self.api_key {
            if self.use_bearer_auth {
                req = req.header("Authorization", format!("Bearer {}", api_key));
            } else {
                req = req.header("Authorization", api_key.clone());
            }
        }

        let response = req.json(&request).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::Api(format!(
                "{} API error ({}): {}",
                self.provider, status, body
            )));
        }

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
        // Most OpenAI-compatible providers support vision
        true
    }
}

// =============================================================================
// Request/Response Types
// =============================================================================

#[derive(Serialize)]
struct OpenAICompatibleRequest {
    model: String,
    messages: Vec<OpenAICompatibleMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAICompatibleTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize)]
struct OpenAICompatibleMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<crate::llm::ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize)]
struct OpenAICompatibleTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAICompatibleFunction,
}

#[derive(Serialize)]
struct OpenAICompatibleFunction {
    name: String,
    description: String,
    parameters: serde_json::Map<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct OpenAICompatibleResponse {
    choices: Vec<OpenAICompatibleChoice>,
    #[serde(default)]
    usage: Option<OpenAICompatibleUsage>,
}

#[derive(Deserialize)]
struct OpenAICompatibleChoice {
    message: OpenAICompatibleMessageResponse,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAICompatibleMessageResponse {
    content: Option<String>,
    tool_calls: Option<Vec<crate::llm::ToolCall>>,
}

#[derive(Deserialize)]
struct OpenAICompatibleUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}
