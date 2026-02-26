//! OpenAI Chat Model implementation

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

const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Reasoning effort levels for o1+ models
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
    #[default]
    Minimal,
}

/// OpenAI Chat Model
#[derive(Builder, Clone)]
#[builder(pattern = "owned", build_fn(skip))]
pub struct ChatOpenAI {
    /// Model identifier
    #[builder(setter(into))]
    model: String,
    /// API key
    api_key: String,
    /// Base URL (for proxies)
    #[builder(setter(into, strip_option), default = "None")]
    base_url: Option<String>,
    /// Temperature for sampling
    #[builder(default = "0.2")]
    temperature: f32,
    /// Maximum completion tokens
    #[builder(default = "Some(4096)")]
    max_completion_tokens: Option<u64>,
    /// Reasoning effort for o1+ models
    #[builder(default = "ReasoningEffort::Low")]
    reasoning_effort: ReasoningEffort,
    /// HTTP client
    #[builder(setter(skip))]
    client: Client,
    /// Context window size
    #[builder(setter(skip))]
    context_window: u64,
}

impl ChatOpenAI {
    /// Create a new OpenAI chat model
    pub fn new(model: impl Into<String>) -> Result<Self, LlmError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| LlmError::Config("OPENAI_API_KEY not set".into()))?;

        Self::builder().model(model).api_key(api_key).build()
    }

    /// Create a builder for configuration
    pub fn builder() -> ChatOpenAIBuilder {
        ChatOpenAIBuilder::default()
    }

    /// Check if this is a reasoning model (o1, o3, o4, gpt-5)
    fn is_reasoning_model(&self) -> bool {
        let model_lower = self.model.to_lowercase();
        model_lower.starts_with("o1")
            || model_lower.starts_with("o3")
            || model_lower.starts_with("o4")
            || model_lower.starts_with("gpt-5")
    }

    /// Get the API URL
    fn api_url(&self) -> &str {
        self.base_url.as_deref().unwrap_or(OPENAI_API_URL)
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

        // GPT-4o family
        if model_lower.contains("gpt-4o") || model_lower.contains("gpt-4-turbo") {
            128_000
        }
        // GPT-4
        else if model_lower.starts_with("gpt-4") {
            8_192
        }
        // GPT-3.5
        else if model_lower.starts_with("gpt-3.5") {
            16_385
        }
        // O1/O3/O4 reasoning models
        else if model_lower.starts_with("o1")
            || model_lower.starts_with("o3")
            || model_lower.starts_with("o4")
        {
            200_000
        }
        // Default
        else {
            128_000
        }
    }
}

impl ChatOpenAIBuilder {
    pub fn build(&self) -> Result<ChatOpenAI, LlmError> {
        let model = self
            .model
            .clone()
            .ok_or_else(|| LlmError::Config("model is required".into()))?;
        let api_key = self
            .api_key
            .clone()
            .ok_or_else(|| LlmError::Config("api_key is required".into()))?;

        Ok(ChatOpenAI {
            context_window: ChatOpenAI::get_context_window(&model),
            client: ChatOpenAI::build_client(),
            model,
            api_key,
            base_url: self.base_url.clone().flatten(),
            temperature: self.temperature.unwrap_or(0.2),
            max_completion_tokens: self.max_completion_tokens.flatten(),
            reasoning_effort: self.reasoning_effort.clone().unwrap_or_default(),
        })
    }
}

#[async_trait]
impl BaseChatModel for ChatOpenAI {
    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &str {
        "openai"
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

        let response = self
            .client
            .post(self.api_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::Api(format!(
                "OpenAI API error ({}): {}",
                status, body
            )));
        }

        let completion: OpenAIResponse = response.json().await?;
        Ok(self.parse_response(completion))
    }

    async fn invoke_stream(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
    ) -> Result<ChatStream, LlmError> {
        let request = self.build_request(messages, tools, tool_choice, true)?;

        let response = self
            .client
            .post(self.api_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::Api(format!(
                "OpenAI API error ({}): {}",
                status, body
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
        let model_lower = self.model.to_lowercase();
        model_lower.contains("gpt-4o")
            || model_lower.contains("gpt-4-turbo")
            || model_lower.contains("gpt-4-vision")
            || model_lower.contains("gpt-4.1")
    }
}

// =============================================================================
// Request/Response Types
// =============================================================================

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize)]
struct OpenAIMessage {
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
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunction,
}

#[derive(Serialize)]
struct OpenAIFunction {
    name: String,
    description: String,
    parameters: serde_json::Map<String, serde_json::Value>,
    strict: bool,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessageResponse,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIMessageResponse {
    content: Option<String>,
    tool_calls: Option<Vec<crate::llm::ToolCall>>,
    reasoning_content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<OpenAIPromptTokenDetails>,
}

#[derive(Deserialize, Default)]
struct OpenAIPromptTokenDetails {
    cached_tokens: u64,
}

impl ChatOpenAI {
    fn build_request(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
        stream: bool,
    ) -> Result<OpenAIRequest, LlmError> {
        let openai_messages: Vec<OpenAIMessage> =
            messages.into_iter().map(Self::convert_message).collect();

        let openai_tools = tools.map(|ts| {
            ts.into_iter()
                .map(|t| OpenAITool {
                    tool_type: "function".to_string(),
                    function: OpenAIFunction {
                        name: t.name,
                        description: t.description,
                        parameters: t.parameters,
                        strict: t.strict,
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

        // For reasoning models, omit temperature
        let temperature = if self.is_reasoning_model() {
            None
        } else {
            Some(self.temperature)
        };

        // For reasoning models, add reasoning_effort
        let reasoning_effort = if self.is_reasoning_model() {
            Some(self.reasoning_effort.clone())
        } else {
            None
        };

        Ok(OpenAIRequest {
            model: self.model.clone(),
            messages: openai_messages,
            tools: openai_tools,
            tool_choice: tool_choice_value,
            temperature,
            max_completion_tokens: self.max_completion_tokens,
            reasoning_effort,
            stream: if stream { Some(true) } else { None },
        })
    }

    fn convert_message(message: Message) -> OpenAIMessage {
        match message {
            Message::User(u) => {
                let content = if u.content.len() == 1 && u.content[0].is_text() {
                    serde_json::json!(u.content[0].as_text().unwrap())
                } else {
                    serde_json::json!(u.content)
                };
                OpenAIMessage {
                    role: "user".to_string(),
                    content: Some(content),
                    name: u.name,
                    tool_calls: None,
                    tool_call_id: None,
                }
            }
            Message::Assistant(a) => OpenAIMessage {
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
            Message::System(s) => OpenAIMessage {
                role: "system".to_string(),
                content: Some(serde_json::json!(s.content)),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message::Developer(d) => OpenAIMessage {
                role: "developer".to_string(),
                content: Some(serde_json::json!(d.content)),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message::Tool(t) => OpenAIMessage {
                role: "tool".to_string(),
                content: Some(serde_json::json!(t.content)),
                name: None,
                tool_calls: None,
                tool_call_id: Some(t.tool_call_id),
            },
        }
    }

    fn parse_response(&self, response: OpenAIResponse) -> ChatCompletion {
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
            .map(|c| {
                let reasoning = c.message.reasoning_content;
                let content = c.message.content.or(reasoning);
                (content, c.message.tool_calls.unwrap_or_default())
            })
            .unwrap_or((None, Vec::new()));

        let usage = response.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            prompt_cached_tokens: u.prompt_tokens_details.map(|d| d.cached_tokens),
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
