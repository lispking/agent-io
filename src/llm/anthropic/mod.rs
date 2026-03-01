//! Anthropic Claude Chat Model implementation

mod request;
mod response;
mod types;

use async_trait::async_trait;
use derive_builder::Builder;
use futures::StreamExt;
use reqwest::Client;
use std::time::Duration;

use crate::llm::{
    BaseChatModel, ChatCompletion, ChatStream, LlmError, Message, ToolChoice, ToolDefinition,
};

use types::*;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";

/// Anthropic Chat Model
#[derive(Builder, Clone)]
#[builder(pattern = "owned", build_fn(skip))]
pub struct ChatAnthropic {
    /// Model identifier
    #[builder(setter(into))]
    pub(super) model: String,
    /// API key
    pub(super) api_key: String,
    /// Base URL (for proxies)
    #[builder(setter(into, strip_option), default = "None")]
    pub(super) base_url: Option<String>,
    /// Maximum output tokens
    #[builder(default = "8192")]
    pub(super) max_tokens: u64,
    /// Temperature for sampling
    #[builder(default = "0.2")]
    pub(super) temperature: f32,
    /// Prompt cache beta header
    #[builder(default = r#"Some("prompt-caching-2024-07-31".to_string())"#)]
    pub(super) prompt_cache_beta: Option<String>,
    /// Enable extended thinking
    #[builder(default = "false")]
    pub(super) thinking: bool,
    /// Thinking budget (tokens)
    #[builder(default = "Some(1024)")]
    pub(super) thinking_budget: Option<u64>,
    /// HTTP client
    #[builder(setter(skip))]
    pub(super) client: Client,
    /// Context window
    #[builder(setter(skip))]
    pub(super) context_window: u64,
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
