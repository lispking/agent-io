//! Google Gemini Chat Model implementation

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

const GOOGLE_API_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Google Gemini Chat Model
#[derive(Builder, Clone)]
#[builder(pattern = "owned", build_fn(skip))]
pub struct ChatGoogle {
    /// Model identifier
    #[builder(setter(into))]
    pub(super) model: String,
    /// API key
    pub(super) api_key: String,
    /// Base URL
    #[builder(setter(into, strip_option), default = "None")]
    pub(super) base_url: Option<String>,
    /// Maximum output tokens
    #[builder(default = "8192")]
    pub(super) max_tokens: u64,
    /// Temperature for sampling
    #[builder(default = "0.2")]
    pub(super) temperature: f32,
    /// Thinking budget (for thinking models)
    #[builder(default = "None")]
    pub(super) thinking_budget: Option<u64>,
    /// HTTP client
    #[builder(setter(skip))]
    pub(super) client: Client,
    /// Context window
    #[builder(setter(skip))]
    pub(super) context_window: u64,
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
