//! OpenAI Chat Model implementation

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

const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// OpenAI Chat Model
#[derive(Builder, Clone)]
#[builder(pattern = "owned", build_fn(skip))]
pub struct ChatOpenAI {
    /// Model identifier
    #[builder(setter(into))]
    pub(super) model: String,
    /// API key
    pub(super) api_key: String,
    /// Base URL (for proxies)
    #[builder(setter(into, strip_option), default = "None")]
    pub(super) base_url: Option<String>,
    /// Temperature for sampling
    #[builder(default = "0.2")]
    pub(super) temperature: f32,
    /// Maximum completion tokens
    #[builder(default = "Some(4096)")]
    pub(super) max_completion_tokens: Option<u64>,
    /// Reasoning effort for o1+ models
    #[builder(default = "ReasoningEffort::Low")]
    pub(super) reasoning_effort: ReasoningEffort,
    /// HTTP client
    #[builder(setter(skip))]
    pub(super) client: Client,
    /// Context window size
    #[builder(setter(skip))]
    pub(super) context_window: u64,
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

// Re-export ReasoningEffort for public API
pub use types::ReasoningEffort;
