//! OpenAI-compatible API base implementation
//!
//! This module provides a base implementation for any LLM provider that uses
//! the OpenAI-compatible API format (Ollama, OpenRouter, DeepSeek, Groq, etc.)

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

/// OpenAI-compatible Chat Model base implementation
#[derive(Builder, Clone)]
#[builder(pattern = "owned", build_fn(skip))]
pub struct ChatOpenAICompatible {
    /// Model identifier
    #[builder(setter(into))]
    pub(super) model: String,
    /// API key (optional for some providers like Ollama)
    #[builder(setter(into), default = "None")]
    pub(super) api_key: Option<String>,
    /// Base URL for the API
    #[builder(setter(into))]
    pub(super) base_url: String,
    /// Provider name for identification
    #[builder(setter(into))]
    pub(super) provider: String,
    /// Temperature for sampling
    #[builder(default = "0.2")]
    pub(super) temperature: f32,
    /// Maximum completion tokens
    #[builder(default = "Some(4096)")]
    pub(super) max_completion_tokens: Option<u64>,
    /// HTTP client
    #[builder(setter(skip))]
    pub(super) client: Client,
    /// Context window size
    #[builder(setter(skip))]
    pub(super) context_window: u64,
    /// Whether to include Bearer prefix in auth header
    #[builder(default = "true")]
    pub(super) use_bearer_auth: bool,
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
