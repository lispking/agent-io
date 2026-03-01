//! OpenRouter Chat Model builder

use crate::llm::{LlmError, openai_compatible::ChatOpenAICompatible};

use super::ChatOpenRouter;

const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1";

/// Builder for OpenRouter chat model
#[derive(Default)]
pub struct ChatOpenRouterBuilder {
    pub(super) model: Option<String>,
    pub(super) api_key: Option<String>,
    pub(super) base_url: Option<String>,
    pub(super) temperature: Option<f32>,
    pub(super) max_tokens: Option<u64>,
}

impl ChatOpenRouterBuilder {
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn max_tokens(mut self, tokens: u64) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    pub fn build(self) -> Result<ChatOpenRouter, LlmError> {
        let model = self
            .model
            .ok_or_else(|| LlmError::Config("model is required".into()))?;

        let api_key = self
            .api_key
            .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
            .ok_or_else(|| LlmError::Config("OPENROUTER_API_KEY not set".into()))?;

        let base_url = self.base_url.unwrap_or_else(|| OPENROUTER_URL.to_string());

        let inner = ChatOpenAICompatible::builder()
            .model(&model)
            .base_url(&base_url)
            .provider("openrouter")
            .api_key(Some(api_key))
            .temperature(self.temperature.unwrap_or(0.2))
            .max_completion_tokens(self.max_tokens)
            .build()?;

        Ok(ChatOpenRouter { inner })
    }
}
