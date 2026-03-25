//! DeepSeek Chat Model builder

use crate::llm::{LlmError, openai_compatible::ChatOpenAICompatible};

use super::ChatDeepSeek;

const DEEPSEEK_URL: &str = "https://api.deepseek.com/v1";

/// Builder for DeepSeek chat model
#[derive(Default)]
pub struct ChatDeepSeekBuilder {
    pub(super) model: Option<String>,
    pub(super) api_key: Option<String>,
    pub(super) temperature: Option<f32>,
    pub(super) max_tokens: Option<u64>,
}

impl ChatDeepSeekBuilder {
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
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

    pub fn build(self) -> Result<ChatDeepSeek, LlmError> {
        let model = self
            .model
            .ok_or_else(|| LlmError::Config("model is required".into()))?;

        let api_key = self
            .api_key
            .or_else(|| std::env::var("DEEPSEEK_API_KEY").ok())
            .ok_or_else(|| LlmError::Config("DEEPSEEK_API_KEY not set".into()))?;

        let inner = ChatOpenAICompatible::builder()
            .model(&model)
            .base_url(DEEPSEEK_URL)
            .provider("deepseek")
            .api_key(Some(api_key))
            .temperature(self.temperature.unwrap_or(0.2))
            .max_completion_tokens(self.max_tokens)
            .build()?;

        Ok(ChatDeepSeek { inner })
    }
}
