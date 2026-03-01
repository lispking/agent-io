//! Ollama Chat Model builder

use crate::llm::{LlmError, openai_compatible::ChatOpenAICompatible};

use super::ChatOllama;

const OLLAMA_DEFAULT_URL: &str = "http://localhost:11434/v1";

/// Builder for Ollama chat model
#[derive(Default)]
pub struct ChatOllamaBuilder {
    pub(super) model: Option<String>,
    pub(super) base_url: Option<String>,
    pub(super) temperature: Option<f32>,
    pub(super) max_tokens: Option<u64>,
}

impl ChatOllamaBuilder {
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
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

    pub fn build(self) -> Result<ChatOllama, LlmError> {
        let model = self
            .model
            .ok_or_else(|| LlmError::Config("model is required".into()))?;

        let base_url = self.base_url.unwrap_or_else(|| {
            std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| OLLAMA_DEFAULT_URL.to_string())
        });

        let inner = ChatOpenAICompatible::builder()
            .model(&model)
            .base_url(&base_url)
            .provider("ollama")
            .api_key(None)
            .use_bearer_auth(false)
            .temperature(self.temperature.unwrap_or(0.2))
            .max_completion_tokens(self.max_tokens)
            .build()?;

        Ok(ChatOllama { inner })
    }
}
