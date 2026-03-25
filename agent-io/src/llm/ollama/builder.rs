//! Ollama Chat Model builder

use crate::llm::{
    LlmError,
    openai_compatible::{ChatOpenAICompatible, OpenAICompatibleProviderConfig},
};

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
        let inner = ChatOpenAICompatible::build_provider(
            OpenAICompatibleProviderConfig {
                provider: "ollama",
                default_base_url: OLLAMA_DEFAULT_URL,
                api_key_env: None,
                base_url_env: Some("OLLAMA_BASE_URL"),
                use_bearer_auth: false,
                default_temperature: 0.2,
            },
            self.model,
            None,
            self.base_url,
            self.temperature,
            self.max_tokens,
        )?;

        Ok(ChatOllama { inner })
    }
}
