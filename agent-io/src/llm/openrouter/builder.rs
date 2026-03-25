//! OpenRouter Chat Model builder

use crate::llm::{
    LlmError,
    openai_compatible::{ChatOpenAICompatible, OpenAICompatibleProviderConfig},
};

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
        let inner = ChatOpenAICompatible::build_provider(
            OpenAICompatibleProviderConfig {
                provider: "openrouter",
                default_base_url: OPENROUTER_URL,
                api_key_env: Some("OPENROUTER_API_KEY"),
                base_url_env: None,
                use_bearer_auth: true,
                default_temperature: 0.2,
            },
            self.model,
            self.api_key,
            self.base_url,
            self.temperature,
            self.max_tokens,
        )?;

        Ok(ChatOpenRouter { inner })
    }
}
