//! Mistral Chat Model builder

use crate::llm::{
    LlmError,
    openai_compatible::{ChatOpenAICompatible, OpenAICompatibleProviderConfig},
};

use super::ChatMistral;

const MISTRAL_URL: &str = "https://api.mistral.ai/v1";

/// Builder for Mistral chat model
#[derive(Default)]
pub struct ChatMistralBuilder {
    pub(super) model: Option<String>,
    pub(super) api_key: Option<String>,
    pub(super) base_url: Option<String>,
    pub(super) temperature: Option<f32>,
    pub(super) max_tokens: Option<u64>,
}

impl ChatMistralBuilder {
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

    pub fn build(self) -> Result<ChatMistral, LlmError> {
        let inner = ChatOpenAICompatible::build_provider(
            OpenAICompatibleProviderConfig {
                provider: "mistral",
                default_base_url: MISTRAL_URL,
                api_key_env: Some("MISTRAL_API_KEY"),
                base_url_env: Some("MISTRAL_BASE_URL"),
                use_bearer_auth: true,
                default_temperature: 0.2,
            },
            self.model,
            self.api_key,
            self.base_url,
            self.temperature,
            self.max_tokens,
        )?;

        Ok(ChatMistral { inner })
    }
}
