//! DeepSeek Chat Model builder

use crate::llm::{
    LlmError,
    openai_compatible::{ChatOpenAICompatible, OpenAICompatibleProviderConfig},
};

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
        let inner = ChatOpenAICompatible::build_provider(
            OpenAICompatibleProviderConfig {
                provider: "deepseek",
                default_base_url: DEEPSEEK_URL,
                api_key_env: Some("DEEPSEEK_API_KEY"),
                base_url_env: None,
                use_bearer_auth: true,
                default_temperature: 0.2,
            },
            self.model,
            self.api_key,
            None,
            self.temperature,
            self.max_tokens,
        )?;

        Ok(ChatDeepSeek { inner })
    }
}
