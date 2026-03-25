//! Groq Chat Model builder

use crate::llm::{
    LlmError,
    openai_compatible::{ChatOpenAICompatible, OpenAICompatibleProviderConfig},
};

use super::ChatGroq;

const GROQ_URL: &str = "https://api.groq.com/openai/v1";

/// Builder for Groq chat model
#[derive(Default)]
pub struct ChatGroqBuilder {
    pub(super) model: Option<String>,
    pub(super) api_key: Option<String>,
    pub(super) temperature: Option<f32>,
    pub(super) max_tokens: Option<u64>,
}

impl ChatGroqBuilder {
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

    pub fn build(self) -> Result<ChatGroq, LlmError> {
        let inner = ChatOpenAICompatible::build_provider(
            OpenAICompatibleProviderConfig {
                provider: "groq",
                default_base_url: GROQ_URL,
                api_key_env: Some("GROQ_API_KEY"),
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

        Ok(ChatGroq { inner })
    }
}
