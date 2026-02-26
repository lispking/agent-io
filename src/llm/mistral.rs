//! Mistral Chat Model implementation

use async_trait::async_trait;

use crate::llm::openai_compatible::ChatOpenAICompatible;
use crate::llm::{
    BaseChatModel, ChatCompletion, ChatStream, LlmError, Message, ToolChoice, ToolDefinition,
};

const MISTRAL_URL: &str = "https://api.mistral.ai/v1";

/// Mistral Chat Model
///
/// # Example
/// ```ignore
/// use agent_io::llm::ChatMistral;
///
/// let llm = ChatMistral::new("mistral-large-latest")?;
/// ```
pub struct ChatMistral {
    inner: ChatOpenAICompatible,
}

impl ChatMistral {
    /// Create a new Mistral chat model
    pub fn new(model: impl Into<String>) -> Result<Self, LlmError> {
        Self::builder().model(model).build()
    }

    /// Create a builder for configuration
    pub fn builder() -> ChatMistralBuilder {
        ChatMistralBuilder::default()
    }
}

#[derive(Default)]
pub struct ChatMistralBuilder {
    model: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u64>,
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
        let model = self
            .model
            .ok_or_else(|| LlmError::Config("model is required".into()))?;

        let api_key = self
            .api_key
            .or_else(|| std::env::var("MISTRAL_API_KEY").ok())
            .ok_or_else(|| LlmError::Config("MISTRAL_API_KEY not set".into()))?;

        let base_url = self
            .base_url
            .or_else(|| std::env::var("MISTRAL_BASE_URL").ok())
            .unwrap_or_else(|| MISTRAL_URL.to_string());

        let inner = ChatOpenAICompatible::builder()
            .model(&model)
            .base_url(&base_url)
            .provider("mistral")
            .api_key(Some(api_key))
            .temperature(self.temperature.unwrap_or(0.2))
            .max_completion_tokens(self.max_tokens)
            .build()?;

        Ok(ChatMistral { inner })
    }
}

#[async_trait]
impl BaseChatModel for ChatMistral {
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn provider(&self) -> &str {
        "mistral"
    }

    fn context_window(&self) -> Option<u64> {
        let model = self.model().to_lowercase();
        if model.contains("large") || model.contains("codestral") {
            Some(128_000)
        } else {
            Some(32_000)
        }
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
    ) -> Result<ChatCompletion, LlmError> {
        self.inner.invoke(messages, tools, tool_choice).await
    }

    async fn invoke_stream(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
    ) -> Result<ChatStream, LlmError> {
        self.inner.invoke_stream(messages, tools, tool_choice).await
    }

    fn supports_vision(&self) -> bool {
        let model = self.model().to_lowercase();
        model.contains("pixtral")
    }
}
