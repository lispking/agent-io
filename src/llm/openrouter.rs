//! OpenRouter Chat Model implementation
//!
//! OpenRouter provides a unified API for many LLM providers.

use async_trait::async_trait;

use crate::llm::openai_compatible::ChatOpenAICompatible;
use crate::llm::{
    BaseChatModel, ChatCompletion, ChatStream, LlmError, Message, ToolChoice, ToolDefinition,
};

const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1";

/// OpenRouter Chat Model
///
/// Access any LLM through OpenRouter's unified API.
///
/// # Example
/// ```ignore
/// use agent_io::llm::ChatOpenRouter;
///
/// let llm = ChatOpenRouter::new("anthropic/claude-3.5-sonnet")?;
/// ```
pub struct ChatOpenRouter {
    inner: ChatOpenAICompatible,
}

impl ChatOpenRouter {
    /// Create a new OpenRouter chat model
    pub fn new(model: impl Into<String>) -> Result<Self, LlmError> {
        Self::builder().model(model).build()
    }

    /// Create a builder for configuration
    pub fn builder() -> ChatOpenRouterBuilder {
        ChatOpenRouterBuilder::default()
    }
}

#[derive(Default)]
pub struct ChatOpenRouterBuilder {
    model: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u64>,
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

#[async_trait]
impl BaseChatModel for ChatOpenRouter {
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn provider(&self) -> &str {
        "openrouter"
    }

    fn context_window(&self) -> Option<u64> {
        self.inner.context_window()
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
        true
    }
}
