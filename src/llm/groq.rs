//! Groq Chat Model implementation

use async_trait::async_trait;

use crate::llm::openai_compatible::ChatOpenAICompatible;
use crate::llm::{
    BaseChatModel, ChatCompletion, ChatStream, LlmError, Message, ToolChoice, ToolDefinition,
};

const GROQ_URL: &str = "https://api.groq.com/openai/v1";

/// Groq Chat Model
///
/// Fast inference using Groq's LPU.
///
/// # Example
/// ```ignore
/// use agent_io::llm::ChatGroq;
///
/// let llm = ChatGroq::new("llama-3.3-70b-versatile")?;
/// ```
pub struct ChatGroq {
    inner: ChatOpenAICompatible,
}

impl ChatGroq {
    /// Create a new Groq chat model
    pub fn new(model: impl Into<String>) -> Result<Self, LlmError> {
        Self::builder().model(model).build()
    }

    /// Create a builder for configuration
    pub fn builder() -> ChatGroqBuilder {
        ChatGroqBuilder::default()
    }
}

#[derive(Default)]
pub struct ChatGroqBuilder {
    model: Option<String>,
    api_key: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u64>,
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
        let model = self
            .model
            .ok_or_else(|| LlmError::Config("model is required".into()))?;

        let api_key = self
            .api_key
            .or_else(|| std::env::var("GROQ_API_KEY").ok())
            .ok_or_else(|| LlmError::Config("GROQ_API_KEY not set".into()))?;

        let inner = ChatOpenAICompatible::builder()
            .model(&model)
            .base_url(GROQ_URL)
            .provider("groq")
            .api_key(Some(api_key))
            .temperature(self.temperature.unwrap_or(0.2))
            .max_completion_tokens(self.max_tokens)
            .build()?;

        Ok(ChatGroq { inner })
    }
}

#[async_trait]
impl BaseChatModel for ChatGroq {
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn provider(&self) -> &str {
        "groq"
    }

    fn context_window(&self) -> Option<u64> {
        Some(128_000)
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
}
