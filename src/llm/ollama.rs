//! Ollama Chat Model implementation
//!
//! Ollama runs models locally and provides an OpenAI-compatible API.

use async_trait::async_trait;

use crate::llm::openai_compatible::ChatOpenAICompatible;
use crate::llm::{
    BaseChatModel, ChatCompletion, ChatStream, LlmError, Message, ToolChoice, ToolDefinition,
};

const OLLAMA_DEFAULT_URL: &str = "http://localhost:11434/v1";

/// Ollama Chat Model
///
/// Connect to a locally running Ollama instance.
///
/// # Example
/// ```ignore
/// use agent_io::llm::ChatOllama;
///
/// let llm = ChatOllama::new("llama3.2")?;
/// ```
pub struct ChatOllama {
    inner: ChatOpenAICompatible,
}

impl ChatOllama {
    /// Create a new Ollama chat model
    pub fn new(model: impl Into<String>) -> Result<Self, LlmError> {
        Self::builder().model(model).build()
    }

    /// Create a builder for configuration
    pub fn builder() -> ChatOllamaBuilder {
        ChatOllamaBuilder::default()
    }
}

#[derive(Default)]
pub struct ChatOllamaBuilder {
    model: Option<String>,
    base_url: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u64>,
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

#[async_trait]
impl BaseChatModel for ChatOllama {
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn provider(&self) -> &str {
        "ollama"
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
        let model = self.model().to_lowercase();
        model.contains("llava") || model.contains("vision")
    }
}
