//! Ollama Chat Model implementation
//!
//! Ollama runs models locally and provides an OpenAI-compatible API.

mod builder;

use async_trait::async_trait;

use crate::llm::{
    BaseChatModel, ChatCompletion, ChatStream, LlmError, Message, ToolChoice, ToolDefinition,
};

pub use builder::ChatOllamaBuilder;

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
    pub(super) inner: crate::llm::openai_compatible::ChatOpenAICompatible,
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
