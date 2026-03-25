//! OpenRouter Chat Model implementation
//!
//! OpenRouter provides a unified API for many LLM providers.

mod builder;

use async_trait::async_trait;

use crate::llm::{
    BaseChatModel, ChatCompletion, ChatStream, LlmError, Message, ToolChoice, ToolDefinition,
};

pub use builder::ChatOpenRouterBuilder;

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
    pub(super) inner: crate::llm::openai_compatible::ChatOpenAICompatible,
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
