//! Groq Chat Model implementation
//!
//! Fast inference using Groq's LPU.

mod builder;

use async_trait::async_trait;

use crate::llm::{
    BaseChatModel, ChatCompletion, ChatStream, LlmError, Message, ToolChoice, ToolDefinition,
};

pub use builder::ChatGroqBuilder;

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
    pub(super) inner: crate::llm::openai_compatible::ChatOpenAICompatible,
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
