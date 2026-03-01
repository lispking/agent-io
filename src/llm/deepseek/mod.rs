//! DeepSeek Chat Model implementation

mod builder;

use async_trait::async_trait;

use crate::llm::{
    BaseChatModel, ChatCompletion, ChatStream, LlmError, Message, ToolChoice, ToolDefinition,
};

pub use builder::ChatDeepSeekBuilder;

/// DeepSeek Chat Model
///
/// # Example
/// ```ignore
/// use agent_io::llm::ChatDeepSeek;
///
/// let llm = ChatDeepSeek::new("deepseek-chat")?;
/// ```
pub struct ChatDeepSeek {
    pub(super) inner: crate::llm::openai_compatible::ChatOpenAICompatible,
}

impl ChatDeepSeek {
    /// Create a new DeepSeek chat model
    pub fn new(model: impl Into<String>) -> Result<Self, LlmError> {
        Self::builder().model(model).build()
    }

    /// Create a builder for configuration
    pub fn builder() -> ChatDeepSeekBuilder {
        ChatDeepSeekBuilder::default()
    }
}

#[async_trait]
impl BaseChatModel for ChatDeepSeek {
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn provider(&self) -> &str {
        "deepseek"
    }

    fn context_window(&self) -> Option<u64> {
        Some(64_000)
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
