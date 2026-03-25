//! Mistral Chat Model implementation

mod builder;

use async_trait::async_trait;

use crate::llm::{
    BaseChatModel, ChatCompletion, ChatStream, LlmError, Message, ToolChoice, ToolDefinition,
};

pub use builder::ChatMistralBuilder;

/// Mistral Chat Model
///
/// # Example
/// ```ignore
/// use agent_io::llm::ChatMistral;
///
/// let llm = ChatMistral::new("mistral-large-latest")?;
/// ```
pub struct ChatMistral {
    pub(super) inner: crate::llm::openai_compatible::ChatOpenAICompatible,
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
