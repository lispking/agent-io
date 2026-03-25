//! Base trait for LLM implementations

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

use super::{ChatCompletion, LlmError, Message, ToolChoice, ToolDefinition};

/// Type alias for boxed stream
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatCompletion, LlmError>> + Send>>;

/// Base trait for chat model implementations
#[async_trait]
pub trait BaseChatModel: Send + Sync {
    /// Get the model name
    fn model(&self) -> &str;

    /// Get the provider name
    fn provider(&self) -> &str;

    /// Get the context window size (max input tokens)
    fn context_window(&self) -> Option<u64> {
        None
    }

    /// Invoke the model with messages
    async fn invoke(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
    ) -> Result<ChatCompletion, LlmError>;

    /// Invoke the model with streaming response
    async fn invoke_stream(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
    ) -> Result<ChatStream, LlmError>;

    /// Count tokens in messages (approximate)
    async fn count_tokens(&self, messages: &[Message]) -> u64 {
        // Default approximation: ~4 chars per token
        let total_chars: usize = messages
            .iter()
            .map(|m| match m {
                Message::User(u) => u
                    .content
                    .iter()
                    .map(|c| c.as_text().map(|t| t.len()).unwrap_or(10))
                    .sum(),
                Message::Assistant(a) => {
                    a.content.as_ref().map(|c| c.len()).unwrap_or(0)
                        + a.tool_calls
                            .iter()
                            .map(|tc| tc.function.arguments.len())
                            .sum::<usize>()
                }
                Message::System(s) => s.content.len(),
                Message::Developer(d) => d.content.len(),
                Message::Tool(t) => t.content.len(),
            })
            .sum();
        (total_chars / 4) as u64
    }

    /// Check if the model supports tools
    fn supports_tools(&self) -> bool {
        true
    }

    /// Check if the model supports streaming
    fn supports_streaming(&self) -> bool {
        true
    }

    /// Check if the model supports vision
    fn supports_vision(&self) -> bool {
        false
    }
}

/// Builder pattern helpers for common model configurations
pub struct ModelBuilder;

impl ModelBuilder {
    /// Create an OpenAI model
    #[cfg(feature = "openai")]
    pub fn openai(model: impl Into<String>) -> super::openai::ChatOpenAIBuilder {
        super::ChatOpenAI::builder().model(model)
    }

    /// Create an Anthropic model
    #[cfg(feature = "anthropic")]
    pub fn anthropic(model: impl Into<String>) -> super::anthropic::ChatAnthropicBuilder {
        super::ChatAnthropic::builder().model(model)
    }

    /// Create a Google model
    #[cfg(feature = "google")]
    pub fn google(model: impl Into<String>) -> super::google::ChatGoogleBuilder {
        super::ChatGoogle::builder().model(model)
    }
}
