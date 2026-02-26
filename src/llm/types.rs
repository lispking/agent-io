//! Core types for LLM interactions

use serde::{Deserialize, Serialize};

// =============================================================================
// Tool Definition
// =============================================================================

/// Tool choice strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ToolChoice {
    /// Let the model decide whether to call tools
    #[default]
    Auto,
    /// Force the model to call a tool
    Required,
    /// Prevent the model from calling tools
    None,
    /// Force a specific tool to be called
    #[serde(untagged)]
    Named(String),
}

impl From<&str> for ToolChoice {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "auto" => ToolChoice::Auto,
            "required" => ToolChoice::Required,
            "none" => ToolChoice::None,
            name => ToolChoice::Named(name.to_string()),
        }
    }
}

/// JSON Schema for tool parameters
pub type JsonSchema = serde_json::Map<String, serde_json::Value>;

/// Definition of a tool that can be called by the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Name of the tool
    pub name: String,
    /// Description of what the tool does
    pub description: String,
    /// JSON Schema for the tool parameters
    pub parameters: JsonSchema,
    /// Whether to use strict schema validation
    #[serde(default = "default_strict")]
    pub strict: bool,
}

fn default_strict() -> bool {
    true
}

impl ToolDefinition {
    /// Create a new tool definition
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: JsonSchema,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            strict: true,
        }
    }

    /// Set strict mode
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }
}

// =============================================================================
// Function Call
// =============================================================================

/// Function call from the LLM
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Function {
    /// Name of the function to call
    pub name: String,
    /// JSON string of arguments
    pub arguments: String,
}

impl Function {
    /// Parse arguments as a specific type
    pub fn parse_args<T: for<'de> Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.arguments)
    }
}

/// Tool call from the LLM
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Unique identifier for the tool call
    pub id: String,
    /// The function to call
    pub function: Function,
    /// Type of tool (always "function" for now)
    #[serde(default = "default_tool_type")]
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Thought signature for Gemini thinking models
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

fn default_tool_type() -> String {
    "function".to_string()
}

impl ToolCall {
    /// Create a new tool call
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            function: Function {
                name: name.into(),
                arguments: arguments.into(),
            },
            tool_type: "function".to_string(),
            thought_signature: None,
        }
    }

    /// Parse arguments as a specific type
    pub fn parse_args<T: for<'de> Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        self.function.parse_args()
    }
}

// =============================================================================
// Content Parts
// =============================================================================

/// Text content part
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContentPartText {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

impl ContentPartText {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            content_type: "text".to_string(),
            text: text.into(),
        }
    }
}

impl From<String> for ContentPartText {
    fn from(text: String) -> Self {
        Self::new(text)
    }
}

impl From<&str> for ContentPartText {
    fn from(text: &str) -> Self {
        Self::new(text)
    }
}

/// Image content part
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPartImage {
    #[serde(rename = "type")]
    pub content_type: String,
    pub image_url: ImageUrl,
}

/// Image URL structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl ContentPartImage {
    /// Create from URL
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            content_type: "image_url".to_string(),
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

    /// Create from base64 data
    pub fn from_base64(media_type: &str, data: &str) -> Self {
        Self {
            content_type: "image_url".to_string(),
            image_url: ImageUrl {
                url: format!("data:{};base64,{}", media_type, data),
                detail: None,
            },
        }
    }
}

/// Document content part (for PDFs etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPartDocument {
    #[serde(rename = "type")]
    pub content_type: String,
    pub source: DocumentSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

impl ContentPartDocument {
    pub fn from_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            content_type: "document".to_string(),
            source: DocumentSource {
                source_type: "base64".to_string(),
                media_type: media_type.into(),
                data: data.into(),
            },
        }
    }
}

/// Thinking content part
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPartThinking {
    #[serde(rename = "type")]
    pub content_type: String,
    pub thinking: String,
}

impl ContentPartThinking {
    pub fn new(thinking: impl Into<String>) -> Self {
        Self {
            content_type: "thinking".to_string(),
            thinking: thinking.into(),
        }
    }
}

/// Redacted thinking content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPartRedactedThinking {
    #[serde(rename = "type")]
    pub content_type: String,
    pub data: String,
}

/// Refusal content part
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPartRefusal {
    #[serde(rename = "type")]
    pub content_type: String,
    pub refusal: String,
}

/// Union type for all content parts
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContentPart {
    Text(ContentPartText),
    Image(ContentPartImage),
    Document(ContentPartDocument),
    Thinking(ContentPartThinking),
    RedactedThinking(ContentPartRedactedThinking),
    Refusal(ContentPartRefusal),
}

impl ContentPart {
    pub fn text(content: impl Into<String>) -> Self {
        ContentPart::Text(ContentPartText::new(content))
    }

    pub fn is_text(&self) -> bool {
        matches!(self, ContentPart::Text(_))
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentPart::Text(t) => Some(&t.text),
            _ => None,
        }
    }
}

impl From<String> for ContentPart {
    fn from(text: String) -> Self {
        ContentPart::text(text)
    }
}

impl From<&str> for ContentPart {
    fn from(text: &str) -> Self {
        ContentPart::text(text)
    }
}

// =============================================================================
// Messages
// =============================================================================

/// User message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessage {
    pub role: String,
    pub content: Vec<ContentPart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl UserMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: vec![ContentPart::text(content)],
            name: None,
        }
    }

    pub fn with_parts(content: Vec<ContentPart>) -> Self {
        Self {
            role: "user".to_string(),
            content,
            name: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// System message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMessage {
    pub role: String,
    pub content: String,
}

impl SystemMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }
}

/// Developer message (for o1+ models)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeveloperMessage {
    pub role: String,
    pub content: String,
}

impl DeveloperMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            role: "developer".to_string(),
            content: content.into(),
        }
    }
}

/// Assistant message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted_thinking: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl AssistantMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content.into()),
            thinking: None,
            redacted_thinking: None,
            tool_calls: Vec::new(),
            refusal: None,
        }
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    pub fn with_thinking(mut self, thinking: impl Into<String>) -> Self {
        self.thinking = Some(thinking.into());
        self
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_none() && self.thinking.is_none() && self.tool_calls.is_empty()
    }
}

/// Tool result message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMessage {
    pub role: String,
    pub content: String,
    pub tool_call_id: String,
    /// Tool name that produced this result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Whether this is an ephemeral message
    #[serde(default)]
    pub ephemeral: bool,
    /// Whether this ephemeral message has been destroyed
    #[serde(default)]
    pub destroyed: bool,
}

impl ToolMessage {
    pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: content.into(),
            tool_call_id: tool_call_id.into(),
            tool_name: None,
            ephemeral: false,
            destroyed: false,
        }
    }

    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = Some(name.into());
        self
    }

    pub fn with_ephemeral(mut self, ephemeral: bool) -> Self {
        self.ephemeral = ephemeral;
        self
    }

    pub fn destroy(&mut self) {
        self.destroyed = true;
        self.content = "<removed to save context>".to_string();
    }
}

/// Union type for all messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
    System(SystemMessage),
    Developer(DeveloperMessage),
    Tool(ToolMessage),
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Message::User(UserMessage::new(content))
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Message::Assistant(AssistantMessage::new(content))
    }

    pub fn system(content: impl Into<String>) -> Self {
        Message::System(SystemMessage::new(content))
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Message::Tool(ToolMessage::new(tool_call_id, content))
    }

    pub fn role(&self) -> &str {
        match self {
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
            Message::System(_) => "system",
            Message::Developer(_) => "developer",
            Message::Tool(_) => "tool",
        }
    }
}

impl From<UserMessage> for Message {
    fn from(msg: UserMessage) -> Self {
        Message::User(msg)
    }
}

impl From<AssistantMessage> for Message {
    fn from(msg: AssistantMessage) -> Self {
        Message::Assistant(msg)
    }
}

impl From<SystemMessage> for Message {
    fn from(msg: SystemMessage) -> Self {
        Message::System(msg)
    }
}

impl From<ToolMessage> for Message {
    fn from(msg: ToolMessage) -> Self {
        Message::Tool(msg)
    }
}

// =============================================================================
// Response Types
// =============================================================================

/// Token usage information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cached_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_creation_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_image_tokens: Option<u64>,
}

impl Usage {
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            ..Default::default()
        }
    }
}

/// Stop reason for completion
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    StopSequence,
    ToolUse,
    MaxTokens,
    #[serde(other)]
    Unknown,
}

/// Chat completion response
#[derive(Debug, Clone)]
pub struct ChatCompletion {
    /// Text content (if any)
    pub content: Option<String>,
    /// Thinking content (for extended thinking models)
    pub thinking: Option<String>,
    /// Redacted thinking content
    pub redacted_thinking: Option<String>,
    /// Tool calls (if any)
    pub tool_calls: Vec<ToolCall>,
    /// Token usage
    pub usage: Option<Usage>,
    /// Why the completion stopped
    pub stop_reason: Option<StopReason>,
}

impl ChatCompletion {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            thinking: None,
            redacted_thinking: None,
            tool_calls: Vec::new(),
            usage: None,
            stop_reason: None,
        }
    }

    pub fn with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            content: None,
            thinking: None,
            redacted_thinking: None,
            tool_calls,
            usage: None,
            stop_reason: Some(StopReason::ToolUse),
        }
    }

    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    pub fn has_content(&self) -> bool {
        self.content.is_some() && self.content.as_ref().is_some_and(|c| !c.is_empty())
    }
}

// =============================================================================
// Cache Control
// =============================================================================

/// Cache control for prompt caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub control_type: CacheControlType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheControlType {
    Ephemeral,
}

impl CacheControl {
    pub fn ephemeral() -> Self {
        Self {
            control_type: CacheControlType::Ephemeral,
        }
    }
}
