//! Message types for LLM conversations

use serde::{Deserialize, Serialize};

use super::content::ContentPart;
use super::tool::ToolCall;

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
