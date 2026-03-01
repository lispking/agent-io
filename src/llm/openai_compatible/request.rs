//! OpenAI-compatible request building

use crate::llm::{LlmError, Message, ToolChoice, ToolDefinition};

use super::types::*;

impl super::ChatOpenAICompatible {
    /// Build a request
    pub(super) fn build_request(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
        stream: bool,
    ) -> Result<OpenAICompatibleRequest, LlmError> {
        let openai_messages: Vec<OpenAICompatibleMessage> =
            messages.into_iter().map(Self::convert_message).collect();

        let openai_tools = tools.map(|ts| {
            ts.into_iter()
                .map(|t| OpenAICompatibleTool {
                    tool_type: "function".to_string(),
                    function: OpenAICompatibleFunction {
                        name: t.name,
                        description: t.description,
                        parameters: t.parameters,
                    },
                })
                .collect()
        });

        let tool_choice_value = tool_choice.map(|tc| match tc {
            ToolChoice::Auto => serde_json::json!("auto"),
            ToolChoice::Required => serde_json::json!("required"),
            ToolChoice::None => serde_json::json!("none"),
            ToolChoice::Named(name) => {
                serde_json::json!({"type": "function", "function": {"name": name}})
            }
        });

        Ok(OpenAICompatibleRequest {
            model: self.model.clone(),
            messages: openai_messages,
            tools: openai_tools,
            tool_choice: tool_choice_value,
            temperature: Some(self.temperature),
            max_tokens: self.max_completion_tokens,
            stream: if stream { Some(true) } else { None },
        })
    }

    /// Convert a message to OpenAI-compatible format
    pub(super) fn convert_message(message: Message) -> OpenAICompatibleMessage {
        match message {
            Message::User(u) => {
                let content = if u.content.len() == 1 && u.content[0].is_text() {
                    serde_json::json!(u.content[0].as_text().unwrap())
                } else {
                    serde_json::json!(u.content)
                };
                OpenAICompatibleMessage {
                    role: "user".to_string(),
                    content: Some(content),
                    name: u.name,
                    tool_calls: None,
                    tool_call_id: None,
                }
            }
            Message::Assistant(a) => OpenAICompatibleMessage {
                role: "assistant".to_string(),
                content: a.content.map(|c| serde_json::json!(c)),
                name: None,
                tool_calls: if a.tool_calls.is_empty() {
                    None
                } else {
                    Some(a.tool_calls)
                },
                tool_call_id: None,
            },
            Message::System(s) => OpenAICompatibleMessage {
                role: "system".to_string(),
                content: Some(serde_json::json!(s.content)),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message::Developer(d) => OpenAICompatibleMessage {
                role: "developer".to_string(),
                content: Some(serde_json::json!(d.content)),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message::Tool(t) => OpenAICompatibleMessage {
                role: "tool".to_string(),
                content: Some(serde_json::json!(t.content)),
                name: None,
                tool_calls: None,
                tool_call_id: Some(t.tool_call_id),
            },
        }
    }
}
