//! OpenAI request building

use crate::llm::{LlmError, Message, ToolChoice, ToolDefinition};

use super::types::*;

impl super::ChatOpenAI {
    /// Build an OpenAI API request
    pub(super) fn build_request(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
        stream: bool,
    ) -> Result<OpenAIRequest, LlmError> {
        let openai_messages: Vec<OpenAIMessage> =
            messages.into_iter().map(Self::convert_message).collect();

        let openai_tools = tools.map(|ts| {
            ts.into_iter()
                .map(|t| OpenAITool {
                    tool_type: "function".to_string(),
                    function: OpenAIFunction {
                        name: t.name,
                        description: t.description,
                        parameters: t.parameters,
                        strict: t.strict,
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

        // For reasoning models, omit temperature
        let temperature = if self.is_reasoning_model() {
            None
        } else {
            Some(self.temperature)
        };

        // For reasoning models, add reasoning_effort
        let reasoning_effort = if self.is_reasoning_model() {
            Some(self.reasoning_effort.clone())
        } else {
            None
        };

        Ok(OpenAIRequest {
            model: self.model.clone(),
            messages: openai_messages,
            tools: openai_tools,
            tool_choice: tool_choice_value,
            temperature,
            max_completion_tokens: self.max_completion_tokens,
            reasoning_effort,
            stream: if stream { Some(true) } else { None },
        })
    }

    /// Convert a message to OpenAI format
    pub(super) fn convert_message(message: Message) -> OpenAIMessage {
        match message {
            Message::User(u) => {
                let content = if u.content.len() == 1 && u.content[0].is_text() {
                    serde_json::json!(u.content[0].as_text().unwrap())
                } else {
                    serde_json::json!(u.content)
                };
                OpenAIMessage {
                    role: "user".to_string(),
                    content: Some(content),
                    name: u.name,
                    tool_calls: None,
                    tool_call_id: None,
                }
            }
            Message::Assistant(a) => OpenAIMessage {
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
            Message::System(s) => OpenAIMessage {
                role: "system".to_string(),
                content: Some(serde_json::json!(s.content)),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message::Developer(d) => OpenAIMessage {
                role: "developer".to_string(),
                content: Some(serde_json::json!(d.content)),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message::Tool(t) => OpenAIMessage {
                role: "tool".to_string(),
                content: Some(serde_json::json!(t.content)),
                name: None,
                tool_calls: None,
                tool_call_id: Some(t.tool_call_id),
            },
        }
    }
}
