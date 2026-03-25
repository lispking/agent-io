//! Anthropic request building

use crate::llm::{ContentPart, LlmError, Message, ToolChoice, ToolDefinition};

use super::types::*;

impl super::ChatAnthropic {
    /// Build an Anthropic API request
    pub(super) fn build_request(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
        stream: bool,
    ) -> Result<AnthropicRequest, LlmError> {
        // Separate system message from other messages
        let mut system: Option<String> = None;
        let mut anthropic_messages: Vec<AnthropicMessage> = Vec::new();

        for message in messages {
            match message {
                Message::System(s) => {
                    system = Some(s.content);
                }
                Message::User(u) => {
                    let content: Vec<AnthropicContent> = u
                        .content
                        .into_iter()
                        .map(|c| match c {
                            ContentPart::Text(t) => AnthropicContent::Text {
                                text: t.text,
                                content_type: "text".to_string(),
                            },
                            ContentPart::Image(img) => {
                                let (media_type, data) = if img.image_url.url.starts_with("data:") {
                                    // Parse data URL
                                    let parts: Vec<&str> =
                                        img.image_url.url.splitn(2, ',').collect();
                                    let mime = parts[0]
                                        .strip_prefix("data:")
                                        .and_then(|s| s.strip_suffix(";base64"))
                                        .unwrap_or("image/png");
                                    (mime.to_string(), parts.get(1).unwrap_or(&"").to_string())
                                } else {
                                    ("image/png".to_string(), img.image_url.url.clone())
                                };
                                AnthropicContent::Image {
                                    source: AnthropicImageSource {
                                        source_type: "base64".to_string(),
                                        media_type,
                                        data,
                                    },
                                    content_type: "image".to_string(),
                                }
                            }
                            _ => AnthropicContent::Text {
                                text: "[Unsupported content type]".to_string(),
                                content_type: "text".to_string(),
                            },
                        })
                        .collect();

                    anthropic_messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content,
                    });
                }
                Message::Assistant(a) => {
                    let mut content = Vec::new();

                    if let Some(t) = a.thinking {
                        content.push(AnthropicContent::Text {
                            text: t,
                            content_type: "thinking".to_string(),
                        });
                    }

                    if let Some(c) = a.content {
                        content.push(AnthropicContent::Text {
                            text: c,
                            content_type: "text".to_string(),
                        });
                    }

                    for tc in a.tool_calls {
                        let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::json!({}));
                        content.push(AnthropicContent::ToolUse {
                            id: tc.id,
                            name: tc.function.name,
                            input,
                            content_type: "tool_use".to_string(),
                        });
                    }

                    anthropic_messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content,
                    });
                }
                Message::Tool(t) => {
                    anthropic_messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: vec![AnthropicContent::ToolResult {
                            tool_use_id: t.tool_call_id,
                            content: t.content,
                            content_type: "tool_result".to_string(),
                        }],
                    });
                }
                Message::Developer(d) => {
                    system = Some(d.content);
                }
            }
        }

        let anthropic_tools = tools.map(|ts| {
            ts.into_iter()
                .map(|t| AnthropicTool {
                    name: t.name,
                    description: t.description,
                    input_schema: t.parameters,
                })
                .collect()
        });

        let tool_choice_value = tool_choice.map(|tc| match tc {
            ToolChoice::Auto => serde_json::json!({"type": "auto"}),
            ToolChoice::Required => serde_json::json!({"type": "any"}),
            ToolChoice::None => serde_json::json!({"type": "none"}),
            ToolChoice::Named(name) => serde_json::json!({"type": "tool", "name": name}),
        });

        let thinking_config = if self.thinking && self.supports_thinking() {
            Some(ThinkingConfig {
                thinking_type: "enabled".to_string(),
                budget_tokens: self.thinking_budget,
            })
        } else {
            None
        };

        Ok(AnthropicRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            messages: anthropic_messages,
            system,
            tools: anthropic_tools,
            tool_choice: tool_choice_value,
            temperature: Some(self.temperature),
            stream: if stream { Some(true) } else { None },
            thinking: thinking_config,
        })
    }
}
