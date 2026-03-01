//! Google Gemini request building

use crate::llm::{ContentPart, LlmError, Message, ToolChoice, ToolDefinition};

use super::types::*;

impl super::ChatGoogle {
    /// Build a Gemini API request
    pub(super) fn build_request(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        _tool_choice: Option<ToolChoice>,
    ) -> Result<GeminiRequest, LlmError> {
        let mut system_instruction: Option<GeminiContent> = None;
        let mut contents: Vec<GeminiContent> = Vec::new();

        for message in messages {
            match message {
                Message::System(s) => {
                    system_instruction = Some(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart::Text { text: s.content }],
                    });
                }
                Message::User(u) => {
                    let parts: Vec<GeminiPart> = u
                        .content
                        .into_iter()
                        .map(|c| match c {
                            ContentPart::Text(t) => GeminiPart::Text { text: t.text },
                            ContentPart::Image(img) => {
                                let (mime_type, data) = if img.image_url.url.starts_with("data:") {
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
                                GeminiPart::InlineData {
                                    inline_data: GeminiInlineData { mime_type, data },
                                }
                            }
                            _ => GeminiPart::Text {
                                text: "[Unsupported content]".to_string(),
                            },
                        })
                        .collect();

                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts,
                    });
                }
                Message::Assistant(a) => {
                    let mut parts = Vec::new();

                    if let Some(t) = a.thinking {
                        parts.push(GeminiPart::Thought { thought: t });
                    }

                    if let Some(c) = a.content {
                        parts.push(GeminiPart::Text { text: c });
                    }

                    for tc in a.tool_calls {
                        let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::json!({}));
                        parts.push(GeminiPart::FunctionCall {
                            function_call: GeminiFunctionCall {
                                name: tc.function.name,
                                args,
                            },
                        });
                    }

                    contents.push(GeminiContent {
                        role: "model".to_string(),
                        parts,
                    });
                }
                Message::Tool(t) => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart::FunctionResponse {
                            function_response: GeminiFunctionResponse {
                                name: "function_result".to_string(),
                                response: GeminiToolResult {
                                    name: "result".to_string(),
                                    content: t.content,
                                },
                            },
                        }],
                    });
                }
                Message::Developer(d) => {
                    system_instruction = Some(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart::Text { text: d.content }],
                    });
                }
            }
        }

        let gemini_tools = tools.map(|ts| GeminiTools {
            function_declarations: ts
                .into_iter()
                .map(|t| GeminiFunctionDeclaration {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                })
                .collect(),
        });

        let thinking_config = if self.is_thinking_model() {
            self.thinking_budget.map(|budget| GeminiThinkingConfig {
                thinking_budget: budget,
            })
        } else {
            None
        };

        Ok(GeminiRequest {
            contents,
            system_instruction,
            tools: gemini_tools,
            generation_config: Some(GeminiGenerationConfig {
                temperature: self.temperature,
                max_output_tokens: self.max_tokens,
                thinking_config,
            }),
        })
    }
}
