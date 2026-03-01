//! Anthropic response parsing

use crate::llm::{ChatCompletion, LlmError, StopReason, ToolCall, Usage};

use super::types::*;

impl super::ChatAnthropic {
    /// Parse an Anthropic API response
    pub(super) fn parse_response(&self, response: AnthropicResponse) -> ChatCompletion {
        let mut content: Option<String> = None;
        let mut thinking: Option<String> = None;
        let mut redacted_thinking: Option<String> = None;
        let mut tool_calls = Vec::new();

        for c in response.content {
            match c {
                AnthropicResponseContent::Text { text } => {
                    content = Some(text);
                }
                AnthropicResponseContent::Thinking { thinking: t } => {
                    thinking = Some(t);
                }
                AnthropicResponseContent::RedactedThinking { data } => {
                    redacted_thinking = Some(data);
                }
                AnthropicResponseContent::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall::new(
                        id,
                        name,
                        serde_json::to_string(&input).unwrap_or_default(),
                    ));
                }
            }
        }

        let stop_reason = response.stop_reason.and_then(|r| match r.as_str() {
            "end_turn" => Some(StopReason::EndTurn),
            "tool_use" => Some(StopReason::ToolUse),
            "max_tokens" => Some(StopReason::MaxTokens),
            _ => None,
        });

        let usage = Usage {
            prompt_tokens: response.usage.input_tokens,
            completion_tokens: response.usage.output_tokens,
            total_tokens: response.usage.input_tokens + response.usage.output_tokens,
            prompt_cached_tokens: Some(response.usage.cache_read_input_tokens),
            prompt_cache_creation_tokens: Some(response.usage.cache_creation_input_tokens),
            ..Default::default()
        };

        ChatCompletion {
            content,
            thinking,
            redacted_thinking,
            tool_calls,
            usage: Some(usage),
            stop_reason,
        }
    }

    /// Parse an SSE event from the stream
    pub(super) fn parse_sse_event(text: &str) -> Option<Result<ChatCompletion, LlmError>> {
        for line in text.lines() {
            let line = line.trim();

            if !line.starts_with("data:") && !line.starts_with("data: ") {
                continue;
            }

            let data = line.strip_prefix("data:").unwrap().trim();

            let event: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let event_type = event.get("type")?.as_str()?;

            match event_type {
                "content_block_delta" => {
                    let delta = event.get("delta")?;
                    let delta_type = delta.get("type")?.as_str()?;

                    match delta_type {
                        "text_delta" => {
                            let text = delta.get("text")?.as_str()?;
                            return Some(Ok(ChatCompletion::text(text)));
                        }
                        "thinking_delta" => {
                            let thinking = delta.get("thinking")?.as_str()?;
                            let mut completion = ChatCompletion::text("");
                            completion.thinking = Some(thinking.to_string());
                            return Some(Ok(completion));
                        }
                        "input_json_delta" => {
                            let partial = delta.get("partial_json")?.as_str()?;
                            let index = event.get("index")?.as_u64()? as usize;
                            return Some(Ok(ChatCompletion {
                                content: None,
                                thinking: None,
                                redacted_thinking: None,
                                tool_calls: vec![ToolCall::new(
                                    format!("pending_{}", index),
                                    "pending",
                                    partial.to_string(),
                                )],
                                usage: None,
                                stop_reason: None,
                            }));
                        }
                        _ => {}
                    }
                }
                "message_start" | "message_stop" | "content_block_start" | "content_block_stop" => {
                    return None;
                }
                _ => {}
            }
        }

        None
    }
}
