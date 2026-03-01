//! OpenAI-compatible response parsing

use crate::llm::{ChatCompletion, LlmError, StopReason, ToolCall, Usage};

use super::types::*;

impl super::ChatOpenAICompatible {
    /// Parse a response
    pub(super) fn parse_response(response: OpenAICompatibleResponse) -> ChatCompletion {
        let stop_reason = response
            .choices
            .first()
            .and_then(|c| c.finish_reason.as_ref())
            .and_then(|r| match r.as_str() {
                "stop" => Some(StopReason::EndTurn),
                "tool_calls" => Some(StopReason::ToolUse),
                "length" => Some(StopReason::MaxTokens),
                _ => None,
            });

        let choice = response.choices.into_iter().next();

        let (content, tool_calls) = choice
            .map(|c| (c.message.content, c.message.tool_calls.unwrap_or_default()))
            .unwrap_or((None, Vec::new()));

        let usage = response.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            ..Default::default()
        });

        ChatCompletion {
            content,
            thinking: None,
            redacted_thinking: None,
            tool_calls,
            usage,
            stop_reason,
        }
    }

    /// Parse a streaming chunk
    pub(super) fn parse_stream_chunk(text: &str) -> Option<Result<ChatCompletion, LlmError>> {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || !line.starts_with("data:") {
                continue;
            }

            let data = line.strip_prefix("data:").unwrap().trim();
            if data == "[DONE]" {
                return None;
            }

            let chunk: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let delta = chunk
                .get("choices")
                .and_then(|c| c.as_array())
                .and_then(|a| a.first())
                .and_then(|c| c.get("delta"));

            if let Some(delta) = delta {
                let content = delta
                    .get("content")
                    .and_then(|c| c.as_str())
                    .map(|s| s.to_string());

                let tool_calls: Vec<ToolCall> = delta
                    .get("tool_calls")
                    .and_then(|tc| tc.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|tc| {
                                let id = tc.get("id")?.as_str()?.to_string();
                                let func = tc.get("function")?;
                                let name = func.get("name")?.as_str()?.to_string();
                                let arguments = func.get("arguments")?.as_str()?.to_string();
                                Some(ToolCall::new(id, name, arguments))
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                if content.is_some() || !tool_calls.is_empty() {
                    return Some(Ok(ChatCompletion {
                        content,
                        thinking: None,
                        redacted_thinking: None,
                        tool_calls,
                        usage: None,
                        stop_reason: None,
                    }));
                }
            }
        }

        None
    }
}
