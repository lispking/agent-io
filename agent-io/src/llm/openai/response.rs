//! OpenAI response parsing

use crate::llm::{ChatCompletion, LlmError, StopReason, ToolCall, Usage};

use super::types::*;

impl super::ChatOpenAI {
    /// Parse an OpenAI API response
    pub(super) fn parse_response(&self, response: OpenAIResponse) -> ChatCompletion {
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
            .map(|c| {
                let reasoning = c.message.reasoning_content;
                let content = c.message.content.or(reasoning);
                (content, c.message.tool_calls.unwrap_or_default())
            })
            .unwrap_or((None, Vec::new()));

        let usage = response.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            prompt_cached_tokens: u.prompt_tokens_details.map(|d| d.cached_tokens),
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

    /// Parse SSE stream body (from proxies that ignore stream=false) into a single ChatCompletion.
    pub(super) fn parse_sse_as_completion(&self, body: &str) -> Result<ChatCompletion, LlmError> {
        let mut content = String::new();
        let mut tool_calls: Vec<crate::llm::ToolCall> = Vec::new();
        let mut usage: Option<crate::llm::Usage> = None;
        let mut finish_reason: Option<String> = None;

        for line in body.lines() {
            let line = line.trim();
            let Some(data) = line.strip_prefix("data:").map(str::trim) else {
                continue;
            };
            if data == "[DONE]" {
                break;
            }
            let Ok(val) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };

            let choice = val.get("choices").and_then(|c| c.get(0));

            if let Some(r) = choice
                .and_then(|c| c.get("finish_reason"))
                .and_then(|r| r.as_str())
                && r != "null"
            {
                finish_reason = Some(r.to_string());
            }

            if let Some(delta) = choice.and_then(|c| c.get("delta")) {
                if let Some(s) = delta.get("content").and_then(|c| c.as_str()) {
                    content.push_str(s);
                }
                if let Some(tc_arr) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
                    for tc in tc_arr {
                        let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                        while tool_calls.len() <= index {
                            tool_calls.push(crate::llm::ToolCall::new(
                                String::new(),
                                String::new(),
                                String::new(),
                            ));
                        }
                        if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                            tool_calls[index].id = id.to_string();
                        }
                        if let Some(f) = tc.get("function") {
                            if let Some(n) = f.get("name").and_then(|n| n.as_str()) {
                                tool_calls[index].function.name.push_str(n);
                            }
                            if let Some(a) = f.get("arguments").and_then(|a| a.as_str()) {
                                tool_calls[index].function.arguments.push_str(a);
                            }
                        }
                    }
                }
            }

            if let Some(u) = val.get("usage") {
                usage = Some(crate::llm::Usage {
                    prompt_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                    completion_tokens: u
                        .get("completion_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                    total_tokens: u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                    prompt_cached_tokens: u
                        .get("prompt_tokens_details")
                        .and_then(|d| d.get("cached_tokens"))
                        .and_then(|v| v.as_u64()),
                    ..Default::default()
                });
            }
        }

        let stop_reason = finish_reason.as_deref().and_then(|r| match r {
            "stop" => Some(crate::llm::StopReason::EndTurn),
            "tool_calls" => Some(crate::llm::StopReason::ToolUse),
            "length" => Some(crate::llm::StopReason::MaxTokens),
            _ => None,
        });
        let final_content = if content.is_empty() {
            None
        } else {
            Some(content)
        };
        let final_tool_calls: Vec<_> = tool_calls
            .into_iter()
            .filter(|tc| !tc.function.name.is_empty())
            .collect();

        // Empty response from proxy — treat as transient, trigger retry.
        if final_content.is_none() && final_tool_calls.is_empty() && finish_reason.is_none() {
            return Err(LlmError::RateLimit);
        }

        Ok(ChatCompletion {
            content: final_content,
            thinking: None,
            redacted_thinking: None,
            tool_calls: final_tool_calls,
            usage,
            stop_reason,
        })
    }
}
