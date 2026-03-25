//! Google Gemini response parsing

use crate::llm::{ChatCompletion, LlmError, StopReason, ToolCall, Usage};

use super::types::*;

impl super::ChatGoogle {
    /// Parse a Gemini API response
    pub(super) fn parse_response(&self, response: GeminiResponse) -> ChatCompletion {
        let stop_reason = response
            .candidates
            .first()
            .and_then(|c| c.finish_reason.as_ref())
            .and_then(|r| match r.as_str() {
                "STOP" => Some(StopReason::EndTurn),
                "MAX_TOKENS" => Some(StopReason::MaxTokens),
                "TOOL_CODE" => Some(StopReason::ToolUse),
                _ => None,
            });

        let candidate = response.candidates.into_iter().next();

        let (content, thinking, tool_calls) = candidate
            .map(|c| {
                let mut text: Option<String> = None;
                let mut think: Option<String> = None;
                let mut calls = Vec::new();

                for part in c.content.parts {
                    match part {
                        GeminiResponsePart::Text { text: t } => {
                            text = Some(t);
                        }
                        GeminiResponsePart::Thought { thought: t } => {
                            think = Some(t);
                        }
                        GeminiResponsePart::FunctionCall { function_call: fc } => {
                            let id = fc.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                            calls.push(ToolCall::new(
                                id,
                                fc.name,
                                serde_json::to_string(&fc.args).unwrap_or_default(),
                            ));
                        }
                    }
                }

                (text, think, calls)
            })
            .unwrap_or((None, None, Vec::new()));

        let usage = response.usage_metadata.map(|u| Usage {
            prompt_tokens: u.prompt_token_count,
            completion_tokens: u.candidates_token_count,
            total_tokens: u.total_token_count,
            prompt_cached_tokens: Some(u.cached_content_token_count),
            ..Default::default()
        });

        ChatCompletion {
            content,
            thinking,
            redacted_thinking: None,
            tool_calls,
            usage,
            stop_reason,
        }
    }

    /// Parse a streaming chunk
    pub(super) fn parse_stream_chunk(text: &str) -> Option<Result<ChatCompletion, LlmError>> {
        // Google returns JSON array chunks
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Handle array wrapping
            let line = line.trim_start_matches('[').trim_end_matches(']');
            if line.is_empty() {
                continue;
            }

            // Handle comma-separated chunks
            for chunk_str in line.split("},") {
                let chunk_str = if !chunk_str.ends_with('}') {
                    format!("{}{}", chunk_str, "}")
                } else {
                    chunk_str.to_string()
                };

                let chunk: serde_json::Value = match serde_json::from_str(&chunk_str) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let parts = chunk
                    .get("candidates")?
                    .as_array()?
                    .first()?
                    .get("content")?
                    .get("parts")?
                    .as_array()?;

                for part in parts {
                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        return Some(Ok(ChatCompletion::text(text)));
                    }

                    if let Some(thought) = part.get("thought").and_then(|t| t.as_str()) {
                        let mut completion = ChatCompletion::text("");
                        completion.thinking = Some(thought.to_string());
                        return Some(Ok(completion));
                    }

                    if let Some(fc) = part.get("function_call") {
                        let name = fc.get("name")?.as_str()?.to_string();
                        let args = fc.get("args").cloned().unwrap_or(serde_json::json!({}));
                        let id = fc.get("id").and_then(|i| i.as_str()).unwrap_or("pending");

                        return Some(Ok(ChatCompletion {
                            content: None,
                            thinking: None,
                            redacted_thinking: None,
                            tool_calls: vec![ToolCall::new(
                                id,
                                name,
                                serde_json::to_string(&args).unwrap_or_default(),
                            )],
                            usage: None,
                            stop_reason: Some(StopReason::ToolUse),
                        }));
                    }
                }
            }
        }

        None
    }
}
