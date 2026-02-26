//! Agent events for streaming responses

use serde::{Deserialize, Serialize};

use crate::llm::Usage;

/// Event emitted during agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Text content from the LLM
    Text(TextEvent),
    /// Thinking/reasoning content
    Thinking(ThinkingEvent),
    /// Tool is being called
    ToolCall(ToolCallEvent),
    /// Tool execution result
    ToolResult(ToolResultEvent),
    /// Final response ready
    FinalResponse(FinalResponseEvent),
    /// Message started
    MessageStart(MessageStartEvent),
    /// Message completed
    MessageComplete(MessageCompleteEvent),
    /// Step started
    StepStart(StepStartEvent),
    /// Step completed
    StepComplete(StepCompleteEvent),
    /// Error occurred
    Error(ErrorEvent),
}

/// Text content event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEvent {
    pub content: String,
    pub delta: bool,
}

impl TextEvent {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            delta: false,
        }
    }

    pub fn delta(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            delta: true,
        }
    }
}

/// Thinking/reasoning content event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingEvent {
    pub content: String,
    pub delta: bool,
}

impl ThinkingEvent {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            delta: false,
        }
    }

    pub fn delta(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            delta: true,
        }
    }
}

/// Tool call event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallEvent {
    pub tool_call_id: String,
    pub name: String,
    pub arguments: String,
    pub step: usize,
}

impl ToolCallEvent {
    pub fn new(tool_call: &crate::llm::ToolCall, step: usize) -> Self {
        Self {
            tool_call_id: tool_call.id.clone(),
            name: tool_call.function.name.clone(),
            arguments: tool_call.function.arguments.clone(),
            step,
        }
    }
}

/// Tool result event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultEvent {
    pub tool_call_id: String,
    pub name: String,
    pub result: String,
    pub step: usize,
    pub ephemeral: bool,
}

impl ToolResultEvent {
    pub fn new(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        result: impl Into<String>,
        step: usize,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            name: name.into(),
            result: result.into(),
            step,
            ephemeral: false,
        }
    }

    pub fn with_ephemeral(mut self, ephemeral: bool) -> Self {
        self.ephemeral = ephemeral;
        self
    }
}

/// Final response event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalResponseEvent {
    pub content: String,
    pub usage: Option<UsageSummary>,
    pub steps: usize,
}

impl FinalResponseEvent {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            usage: None,
            steps: 0,
        }
    }

    pub fn with_usage(mut self, usage: UsageSummary) -> Self {
        self.usage = Some(usage);
        self
    }

    pub fn with_steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }
}

/// Message start event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageStartEvent {
    pub role: String,
}

impl MessageStartEvent {
    pub fn user() -> Self {
        Self {
            role: "user".to_string(),
        }
    }

    pub fn assistant() -> Self {
        Self {
            role: "assistant".to_string(),
        }
    }
}

/// Message complete event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageCompleteEvent {
    pub role: String,
}

impl MessageCompleteEvent {
    pub fn user() -> Self {
        Self {
            role: "user".to_string(),
        }
    }

    pub fn assistant() -> Self {
        Self {
            role: "assistant".to_string(),
        }
    }
}

/// Step start event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepStartEvent {
    pub step: usize,
}

impl StepStartEvent {
    pub fn new(step: usize) -> Self {
        Self { step }
    }
}

/// Step complete event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepCompleteEvent {
    pub step: usize,
}

impl StepCompleteEvent {
    pub fn new(step: usize) -> Self {
        Self { step }
    }
}

/// Error event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEvent {
    pub message: String,
    pub code: Option<String>,
}

impl ErrorEvent {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: None,
        }
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }
}

/// Usage summary for the session
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageSummary {
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub total_tokens: u64,
    pub total_cost: Option<f64>,
    pub by_model: std::collections::HashMap<String, ModelUsage>,
}

impl UsageSummary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_usage(&mut self, model: &str, usage: &Usage) {
        self.total_prompt_tokens += usage.prompt_tokens;
        self.total_completion_tokens += usage.completion_tokens;
        self.total_tokens += usage.total_tokens;

        let model_usage = self.by_model.entry(model.to_string()).or_default();
        model_usage.prompt_tokens += usage.prompt_tokens;
        model_usage.completion_tokens += usage.completion_tokens;
        model_usage.total_tokens += usage.total_tokens;
        model_usage.calls += 1;
    }
}

/// Per-model usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub calls: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_event() {
        let event = TextEvent::new("Hello");
        assert_eq!(event.content, "Hello");
        assert!(!event.delta);

        let delta = TextEvent::delta("Hello");
        assert!(delta.delta);
    }

    #[test]
    fn test_usage_summary() {
        let mut summary = UsageSummary::new();
        let usage = Usage::new(100, 50);

        summary.add_usage("gpt-4o", &usage);

        assert_eq!(summary.total_prompt_tokens, 100);
        assert_eq!(summary.total_completion_tokens, 50);
        assert_eq!(summary.total_tokens, 150);
        assert!(summary.by_model.contains_key("gpt-4o"));
    }
}
