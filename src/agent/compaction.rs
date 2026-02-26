//! Context compaction for long-running conversations

use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use crate::llm::Message;

/// Default summary prompt for compaction
pub const DEFAULT_SUMMARY_PROMPT: &str = r#"Summarize the conversation so far, preserving:
1. Key decisions and their reasons
2. Important facts learned
3. Current task state and next steps
4. Any user preferences or constraints

Format the summary as a structured markdown document."#;

/// Compaction configuration
#[derive(Debug, Clone, Builder)]
#[builder(pattern = "owned")]
pub struct CompactionConfig {
    /// Enable compaction
    #[builder(default = "true")]
    pub enabled: bool,

    /// Threshold ratio of context window to trigger compaction
    #[builder(default = "0.80")]
    pub threshold_ratio: f32,

    /// Model to use for generating summaries
    #[builder(setter(into, strip_option), default = "None")]
    pub model: Option<String>,

    /// Custom summary prompt
    #[builder(default = "DEFAULT_SUMMARY_PROMPT.to_string()")]
    pub summary_prompt: String,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_ratio: 0.80,
            model: None,
            summary_prompt: DEFAULT_SUMMARY_PROMPT.to_string(),
        }
    }
}

/// Token usage tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
}

impl TokenUsage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    pub fn add_input(&mut self, tokens: u64, cached: bool) {
        if cached {
            self.cache_read_tokens += tokens;
        } else {
            self.input_tokens += tokens;
        }
    }

    pub fn add_output(&mut self, tokens: u64) {
        self.output_tokens += tokens;
    }

    pub fn add_cache_creation(&mut self, tokens: u64) {
        self.cache_creation_tokens += tokens;
    }
}

/// Compaction service
pub struct CompactionService {
    config: CompactionConfig,
    usage: TokenUsage,
}

impl CompactionService {
    pub fn new(config: CompactionConfig) -> Self {
        Self {
            config,
            usage: TokenUsage::new(),
        }
    }

    /// Check if compaction is needed based on current usage
    pub fn should_compact(&self, current_tokens: u64, context_window: u64) -> bool {
        if !self.config.enabled || context_window == 0 {
            return false;
        }

        let threshold = (context_window as f32 * self.config.threshold_ratio) as u64;
        current_tokens >= threshold
    }

    /// Get the summary prompt
    pub fn summary_prompt(&self) -> &str {
        &self.config.summary_prompt
    }

    /// Update token usage
    pub fn update_usage(&mut self, usage: &TokenUsage) {
        self.usage.input_tokens += usage.input_tokens;
        self.usage.output_tokens += usage.output_tokens;
        self.usage.cache_creation_tokens += usage.cache_creation_tokens;
        self.usage.cache_read_tokens += usage.cache_read_tokens;
    }

    /// Get current usage
    pub fn get_usage(&self) -> &TokenUsage {
        &self.usage
    }
}

/// Result of compaction
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// The summary message
    pub summary: Message,
    /// Number of messages removed
    pub messages_removed: usize,
    /// Tokens saved (approximate)
    pub tokens_saved: u64,
}

impl CompactionResult {
    pub fn new(summary: impl Into<String>, messages_removed: usize, tokens_saved: u64) -> Self {
        Self {
            summary: Message::system(summary.into()),
            messages_removed,
            tokens_saved,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_compact() {
        let config = CompactionConfig::default();
        let service = CompactionService::new(config);

        // Below threshold
        assert!(!service.should_compact(50, 100));

        // At threshold
        assert!(service.should_compact(80, 100));

        // Above threshold
        assert!(service.should_compact(90, 100));
    }

    #[test]
    fn test_disabled_compaction() {
        let config = CompactionConfig {
            enabled: false,
            ..Default::default()
        };
        let service = CompactionService::new(config);

        assert!(!service.should_compact(99, 100));
    }

    #[test]
    fn test_token_usage() {
        let mut usage = TokenUsage::new();
        usage.add_input(100, false);
        usage.add_input(50, true);
        usage.add_output(75);

        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.cache_read_tokens, 50);
        assert_eq!(usage.output_tokens, 75);
        assert_eq!(usage.total(), 175);
    }
}
