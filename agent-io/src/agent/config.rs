//! Agent configuration types

use std::collections::HashMap;
use std::sync::Arc;

use derive_builder::Builder;

use crate::llm::ToolChoice;
use crate::tools::{EphemeralConfig as ToolEphemeralConfig, Tool};

/// Default maximum iterations
pub const DEFAULT_MAX_ITERATIONS: usize = 200;

/// Agent configuration
#[derive(Builder, Clone)]
#[builder(pattern = "owned")]
pub struct AgentConfig {
    /// System prompt
    #[builder(setter(into, strip_option), default = "None")]
    pub system_prompt: Option<String>,

    /// Maximum iterations before stopping
    #[builder(default = "DEFAULT_MAX_ITERATIONS")]
    pub max_iterations: usize,

    /// Tool choice strategy
    #[builder(default = "ToolChoice::Auto")]
    pub tool_choice: ToolChoice,

    /// Enable context compaction
    #[builder(default = "false")]
    pub enable_compaction: bool,

    /// Compaction threshold (ratio of context window)
    #[builder(default = "0.80")]
    pub compaction_threshold: f32,

    /// Enable cost tracking
    #[builder(default = "false")]
    pub include_cost: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tool_choice: ToolChoice::Auto,
            enable_compaction: false,
            compaction_threshold: 0.80,
            include_cost: false,
        }
    }
}

/// Ephemeral message configuration per tool
#[derive(Debug, Clone, Copy)]
pub struct EphemeralConfig {
    /// How many outputs to keep (None = not ephemeral)
    pub keep_count: usize,
}

impl Default for EphemeralConfig {
    fn default() -> Self {
        Self { keep_count: 1 }
    }
}

pub(crate) fn build_ephemeral_config(tools: &[Arc<dyn Tool>]) -> HashMap<String, EphemeralConfig> {
    tools
        .iter()
        .filter_map(|tool| {
            let keep_count = match tool.ephemeral() {
                ToolEphemeralConfig::None => return None,
                ToolEphemeralConfig::Single => 1,
                ToolEphemeralConfig::Count(count) => count,
            };
            Some((tool.name().to_string(), EphemeralConfig { keep_count }))
        })
        .collect()
}
