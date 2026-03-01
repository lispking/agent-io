//! Agent builder

use std::sync::Arc;

use crate::Result;
use crate::llm::BaseChatModel;
use crate::tools::Tool;

use super::config::{AgentConfig, EphemeralConfig};
use super::service::Agent;

/// Agent builder
#[derive(Default)]
pub struct AgentBuilder {
    llm: Option<Arc<dyn BaseChatModel>>,
    tools: Vec<Arc<dyn Tool>>,
    config: Option<AgentConfig>,
}

impl AgentBuilder {
    pub fn with_llm(mut self, llm: Arc<dyn BaseChatModel>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }

    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        let mut config = self.config.unwrap_or_default();
        config.system_prompt = Some(prompt.into());
        self.config = Some(config);
        self
    }

    pub fn max_iterations(mut self, max: usize) -> Self {
        let mut config = self.config.unwrap_or_default();
        config.max_iterations = max;
        self.config = Some(config);
        self
    }

    pub fn build(self) -> Result<Agent> {
        let llm = self
            .llm
            .ok_or_else(|| crate::Error::Config("LLM is required".into()))?;

        // Build ephemeral config from tools
        let ephemeral_config = self
            .tools
            .iter()
            .filter_map(|t| {
                let cfg = t.ephemeral();
                if cfg != crate::tools::EphemeralConfig::None {
                    let keep_count = match cfg {
                        crate::tools::EphemeralConfig::Single => 1,
                        crate::tools::EphemeralConfig::Count(n) => n,
                        crate::tools::EphemeralConfig::None => 0,
                    };
                    Some((t.name().to_string(), EphemeralConfig { keep_count }))
                } else {
                    None
                }
            })
            .collect();

        Ok(Agent::new_with_config(
            llm,
            self.tools,
            self.config.unwrap_or_default(),
            ephemeral_config,
        ))
    }
}
