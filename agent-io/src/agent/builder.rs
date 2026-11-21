//! Agent builder

use std::sync::Arc;

use crate::Result;
use crate::llm::BaseChatModel;
use crate::memory::MemoryManager;
use crate::tools::Tool;
use llm_resilience::CircuitBreaker;

use super::config::{AgentConfig, build_ephemeral_config};
use super::service::Agent;

/// Agent builder
#[derive(Default)]
pub struct AgentBuilder {
    llm: Option<Arc<dyn BaseChatModel>>,
    tools: Vec<Arc<dyn Tool>>,
    config: Option<AgentConfig>,
    memory: Option<Arc<RwLock<MemoryManager>>>,
    breaker: Option<CircuitBreaker>,
}

use tokio::sync::RwLock;

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

    /// Add memory manager to the agent
    pub fn with_memory(mut self, memory: Arc<RwLock<MemoryManager>>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Guard LLM calls with this circuit breaker. Clones share state, so
    /// passing the same breaker to several agents makes them back off together
    /// when their provider is rate limiting.
    pub fn with_circuit_breaker(mut self, breaker: CircuitBreaker) -> Self {
        self.breaker = Some(breaker);
        self
    }

    pub fn build(self) -> Result<Agent> {
        let llm = self
            .llm
            .ok_or_else(|| crate::Error::Config("LLM is required".into()))?;

        let ephemeral_config = build_ephemeral_config(&self.tools);

        Ok(Agent::new_with_config(
            llm,
            self.tools,
            self.config.unwrap_or_default(),
            ephemeral_config,
            self.memory,
            self.breaker,
        ))
    }
}
