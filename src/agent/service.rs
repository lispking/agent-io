//! Agent service - core execution loop

use std::collections::HashMap;
use std::sync::Arc;

use derive_builder::Builder;
use futures::{Stream, StreamExt};
use tokio::sync::RwLock;
use tracing::debug;

use crate::agent::{
    AgentEvent, ErrorEvent, FinalResponseEvent, StepCompleteEvent, StepStartEvent, UsageSummary,
};
use crate::llm::{BaseChatModel, ChatCompletion, Message, ToolChoice, ToolDefinition, ToolMessage};
use crate::tools::Tool;
use crate::{Error, Result};

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

/// Agent - the main orchestrator for LLM interactions
pub struct Agent {
    /// LLM provider
    llm: Arc<dyn BaseChatModel>,
    /// Available tools
    tools: Vec<Arc<dyn Tool>>,
    /// Configuration
    config: AgentConfig,
    /// Message history
    history: Arc<RwLock<Vec<Message>>>,
    /// Usage tracking
    usage: Arc<RwLock<UsageSummary>>,
    /// Ephemeral config per tool name
    ephemeral_config: HashMap<String, EphemeralConfig>,
}

impl Agent {
    /// Create a new agent
    pub fn new(llm: Arc<dyn BaseChatModel>, tools: Vec<Arc<dyn Tool>>) -> Self {
        // Build ephemeral config from tools
        let ephemeral_config = tools
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

        Self {
            llm,
            tools,
            config: AgentConfig::default(),
            history: Arc::new(RwLock::new(Vec::new())),
            usage: Arc::new(RwLock::new(UsageSummary::new())),
            ephemeral_config,
        }
    }

    /// Create an agent builder
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    /// Set configuration
    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    /// Query the agent synchronously (returns final response)
    pub async fn query(&self, message: impl Into<String>) -> Result<String> {
        // Add user message to history
        {
            let mut history = self.history.write().await;
            history.push(Message::user(message.into()));
        }

        // Execute and collect result
        let stream = self.execute_loop();
        futures::pin_mut!(stream);

        while let Some(event) = stream.next().await {
            if let AgentEvent::FinalResponse(response) = event {
                return Ok(response.content);
            }
        }

        Err(Error::Agent("No final response received".into()))
    }

    /// Query the agent with streaming events
    pub async fn query_stream<'a, M: Into<String> + 'a>(
        &'a self,
        message: M,
    ) -> Result<impl Stream<Item = AgentEvent> + 'a> {
        // Add user message to history
        {
            let mut history = self.history.write().await;
            history.push(Message::user(message.into()));
        }

        Ok(self.execute_loop())
    }

    /// Main execution loop
    fn execute_loop(&self) -> impl Stream<Item = AgentEvent> + '_ {
        async_stream::stream! {
            let mut step = 0;

            loop {
                if step >= self.config.max_iterations {
                    yield AgentEvent::Error(ErrorEvent::new("Max iterations exceeded"));
                    break;
                }

                yield AgentEvent::StepStart(StepStartEvent::new(step));

                // Destroy ephemeral messages from previous iteration
                {
                    let mut h = self.history.write().await;
                    Self::destroy_ephemeral_messages(&mut h, &self.ephemeral_config);
                }

                // Get current history
                let messages = {
                    let h = self.history.read().await;
                    h.clone()
                };

                // Build system prompt + messages
                let mut full_messages = Vec::new();
                if let Some(ref prompt) = self.config.system_prompt
                    && step == 0 {
                        full_messages.push(Message::system(prompt));
                    }
                full_messages.extend(messages);

                // Build tool definitions
                let tool_defs: Vec<ToolDefinition> = self.tools.iter()
                    .map(|t| t.definition())
                    .collect();

                // Call LLM with retry
                let completion = match Self::call_llm_with_retry(
                    self.llm.as_ref(),
                    full_messages.clone(),
                    if tool_defs.is_empty() { None } else { Some(tool_defs) },
                    Some(self.config.tool_choice.clone()),
                ).await {
                    Ok(c) => c,
                    Err(e) => {
                        yield AgentEvent::Error(ErrorEvent::new(e.to_string()));
                        break;
                    }
                };

                // Track usage
                if let Some(ref u) = completion.usage {
                    let mut us = self.usage.write().await;
                    us.add_usage(self.llm.model(), u);
                }

                // Yield thinking content
                if let Some(ref thinking) = completion.thinking {
                    yield AgentEvent::Thinking(crate::agent::ThinkingEvent::new(thinking));
                }

                // Yield text content
                if let Some(ref content) = completion.content {
                    yield AgentEvent::Text(crate::agent::TextEvent::new(content));
                }

                // Handle tool calls
                if completion.has_tool_calls() {
                    // Add assistant message to history
                    {
                        let mut h = self.history.write().await;
                        h.push(Message::Assistant(crate::llm::AssistantMessage {
                            role: "assistant".to_string(),
                            content: completion.content.clone(),
                            thinking: completion.thinking.clone(),
                            redacted_thinking: None,
                            tool_calls: completion.tool_calls.clone(),
                            refusal: None,
                        }));
                    }

                    // Execute tools
                    for tool_call in &completion.tool_calls {
                        yield AgentEvent::ToolCall(crate::agent::ToolCallEvent::new(tool_call, step));

                        // Find tool
                        let tool = self.tools.iter().find(|t| t.name() == tool_call.function.name);

                        let result = if let Some(t) = tool {
                            let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
                                .unwrap_or(serde_json::json!({}));
                            t.execute(args, None).await
                        } else {
                            Ok(crate::tools::ToolResult::new(&tool_call.id, format!("Unknown tool: {}", tool_call.function.name)))
                        };

                        match result {
                            Ok(tool_result) => {
                                yield AgentEvent::ToolResult(
                                    crate::agent::ToolResultEvent::new(
                                        &tool_call.id,
                                        &tool_call.function.name,
                                        &tool_result.content,
                                        step,
                                    ).with_ephemeral(tool_result.ephemeral)
                                );

                                // Add tool result to history with ephemeral metadata
                                {
                                    let mut h = self.history.write().await;
                                    let mut msg = ToolMessage::new(&tool_call.id, tool_result.content);
                                    msg.tool_name = Some(tool_call.function.name.clone());
                                    msg.ephemeral = tool_result.ephemeral;
                                    h.push(Message::Tool(msg));
                                }
                            }
                            Err(e) => {
                                yield AgentEvent::Error(ErrorEvent::new(format!(
                                    "Tool execution failed: {}",
                                    e
                                )));
                            }
                        }
                    }

                    step += 1;
                    yield AgentEvent::StepComplete(StepCompleteEvent::new(step - 1));
                    continue;
                }

                // No tool calls - we're done
                let final_response = FinalResponseEvent::new(completion.content.clone().unwrap_or_default())
                    .with_steps(step);

                yield AgentEvent::FinalResponse(final_response);
                yield AgentEvent::StepComplete(StepCompleteEvent::new(step));
                break;
            }
        }
    }

    /// Call LLM with exponential backoff retry
    async fn call_llm_with_retry(
        llm: &dyn BaseChatModel,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
    ) -> Result<ChatCompletion> {
        let max_retries = 3;
        let mut delay = std::time::Duration::from_millis(100);

        for attempt in 0..=max_retries {
            match llm
                .invoke(messages.clone(), tools.clone(), tool_choice.clone())
                .await
            {
                Ok(completion) => return Ok(completion),
                Err(crate::llm::LlmError::RateLimit) if attempt < max_retries => {
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                }
                Err(e) => return Err(Error::Llm(e)),
            }
        }

        Err(Error::Agent("Max retries exceeded".into()))
    }

    /// Get usage summary
    pub async fn get_usage(&self) -> UsageSummary {
        self.usage.read().await.clone()
    }

    /// Destroy old ephemeral message content, keeping the last N per tool.
    ///
    /// This should be called before each LLM invocation to manage context size.
    fn destroy_ephemeral_messages(
        history: &mut [Message],
        ephemeral_config: &HashMap<String, EphemeralConfig>,
    ) {
        // First pass: collect indices of ephemeral messages by tool name
        let mut ephemeral_indices_by_tool: HashMap<String, Vec<usize>> = HashMap::new();

        for (idx, msg) in history.iter().enumerate() {
            let tool_msg = match msg {
                Message::Tool(t) => t,
                _ => continue,
            };

            if !tool_msg.ephemeral || tool_msg.destroyed {
                continue;
            }

            let tool_name = match &tool_msg.tool_name {
                Some(name) => name.clone(),
                None => continue,
            };

            ephemeral_indices_by_tool
                .entry(tool_name)
                .or_default()
                .push(idx);
        }

        // Collect all indices to destroy
        let mut indices_to_destroy: Vec<usize> = Vec::new();

        for (tool_name, indices) in ephemeral_indices_by_tool {
            let keep_count = ephemeral_config
                .get(&tool_name)
                .map(|c| c.keep_count)
                .unwrap_or(1);

            // Destroy messages beyond the keep limit (older ones first)
            let to_destroy = if keep_count > 0 && indices.len() > keep_count {
                &indices[..indices.len() - keep_count]
            } else {
                &indices[..]
            };

            indices_to_destroy.extend(to_destroy.iter().copied());
        }

        // Second pass: destroy the messages
        for idx in indices_to_destroy {
            if let Message::Tool(tool_msg) = &mut history[idx] {
                debug!("Destroying ephemeral message at index {}", idx);
                tool_msg.destroy();
            }
        }
    }

    /// Clear history
    pub async fn clear_history(&self) {
        let mut history = self.history.write().await;
        history.clear();
    }

    /// Load history
    pub async fn load_history(&self, messages: Vec<Message>) {
        let mut history = self.history.write().await;
        *history = messages;
    }

    /// Get current history
    pub async fn get_history(&self) -> Vec<Message> {
        self.history.read().await.clone()
    }
}

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
            .ok_or_else(|| Error::Config("LLM is required".into()))?;

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

        Ok(Agent {
            llm,
            tools: self.tools,
            config: self.config.unwrap_or_default(),
            history: Arc::new(RwLock::new(Vec::new())),
            usage: Arc::new(RwLock::new(UsageSummary::new())),
            ephemeral_config,
        })
    }
}
