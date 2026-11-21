//! Agent service - core execution loop

use std::collections::HashMap;
use std::sync::Arc;

use futures::{Stream, StreamExt};
use llm_resilience::CircuitBreaker;
use tokio::sync::RwLock;
use tracing::debug;

use crate::agent::{
    AgentEvent, ErrorEvent, FinalResponseEvent, StepCompleteEvent, StepStartEvent, UsageSummary,
};
use crate::llm::{
    AssistantMessage, BaseChatModel, ChatCompletion, Message, ToolDefinition, ToolMessage,
};
use crate::memory::{MemoryManager, MemoryType};
use crate::tools::Tool;
use crate::{Error, Result};

use super::builder::AgentBuilder;
use super::config::{AgentConfig, EphemeralConfig, build_ephemeral_config};

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
    /// Memory manager (optional)
    memory: Option<Arc<RwLock<MemoryManager>>>,
    /// Circuit breaker around LLM calls. Clones share state, so one breaker
    /// can be shared by every agent that talks to the same provider.
    breaker: CircuitBreaker,
}

/// Longest single backoff sleep between LLM retries.
pub(crate) const MAX_RETRY_DELAY: std::time::Duration = std::time::Duration::from_secs(30);

/// Double the backoff, but never past [`MAX_RETRY_DELAY`].
pub(crate) fn next_retry_delay(delay: std::time::Duration) -> std::time::Duration {
    (delay * 2).min(MAX_RETRY_DELAY)
}

/// Default breaker for LLM calls: opens after 5 consecutive rate limits,
/// lets one probe through after 30s, and closes when that probe succeeds.
pub fn default_llm_circuit_breaker() -> CircuitBreaker {
    CircuitBreaker::new(5, 1.0, std::time::Duration::from_secs(30))
}

impl Agent {
    /// Create a new agent
    pub fn new(llm: Arc<dyn BaseChatModel>, tools: Vec<Arc<dyn Tool>>) -> Self {
        let ephemeral_config = build_ephemeral_config(&tools);

        Self {
            llm,
            tools,
            config: AgentConfig::default(),
            history: Arc::new(RwLock::new(Vec::new())),
            usage: Arc::new(RwLock::new(UsageSummary::new())),
            ephemeral_config,
            memory: None,
            breaker: default_llm_circuit_breaker(),
        }
    }

    /// Use a specific circuit breaker for LLM calls (for example one shared
    /// with other agents that call the same provider).
    pub fn with_circuit_breaker(mut self, breaker: CircuitBreaker) -> Self {
        self.breaker = breaker;
        self
    }

    /// The circuit breaker guarding this agent's LLM calls.
    pub fn circuit_breaker(&self) -> &CircuitBreaker {
        &self.breaker
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

    /// Create agent with all components (used by builder)
    pub(super) fn new_with_config(
        llm: Arc<dyn BaseChatModel>,
        tools: Vec<Arc<dyn Tool>>,
        config: AgentConfig,
        ephemeral_config: HashMap<String, EphemeralConfig>,
        memory: Option<Arc<RwLock<MemoryManager>>>,
        breaker: Option<CircuitBreaker>,
    ) -> Self {
        Self {
            llm,
            tools,
            config,
            history: Arc::new(RwLock::new(Vec::new())),
            usage: Arc::new(RwLock::new(UsageSummary::new())),
            ephemeral_config,
            memory,
            breaker: breaker.unwrap_or_else(default_llm_circuit_breaker),
        }
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

    /// Query with memory context
    pub async fn query_with_memory(&self, message: impl Into<String>) -> Result<String> {
        let message = message.into();
        let context = self.recall_memory_context(&message).await?;

        {
            let mut history = self.history.write().await;
            if let Some(context_message) = self.memory_context_message(context) {
                history.push(context_message);
            }
            history.push(Message::user(message.clone()));
        }

        let stream = self.execute_loop();
        futures::pin_mut!(stream);

        while let Some(event) = stream.next().await {
            if let AgentEvent::FinalResponse(response) = event {
                self.remember_short_term(&message).await?;
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

    async fn recall_memory_context(&self, message: &str) -> Result<String> {
        if let Some(memory) = &self.memory {
            let mem = memory.read().await;
            mem.recall_context(message).await
        } else {
            Ok(String::new())
        }
    }

    fn memory_context_message(&self, context: String) -> Option<Message> {
        if context.is_empty() {
            None
        } else {
            Some(Message::developer(format!(
                "Relevant memory context:\n{}",
                context
            )))
        }
    }

    async fn remember_short_term(&self, message: &str) -> Result<()> {
        if let Some(memory) = &self.memory {
            let mut mem = memory.write().await;
            mem.remember(message, MemoryType::ShortTerm).await?;
        }
        Ok(())
    }

    async fn build_request_messages(&self) -> Vec<Message> {
        let history = self.history.read().await;
        let mut messages =
            Vec::with_capacity(history.len() + usize::from(self.config.system_prompt.is_some()));
        if let Some(ref prompt) = self.config.system_prompt {
            messages.push(Message::system(prompt));
        }
        messages.extend(history.iter().cloned());
        messages
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

                let full_messages = self.build_request_messages().await;

                // Build tool definitions
                let tool_defs: Vec<ToolDefinition> = self.tools.iter()
                    .map(|t| t.definition())
                    .collect();
                let tool_defs = if tool_defs.is_empty() { None } else { Some(tool_defs) };
                let tool_choice = self.config.tool_choice.clone();

                // Call LLM with retry
                let completion = match Self::call_llm_with_retry(
                    self.llm.as_ref(),
                    &self.breaker,
                    &full_messages,
                    tool_defs.as_deref(),
                    Some(&tool_choice),
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
                        h.push(Message::Assistant(AssistantMessage {
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
                            t.execute(args).await
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

    /// Call LLM with capped exponential backoff, behind a circuit breaker.
    ///
    /// Each rate limit counts as a failure on `breaker`. Once the provider has
    /// rate limited enough consecutive calls the breaker opens and this
    /// returns at once instead of sleeping, so a provider outage costs
    /// seconds rather than minutes per call, and agents sharing the breaker
    /// stop hammering the provider together.
    async fn call_llm_with_retry(
        llm: &dyn BaseChatModel,
        breaker: &CircuitBreaker,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        tool_choice: Option<&crate::llm::ToolChoice>,
    ) -> Result<ChatCompletion> {
        let max_retries = 10;
        let mut delay = std::time::Duration::from_millis(500);

        for attempt in 0..=max_retries {
            let Some(permit) = breaker.try_acquire() else {
                return Err(Self::circuit_open_error(llm, breaker));
            };

            let request_messages = messages.to_vec();
            let request_tools = tools.map(|value| value.to_vec());
            let request_tool_choice = tool_choice.cloned();

            match llm
                .invoke(request_messages, request_tools, request_tool_choice)
                .await
            {
                Ok(completion) => {
                    permit.success();
                    return Ok(completion);
                }
                Err(crate::llm::LlmError::RateLimit) => {
                    permit.failure();
                    if breaker.is_open_sync() {
                        // Do not sleep toward a retry the breaker will refuse.
                        return Err(Self::circuit_open_error(llm, breaker));
                    }
                    if attempt == max_retries {
                        return Err(Error::Llm(crate::llm::LlmError::RateLimit));
                    }
                    tracing::warn!(
                        "Rate limit or empty response, retrying in {:?} (attempt {}/{})",
                        delay,
                        attempt + 1,
                        max_retries
                    );
                    tokio::time::sleep(delay).await;
                    delay = next_retry_delay(delay);
                }
                // Other errors (auth, bad request, config) say nothing about
                // provider load: drop the permit without recording.
                Err(e) => return Err(Error::Llm(e)),
            }
        }

        Err(Error::Agent("Max retries exceeded".into()))
    }

    fn circuit_open_error(llm: &dyn BaseChatModel, breaker: &CircuitBreaker) -> Error {
        let wait = breaker.retry_after_sync().unwrap_or_default();
        Error::Agent(format!(
            "LLM circuit breaker open after repeated rate limits from {}; next probe in {:?}",
            llm.provider(),
            wait
        ))
    }

    /// Get usage summary
    pub async fn get_usage(&self) -> UsageSummary {
        self.usage.read().await.clone()
    }

    /// Destroy old ephemeral message content, keeping the last N per tool.
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

    /// Check if memory is enabled
    pub fn has_memory(&self) -> bool {
        self.memory.is_some()
    }

    /// Get memory manager reference
    pub fn get_memory(&self) -> Option<&Arc<RwLock<MemoryManager>>> {
        self.memory.as_ref()
    }
}

#[cfg(test)]
mod retry_tests {
    use super::*;
    use crate::llm::{ChatStream, LlmError, ToolChoice};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Returns `RateLimit` for the first `fail_first` calls, then succeeds.
    struct FlakyLlm {
        calls: AtomicUsize,
        fail_first: usize,
    }

    impl FlakyLlm {
        fn new(fail_first: usize) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                fail_first,
            }
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl BaseChatModel for FlakyLlm {
        fn model(&self) -> &str {
            "flaky"
        }
        fn provider(&self) -> &str {
            "test"
        }
        async fn invoke(
            &self,
            _messages: Vec<Message>,
            _tools: Option<Vec<ToolDefinition>>,
            _tool_choice: Option<ToolChoice>,
        ) -> std::result::Result<ChatCompletion, LlmError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_first {
                Err(LlmError::RateLimit)
            } else {
                Ok(ChatCompletion::text("ok"))
            }
        }
        async fn invoke_stream(
            &self,
            _messages: Vec<Message>,
            _tools: Option<Vec<ToolDefinition>>,
            _tool_choice: Option<ToolChoice>,
        ) -> std::result::Result<ChatStream, LlmError> {
            Err(LlmError::Config("not used".into()))
        }
    }

    #[test]
    fn backoff_is_capped() {
        // Old schedule: 500ms doubling 10 times, last sleep 256s, ~8.5 min total.
        let mut delay = Duration::from_millis(500);
        let mut total = Duration::ZERO;
        for _ in 0..10 {
            total += delay;
            delay = next_retry_delay(delay);
        }
        assert_eq!(delay, MAX_RETRY_DELAY);
        assert!(total <= Duration::from_secs(200), "total backoff {total:?}");
    }

    #[tokio::test]
    async fn retries_through_transient_rate_limit() {
        let llm = FlakyLlm::new(1);
        let breaker = default_llm_circuit_breaker();
        let out = Agent::call_llm_with_retry(&llm, &breaker, &[], None, None)
            .await
            .expect("second attempt succeeds");
        assert_eq!(out.content.as_deref(), Some("ok"));
        assert_eq!(llm.calls(), 2);
        assert!(breaker.is_closed_sync());
    }

    #[tokio::test]
    async fn breaker_opens_and_later_calls_fail_fast() {
        let llm = FlakyLlm::new(usize::MAX);
        let breaker = CircuitBreaker::new(2, 1.0, Duration::from_secs(60));

        let started = std::time::Instant::now();
        let err = Agent::call_llm_with_retry(&llm, &breaker, &[], None, None)
            .await
            .unwrap_err();
        // Two attempts and one 500ms sleep, not ten attempts and minutes of sleep.
        assert_eq!(llm.calls(), 2);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(err.to_string().contains("circuit breaker open"), "{err}");
        assert!(breaker.is_open_sync());

        // A second call (another step, or another agent sharing the breaker)
        // does not reach the provider at all while the breaker is open.
        let err = Agent::call_llm_with_retry(&llm, &breaker, &[], None, None)
            .await
            .unwrap_err();
        assert_eq!(llm.calls(), 2);
        assert!(err.to_string().contains("circuit breaker open"), "{err}");
    }
}
