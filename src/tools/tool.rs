//! Tool trait and implementations

use async_trait::async_trait;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::Result;
use crate::llm::ToolDefinition;

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Tool call ID
    pub tool_call_id: String,
    /// Result content
    pub content: String,
    /// Whether this result should be ephemeral (removed after use)
    #[serde(default)]
    pub ephemeral: bool,
}

impl ToolResult {
    pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            ephemeral: false,
        }
    }

    pub fn with_ephemeral(mut self, ephemeral: bool) -> Self {
        self.ephemeral = ephemeral;
        self
    }
}

/// Trait for defining tools that can be called by an LLM
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get the tool name
    fn name(&self) -> &str;

    /// Get the tool description
    fn description(&self) -> &str;

    /// Get the tool definition (JSON Schema)
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with given arguments
    async fn execute(
        &self,
        args: Value,
        overrides: Option<DependencyOverrides>,
    ) -> Result<ToolResult>;

    /// Whether tool outputs should be ephemeral (removed from context after use)
    fn ephemeral(&self) -> EphemeralConfig {
        EphemeralConfig::None
    }
}

/// Configuration for ephemeral tool outputs
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum EphemeralConfig {
    /// Not ephemeral
    #[default]
    None,
    /// Ephemeral, removed after one use
    Single,
    /// Keep last N outputs in context
    Count(usize),
}

/// A tool implementation using a function
pub struct FunctionTool<T, F>
where
    T: DeserializeOwned + Send + Sync + 'static,
    F: Fn(T) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
{
    name: String,
    description: String,
    parameters_schema: serde_json::Map<String, Value>,
    func: F,
    ephemeral_config: EphemeralConfig,
    _marker: std::marker::PhantomData<T>,
}

impl<T, F> FunctionTool<T, F>
where
    T: DeserializeOwned + Send + Sync + 'static,
    F: Fn(T) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
{
    /// Create a new function tool
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters_schema: serde_json::Map<String, Value>,
        func: F,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters_schema,
            func,
            ephemeral_config: EphemeralConfig::None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set ephemeral configuration
    pub fn with_ephemeral(mut self, config: EphemeralConfig) -> Self {
        self.ephemeral_config = config;
        self
    }
}

#[async_trait]
impl<T, F> Tool for FunctionTool<T, F>
where
    T: DeserializeOwned + Send + Sync + 'static,
    F: Fn(T) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            &self.name,
            &self.description,
            self.parameters_schema.clone(),
        )
    }

    async fn execute(
        &self,
        args: Value,
        _overrides: Option<DependencyOverrides>,
    ) -> Result<ToolResult> {
        let parsed: T = serde_json::from_value(args)?;
        let content = (self.func)(parsed).await?;
        Ok(ToolResult::new("", content)
            .with_ephemeral(self.ephemeral_config != EphemeralConfig::None))
    }

    fn ephemeral(&self) -> EphemeralConfig {
        self.ephemeral_config
    }
}

/// Builder for creating tools
pub struct ToolBuilder {
    name: String,
    description: String,
    parameters_schema: serde_json::Map<String, Value>,
    ephemeral: EphemeralConfig,
}

impl ToolBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            parameters_schema: serde_json::Map::new(),
            ephemeral: EphemeralConfig::None,
        }
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn parameter(mut self, name: &str, schema: Value) -> Self {
        self.parameters_schema.insert(name.to_string(), schema);
        self
    }

    pub fn string_param(self, name: &str, description: &str) -> Self {
        self.parameter(
            name,
            json!({
                "type": "string",
                "description": description
            }),
        )
    }

    pub fn number_param(self, name: &str, description: &str) -> Self {
        self.parameter(
            name,
            json!({
                "type": "number",
                "description": description
            }),
        )
    }

    pub fn boolean_param(self, name: &str, description: &str) -> Self {
        self.parameter(
            name,
            json!({
                "type": "boolean",
                "description": description
            }),
        )
    }

    pub fn ephemeral(mut self, config: EphemeralConfig) -> Self {
        self.ephemeral = config;
        self
    }

    pub fn build<F, T>(self, func: F) -> Box<dyn Tool>
    where
        T: DeserializeOwned + Send + Sync + 'static,
        F: Fn(T) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>>
            + Send
            + Sync
            + 'static,
    {
        let mut tool = FunctionTool::new(self.name, self.description, self.parameters_schema, func);
        tool.ephemeral_config = self.ephemeral;
        Box::new(tool)
    }
}

/// Dependency overrides for testing
pub type DependencyOverrides =
    std::collections::HashMap<String, Box<dyn std::any::Any + Send + Sync>>;

/// Simple tool that takes no arguments
pub struct SimpleTool<F>
where
    F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
{
    name: String,
    description: String,
    func: F,
}

impl<F> SimpleTool<F>
where
    F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
{
    pub fn new(name: impl Into<String>, description: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            func,
        }
    }
}

#[async_trait]
impl<F> Tool for SimpleTool<F>
where
    F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(&self.name, &self.description, serde_json::Map::new())
    }

    async fn execute(
        &self,
        _args: Value,
        _overrides: Option<DependencyOverrides>,
    ) -> Result<ToolResult> {
        let content = (self.func)().await?;
        Ok(ToolResult::new("", content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_tool() {
        let tool = SimpleTool::new("ping", "Returns pong", || {
            Box::pin(async { Ok("pong".to_string()) })
        });

        assert_eq!(tool.name(), "ping");

        let result = tool.execute(json!({}), None).await.unwrap();
        assert_eq!(result.content, "pong");
    }

    #[tokio::test]
    async fn test_function_tool() {
        #[derive(Deserialize)]
        struct EchoArgs {
            message: String,
        }

        let tool = FunctionTool::new(
            "echo",
            "Echoes the message back",
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                },
                "required": ["message"]
            })
            .as_object()
            .unwrap()
            .clone(),
            |args: EchoArgs| Box::pin(async move { Ok(args.message) }),
        );

        let result = tool
            .execute(json!({"message": "hello"}), None)
            .await
            .unwrap();
        assert_eq!(result.content, "hello");
    }
}
