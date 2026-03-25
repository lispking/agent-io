//! Function tool implementation

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::Result;
use crate::llm::ToolDefinition;

use super::tool::{EphemeralConfig, Tool, ToolResult};

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
    pub(super) ephemeral_config: EphemeralConfig,
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

    async fn execute(&self, args: Value) -> Result<ToolResult> {
        let parsed: T = serde_json::from_value(args)?;
        let content = (self.func)(parsed).await?;
        Ok(ToolResult::new("", content)
            .with_ephemeral(self.ephemeral_config != EphemeralConfig::None))
    }

    fn ephemeral(&self) -> EphemeralConfig {
        self.ephemeral_config
    }
}
