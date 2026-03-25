//! Simple tool implementation

use async_trait::async_trait;

use crate::Result;
use crate::llm::ToolDefinition;

use super::tool::{Tool, ToolResult};

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

    async fn execute(&self, _args: serde_json::Value) -> Result<ToolResult> {
        let content = (self.func)().await?;
        Ok(ToolResult::new("", content))
    }
}
