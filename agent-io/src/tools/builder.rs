//! Tool builder

use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::Result;

use super::function::FunctionTool;
use super::tool::{EphemeralConfig, Tool};

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
