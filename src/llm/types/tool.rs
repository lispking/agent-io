//! Tool definition and function call types

use serde::{Deserialize, Serialize};

use super::content::JsonSchema;

/// Tool choice strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ToolChoice {
    /// Let the model decide whether to call tools
    #[default]
    Auto,
    /// Force the model to call a tool
    Required,
    /// Prevent the model from calling tools
    None,
    /// Force a specific tool to be called
    #[serde(untagged)]
    Named(String),
}

impl From<&str> for ToolChoice {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "auto" => ToolChoice::Auto,
            "required" => ToolChoice::Required,
            "none" => ToolChoice::None,
            name => ToolChoice::Named(name.to_string()),
        }
    }
}

/// Definition of a tool that can be called by the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Name of the tool
    pub name: String,
    /// Description of what the tool does
    pub description: String,
    /// JSON Schema for the tool parameters
    pub parameters: JsonSchema,
    /// Whether to use strict schema validation
    #[serde(default = "default_strict")]
    pub strict: bool,
}

fn default_strict() -> bool {
    true
}

impl ToolDefinition {
    /// Create a new tool definition
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: JsonSchema,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            strict: true,
        }
    }

    /// Set strict mode
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }
}

/// Function call from the LLM
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Function {
    /// Name of the function to call
    pub name: String,
    /// JSON string of arguments
    pub arguments: String,
}

impl Function {
    /// Parse arguments as a specific type
    pub fn parse_args<T: for<'de> Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.arguments)
    }
}

/// Tool call from the LLM
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Unique identifier for the tool call
    pub id: String,
    /// The function to call
    pub function: Function,
    /// Type of tool (always "function" for now)
    #[serde(default = "default_tool_type")]
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Thought signature for Gemini thinking models
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

fn default_tool_type() -> String {
    "function".to_string()
}

impl ToolCall {
    /// Create a new tool call
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            function: Function {
                name: name.into(),
                arguments: arguments.into(),
            },
            tool_type: "function".to_string(),
            thought_signature: None,
        }
    }

    /// Parse arguments as a specific type
    pub fn parse_args<T: for<'de> Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        self.function.parse_args()
    }
}
