//! JSON Schema optimizer for tool definitions

use serde_json::{Map, Value, json};
use std::collections::HashSet;

/// Schema optimizer for creating LLM-compatible JSON schemas
pub struct SchemaOptimizer;

impl SchemaOptimizer {
    /// Create an optimized JSON schema from a raw schema
    ///
    /// This function:
    /// - Flattens $ref and $defs
    /// - Removes additionalProperties
    /// - Ensures all properties are required
    pub fn optimize(schema: &Value) -> Value {
        let mut result = schema.clone();
        Self::optimize_recursive(&mut result);
        result
    }

    fn optimize_recursive(value: &mut Value) {
        match value {
            Value::Object(obj) => {
                // Remove additionalProperties
                obj.remove("additionalProperties");

                // Handle $ref by inlining the definition
                if let Some(ref_val) = obj.get("$ref").and_then(|r| r.as_str())
                    && let Some(defs) = obj.get("$defs")
                    && let Some(def) = ref_val.strip_prefix("#/$defs/")
                    && let Some(resolved) = defs.get(def)
                {
                    *value = resolved.clone();
                    Self::optimize_recursive(value);
                    return;
                }

                // Remove $defs if present
                obj.remove("$defs");

                // Make all properties required
                if let Some(properties) = obj.get("properties").and_then(|p| p.as_object()) {
                    let all_keys: HashSet<&str> = properties.keys().map(|k| k.as_str()).collect();
                    obj.insert("required".to_string(), json!(all_keys));
                }

                // Recursively process nested objects
                for (_, v) in obj.iter_mut() {
                    Self::optimize_recursive(v);
                }
            }
            Value::Array(arr) => {
                for item in arr.iter_mut() {
                    Self::optimize_recursive(item);
                }
            }
            _ => {}
        }
    }

    /// Create a tool definition from a JSON schema
    pub fn create_tool_definition(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: Value,
    ) -> super::ToolDefinition {
        let optimized = Self::optimize(&schema);
        let parameters = optimized.as_object().cloned().unwrap_or_else(|| {
            let mut map = Map::new();
            map.insert("type".to_string(), json!("object"));
            map.insert("properties".to_string(), json!({}));
            map
        });

        super::ToolDefinition {
            name: name.into(),
            description: description.into(),
            parameters,
            strict: true,
        }
    }

    /// Create a minimal schema for a simple string parameter
    pub fn string_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "The string value"
                }
            },
            "required": ["value"]
        })
    }

    /// Create a schema for multiple string parameters
    pub fn string_params_schema(params: &[(&str, &str)]) -> Value {
        let properties: Map<String, Value> = params
            .iter()
            .map(|(name, desc)| {
                (
                    name.to_string(),
                    json!({
                        "type": "string",
                        "description": desc
                    }),
                )
            })
            .collect();

        let required: Vec<&str> = params.iter().map(|(name, _)| *name).collect();

        json!({
            "type": "object",
            "properties": properties,
            "required": required
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_optimize_schema() {
        let schema = json!({
            "$ref": "#/$defs/MyType",
            "$defs": {
                "MyType": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    },
                    "additionalProperties": false
                }
            }
        });

        let optimized = SchemaOptimizer::optimize(&schema);

        assert!(optimized.get("$ref").is_none());
        assert!(optimized.get("$defs").is_none());
        assert!(optimized.get("additionalProperties").is_none());
        assert!(optimized.get("required").is_some());
    }
}
