//! Model name mappings for pricing lookup

use std::collections::HashMap;

/// Map model names to LiteLLM compatible names
pub fn model_to_litellm() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();

    // Gemini models
    map.insert("gemini-flash-latest", "gemini/gemini-flash-latest");
    map.insert("gemini-pro-latest", "gemini/gemini-pro-latest");
    map.insert("gemini-2.0-flash", "gemini/gemini-2.0-flash");
    map.insert("gemini-2.0-flash-exp", "gemini/gemini-2.0-flash-exp");
    map.insert("gemini-2.5-pro", "gemini/gemini-2.5-pro-preview-06-05");
    map.insert("gemini-2.5-flash", "gemini/gemini-2.5-flash-preview-05-20");

    // DeepSeek models
    map.insert("deepseek-chat", "deepseek/deepseek-chat");
    map.insert("deepseek-coder", "deepseek/deepseek-coder");
    map.insert("deepseek-reasoner", "deepseek/deepseek-reasoner");

    // Groq models
    map.insert("llama-3.3-70b-versatile", "groq/llama-3.3-70b-versatile");
    map.insert("llama-3.3-70b-specdec", "groq/llama-3.3-70b-specdec");
    map.insert("llama-3.1-8b-instant", "groq/llama-3.1-8b-instant");
    map.insert("mixtral-8x7b-32768", "groq/mixtral-8x7b-32768");

    // Mistral models
    map.insert("mistral-large-latest", "mistral/mistral-large-latest");
    map.insert("mistral-medium-latest", "mistral/mistral-medium-latest");
    map.insert("mistral-small-latest", "mistral/mistral-small-latest");
    map.insert("codestral-latest", "mistral/codestral-latest");

    // Ollama models (local, no pricing)
    // These won't have pricing but we map them for consistency

    map
}

/// Normalize model name for pricing lookup
pub fn normalize_model_name(name: &str) -> String {
    let name = name.trim();

    // Check mapping first
    if let Some(mapped) = model_to_litellm().get(name) {
        return mapped.to_string();
    }

    // Remove provider prefixes
    for prefix in &[
        "openai/",
        "anthropic/",
        "google/",
        "gemini/",
        "deepseek/",
        "groq/",
        "mistral/",
        "ollama/",
        "openrouter/",
    ] {
        if let Some(stripped) = name.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }

    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_model_name() {
        assert_eq!(normalize_model_name("openai/gpt-4o"), "gpt-4o");
        assert_eq!(
            normalize_model_name("anthropic/claude-3-5-sonnet"),
            "claude-3-5-sonnet"
        );
        assert_eq!(
            normalize_model_name("gemini-flash-latest"),
            "gemini/gemini-flash-latest"
        );
    }
}
