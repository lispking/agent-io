//! Model pricing data and mappings

use std::collections::HashMap;

/// Hardcoded pricing for common models (fallback when remote fetch fails)
pub fn get_default_pricing() -> HashMap<String, super::ModelPricing> {
    let mut pricing = HashMap::new();

    // OpenAI models
    pricing.insert(
        "gpt-4o".to_string(),
        super::ModelPricing {
            model: "gpt-4o".to_string(),
            input_cost_per_token: Some(0.0000025),
            output_cost_per_token: Some(0.00001),
            cache_read_input_token_cost: Some(0.00000125),
            cache_creation_input_token_cost: Some(0.000003125),
            max_tokens: Some(128000),
            max_input_tokens: Some(128000),
            max_output_tokens: Some(4096),
        },
    );

    pricing.insert(
        "gpt-4o-mini".to_string(),
        super::ModelPricing {
            model: "gpt-4o-mini".to_string(),
            input_cost_per_token: Some(0.00000015),
            output_cost_per_token: Some(0.0000006),
            cache_read_input_token_cost: Some(0.000000075),
            cache_creation_input_token_cost: Some(0.0000001875),
            max_tokens: Some(128000),
            max_input_tokens: Some(128000),
            max_output_tokens: Some(16384),
        },
    );

    pricing.insert(
        "gpt-4-turbo".to_string(),
        super::ModelPricing {
            model: "gpt-4-turbo".to_string(),
            input_cost_per_token: Some(0.00001),
            output_cost_per_token: Some(0.00003),
            cache_read_input_token_cost: None,
            cache_creation_input_token_cost: None,
            max_tokens: Some(128000),
            max_input_tokens: Some(128000),
            max_output_tokens: Some(4096),
        },
    );

    // Anthropic models
    pricing.insert(
        "claude-3-5-sonnet-20241022".to_string(),
        super::ModelPricing {
            model: "claude-3-5-sonnet-20241022".to_string(),
            input_cost_per_token: Some(0.000003),
            output_cost_per_token: Some(0.000015),
            cache_read_input_token_cost: Some(0.0000003),
            cache_creation_input_token_cost: Some(0.00000375),
            max_tokens: Some(200000),
            max_input_tokens: Some(200000),
            max_output_tokens: Some(8192),
        },
    );

    pricing.insert(
        "claude-3-opus-20240229".to_string(),
        super::ModelPricing {
            model: "claude-3-opus-20240229".to_string(),
            input_cost_per_token: Some(0.000015),
            output_cost_per_token: Some(0.000075),
            cache_read_input_token_cost: Some(0.0000015),
            cache_creation_input_token_cost: Some(0.00001875),
            max_tokens: Some(200000),
            max_input_tokens: Some(200000),
            max_output_tokens: Some(4096),
        },
    );

    // Google models
    pricing.insert(
        "gemini-2.0-flash".to_string(),
        super::ModelPricing {
            model: "gemini-2.0-flash".to_string(),
            input_cost_per_token: Some(0.0000001),
            output_cost_per_token: Some(0.0000004),
            cache_read_input_token_cost: Some(0.000000025),
            cache_creation_input_token_cost: None,
            max_tokens: Some(1048576),
            max_input_tokens: Some(1048576),
            max_output_tokens: Some(8192),
        },
    );

    pricing.insert(
        "gemini-1.5-pro".to_string(),
        super::ModelPricing {
            model: "gemini-1.5-pro".to_string(),
            input_cost_per_token: Some(0.00000125),
            output_cost_per_token: Some(0.000005),
            cache_read_input_token_cost: Some(0.0000003125),
            cache_creation_input_token_cost: None,
            max_tokens: Some(2097152),
            max_input_tokens: Some(2097152),
            max_output_tokens: Some(8192),
        },
    );

    pricing
}
