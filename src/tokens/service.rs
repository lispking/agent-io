//! Token usage tracking and cost calculation

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use crate::tokens::mappings::normalize_model_name;

/// Default pricing URL (LiteLLM model prices)
pub const PRICING_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

/// Cache duration for pricing data
pub const CACHE_DURATION: Duration = Duration::hours(24);

/// Cache file name
pub const CACHE_FILE_NAME: &str = "agent-io-pricing-cache.json";

/// Model pricing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    pub model: String,
    pub input_cost_per_token: Option<f64>,
    pub output_cost_per_token: Option<f64>,
    pub cache_read_input_token_cost: Option<f64>,
    pub cache_creation_input_token_cost: Option<f64>,
    pub max_tokens: Option<u64>,
    pub max_input_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
}

impl ModelPricing {
    /// Calculate cost for given token usage
    pub fn calculate_cost(
        &self,
        input_tokens: u64,
        output_tokens: u64,
        cached_tokens: u64,
        cache_creation_tokens: u64,
    ) -> TokenCostCalculated {
        let mut prompt_cost = 0.0;
        let mut completion_cost = 0.0;

        // Input tokens cost
        if let Some(cost) = self.input_cost_per_token {
            prompt_cost += (input_tokens as f64) * cost;
        }

        // Cached tokens cost (usually cheaper)
        if let Some(cost) = self.cache_read_input_token_cost {
            prompt_cost -= (input_tokens as f64) * (self.input_cost_per_token.unwrap_or(0.0));
            prompt_cost += (cached_tokens as f64) * cost;
        }

        // Cache creation cost
        if let Some(cost) = self.cache_creation_input_token_cost {
            prompt_cost += (cache_creation_tokens as f64) * cost;
        }

        // Output tokens cost
        if let Some(cost) = self.output_cost_per_token {
            completion_cost = (output_tokens as f64) * cost;
        }

        TokenCostCalculated {
            prompt_cost,
            completion_cost,
            total_cost: prompt_cost + completion_cost,
        }
    }
}

/// Calculated token cost
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenCostCalculated {
    pub prompt_cost: f64,
    pub completion_cost: f64,
    pub total_cost: f64,
}

/// Per-model usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelUsageStats {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub prompt_cost: f64,
    pub completion_cost: f64,
    pub total_cost: f64,
    pub calls: u64,
}

/// Usage summary for the session
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageSummary {
    pub total_prompt_tokens: u64,
    pub total_prompt_cost: f64,
    pub total_completion_tokens: u64,
    pub total_completion_cost: f64,
    pub total_tokens: u64,
    pub total_cost: f64,
    pub by_model: HashMap<String, ModelUsageStats>,
}

impl UsageSummary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add usage from a completion
    pub fn add(&mut self, model: &str, usage: &crate::llm::Usage, pricing: Option<&ModelPricing>) {
        self.total_prompt_tokens += usage.prompt_tokens;
        self.total_completion_tokens += usage.completion_tokens;
        self.total_tokens += usage.total_tokens;

        let model_stats = self.by_model.entry(model.to_string()).or_default();
        model_stats.prompt_tokens += usage.prompt_tokens;
        model_stats.completion_tokens += usage.completion_tokens;
        model_stats.total_tokens += usage.total_tokens;
        model_stats.calls += 1;

        if let Some(pricing) = pricing {
            let cost = pricing.calculate_cost(
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.prompt_cached_tokens.unwrap_or(0),
                usage.prompt_cache_creation_tokens.unwrap_or(0),
            );

            self.total_prompt_cost += cost.prompt_cost;
            self.total_completion_cost += cost.completion_cost;
            self.total_cost += cost.total_cost;

            model_stats.prompt_cost += cost.prompt_cost;
            model_stats.completion_cost += cost.completion_cost;
            model_stats.total_cost += cost.total_cost;
        }
    }

    /// Merge another usage summary
    pub fn merge(&mut self, other: &UsageSummary) {
        self.total_prompt_tokens += other.total_prompt_tokens;
        self.total_prompt_cost += other.total_prompt_cost;
        self.total_completion_tokens += other.total_completion_tokens;
        self.total_completion_cost += other.total_completion_cost;
        self.total_tokens += other.total_tokens;
        self.total_cost += other.total_cost;

        for (model, stats) in &other.by_model {
            let entry = self.by_model.entry(model.clone()).or_default();
            entry.prompt_tokens += stats.prompt_tokens;
            entry.completion_tokens += stats.completion_tokens;
            entry.total_tokens += stats.total_tokens;
            entry.prompt_cost += stats.prompt_cost;
            entry.completion_cost += stats.completion_cost;
            entry.total_cost += stats.total_cost;
            entry.calls += stats.calls;
        }
    }
}

/// Cached pricing data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedPricing {
    pricing: HashMap<String, ModelPricing>,
    last_update: DateTime<Utc>,
}

/// Token cost service
pub struct TokenCost {
    pricing: HashMap<String, ModelPricing>,
    last_update: Option<DateTime<Utc>>,
    cache_path: Option<PathBuf>,
}

impl TokenCost {
    pub fn new() -> Self {
        Self {
            pricing: HashMap::new(),
            last_update: None,
            cache_path: Self::get_cache_path(),
        }
    }

    /// Get the cache file path
    fn get_cache_path() -> Option<PathBuf> {
        // Try XDG cache directory first
        if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
            let cache_dir = PathBuf::from(xdg_cache);
            let _ = fs::create_dir_all(&cache_dir);
            return Some(cache_dir.join(CACHE_FILE_NAME));
        }

        // Fallback to home directory
        if let Some(home) = dirs::home_dir() {
            let cache_dir = home.join(".cache");
            let _ = fs::create_dir_all(&cache_dir);
            return Some(cache_dir.join(CACHE_FILE_NAME));
        }

        None
    }

    /// Load pricing from local cache
    pub fn load_cache(&mut self) -> Result<(), String> {
        let cache_path = match &self.cache_path {
            Some(p) => p,
            None => return Err("No cache path available".into()),
        };

        if !cache_path.exists() {
            return Err("Cache file does not exist".into());
        }

        let content =
            fs::read_to_string(cache_path).map_err(|e| format!("Failed to read cache: {}", e))?;

        let cached: CachedPricing =
            serde_json::from_str(&content).map_err(|e| format!("Failed to parse cache: {}", e))?;

        self.pricing = cached.pricing;
        self.last_update = Some(cached.last_update);

        Ok(())
    }

    /// Save pricing to local cache
    fn save_cache(&self) -> Result<(), String> {
        let cache_path = match &self.cache_path {
            Some(p) => p,
            None => return Ok(()),
        };

        let cached = CachedPricing {
            pricing: self.pricing.clone(),
            last_update: self.last_update.unwrap_or_else(Utc::now),
        };

        let content = serde_json::to_string_pretty(&cached)
            .map_err(|e| format!("Failed to serialize cache: {}", e))?;

        fs::write(cache_path, content).map_err(|e| format!("Failed to write cache: {}", e))?;

        Ok(())
    }

    /// Fetch pricing data from remote or load from cache
    pub async fn fetch_pricing(&mut self) -> Result<(), String> {
        // Try to load from cache first
        if self.load_cache().is_ok() && !self.needs_refresh() {
            return Ok(());
        }

        // Fetch from remote
        let response = reqwest::get(PRICING_URL)
            .await
            .map_err(|e| format!("Failed to fetch pricing: {}", e))?;

        if !response.status().is_success() {
            // If we have cached data, use it even if expired
            if self.last_update.is_some() {
                return Ok(());
            }
            return Err(format!(
                "Failed to fetch pricing: HTTP {}",
                response.status()
            ));
        }

        let pricing_data: HashMap<String, ModelPricing> = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse pricing: {}", e))?;

        self.pricing = pricing_data;
        self.last_update = Some(Utc::now());

        // Save to cache
        let _ = self.save_cache();

        Ok(())
    }

    /// Check if pricing needs refresh
    pub fn needs_refresh(&self) -> bool {
        match self.last_update {
            None => true,
            Some(last) => {
                let elapsed = Utc::now() - last;
                elapsed > CACHE_DURATION
            }
        }
    }

    /// Get pricing for a model
    pub fn get_model_pricing(&self, model_name: &str) -> Option<&ModelPricing> {
        // Try exact match first
        if let Some(pricing) = self.pricing.get(model_name) {
            return Some(pricing);
        }

        // Try normalized model name
        let normalized = normalize_model_name(model_name);

        // Try with the normalized name
        if let Some(pricing) = self.pricing.get(&normalized) {
            return Some(pricing);
        }

        // Try without provider prefix
        self.pricing.get(&normalized.replace('/', "-"))
    }

    /// Calculate cost for a completion
    pub fn calculate_cost(
        &self,
        model: &str,
        usage: &crate::llm::Usage,
    ) -> Option<TokenCostCalculated> {
        let pricing = self.get_model_pricing(model)?;
        Some(pricing.calculate_cost(
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.prompt_cached_tokens.unwrap_or(0),
            usage.prompt_cache_creation_tokens.unwrap_or(0),
        ))
    }
}

impl Default for TokenCost {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_pricing() {
        let pricing = ModelPricing {
            model: "gpt-4o".to_string(),
            input_cost_per_token: Some(0.0000025),
            output_cost_per_token: Some(0.00001),
            cache_read_input_token_cost: Some(0.00000125),
            cache_creation_input_token_cost: Some(0.000003125),
            max_tokens: Some(128000),
            max_input_tokens: Some(128000),
            max_output_tokens: Some(4096),
        };

        let cost = pricing.calculate_cost(1000, 500, 200, 100);

        assert!(cost.prompt_cost > 0.0);
        assert!(cost.completion_cost > 0.0);
        assert!(cost.total_cost > 0.0);
    }

    #[test]
    fn test_usage_summary() {
        let mut summary = UsageSummary::new();
        let usage = crate::llm::Usage::new(100, 50);

        summary.add("gpt-4o", &usage, None);

        assert_eq!(summary.total_prompt_tokens, 100);
        assert_eq!(summary.total_completion_tokens, 50);
        assert_eq!(summary.total_tokens, 150);
    }
}
