//! Observability module for agent_io with tracing integration.
//!
//! This module provides:
//! - `observe` macro for tracing functions
//! - Span helpers for manual instrumentation
//!
//! Uses the `tracing` crate for structured logging and tracing.
//!
//! # Example
//!
//! ```rust,ignore
//! use agent_io::observe;
//!
//! #[observe(name = "my_function")]
//! async fn my_function() -> Result<(), Box<dyn std::error::Error>> {
//!     Ok(())
//! }
//! ```

use tracing::{Span, info_span};
use tracing_subscriber::EnvFilter;

/// Span types for categorization
#[derive(Debug, Clone, Copy)]
pub enum SpanType {
    Default,
    Llm,
    Tool,
}

impl SpanType {
    pub fn as_str(&self) -> &'static str {
        match self {
            SpanType::Default => "DEFAULT",
            SpanType::Llm => "LLM",
            SpanType::Tool => "TOOL",
        }
    }
}

/// Extension trait for spans
pub trait SpanExt {
    /// Set the span output
    fn set_output(&self, output: &str);

    /// Mark the span as having an error
    fn set_error(&self, error: &str);
}

impl SpanExt for Span {
    fn set_output(&self, output: &str) {
        self.record("output", output);
    }

    fn set_error(&self, error: &str) {
        self.record("error", true);
        self.record("error.message", error);
    }
}

/// Check if tracing is enabled
pub fn is_tracing_enabled() -> bool {
    tracing::dispatcher::has_been_set()
}

/// Get observability status
pub fn get_observability_status() -> ObservabilityStatus {
    ObservabilityStatus {
        tracing_enabled: is_tracing_enabled(),
    }
}

/// Observability status information
#[derive(Debug, Clone)]
pub struct ObservabilityStatus {
    pub tracing_enabled: bool,
}

/// Macro to create an observed function
///
/// # Example
///
/// ```rust,ignore
/// #[observe(name = "my_function")]
/// async fn my_function() -> Result<(), Error> {
///     // function body
///     Ok(())
/// }
/// ```
#[macro_export]
macro_rules! observe {
    (name = $name:expr, $item:item) => {
        #[tracing::instrument(skip_all, name = $name)]
        $item
    };
    ($item:item) => {
        #[tracing::instrument(skip_all)]
        $item
    };
}

/// Macro for debug-only observation
#[macro_export]
macro_rules! observe_debug {
    (name = $name:expr, $item:item) => {
        #[cfg(debug_assertions)]
        #[tracing::instrument(skip_all, name = $name)]
        $item
        #[cfg(not(debug_assertions))]
        $item
    };
    ($item:item) => {
        #[cfg(debug_assertions)]
        #[tracing::instrument(skip_all)]
        $item
        #[cfg(not(debug_assertions))]
        $item
    };
}

/// Initialize default tracing subscriber for development
pub fn init_default_subscriber() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .try_init();
}

/// Create a span for LLM invocation
pub fn llm_span(model: &str, provider: &str) -> Span {
    info_span!(
        target: "agent_io::llm",
        "llm_invoke",
        model = %model,
        provider = %provider
    )
}

/// Create a span for tool execution
pub fn tool_span(name: &str) -> Span {
    info_span!(
        target: "agent_io::tool",
        "tool_execute",
        tool_name = %name
    )
}

/// Create a span for agent step
pub fn agent_span(step: usize) -> Span {
    info_span!(
        target: "agent_io::agent",
        "agent_step",
        step = step
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_types() {
        assert_eq!(SpanType::Default.as_str(), "DEFAULT");
        assert_eq!(SpanType::Llm.as_str(), "LLM");
        assert_eq!(SpanType::Tool.as_str(), "TOOL");
    }
}
