//! Agent core implementation

mod builder;
mod compaction;
mod config;
mod events;
mod service;

pub use builder::AgentBuilder;
pub use compaction::*;
pub use config::{AgentConfig, DEFAULT_MAX_ITERATIONS, EphemeralConfig};
pub use events::*;
pub use llm_resilience::{CircuitBreaker, CircuitStatus};
pub use service::{Agent, default_llm_circuit_breaker};

/// Default threshold ratio for context compaction
pub const DEFAULT_COMPACTION_THRESHOLD: f32 = 0.80;
