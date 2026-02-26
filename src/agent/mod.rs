//! Agent core implementation

mod compaction;
mod events;
mod service;

pub use compaction::*;
pub use events::*;
pub use service::*;

/// Maximum iterations for agent loop
pub const DEFAULT_MAX_ITERATIONS: usize = 200;

/// Default threshold ratio for context compaction
pub const DEFAULT_COMPACTION_THRESHOLD: f32 = 0.80;
