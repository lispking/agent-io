//! Backends module

pub mod in_memory;
#[cfg(feature = "memory-lancedb")]
pub mod lancedb;
