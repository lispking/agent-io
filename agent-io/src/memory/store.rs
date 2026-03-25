//! Memory store trait

use async_trait::async_trait;

use super::entry::MemoryEntry;
use crate::Result;

/// Memory store trait for persisting and retrieving memories
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Add a memory entry, returns the entry ID
    async fn add(&self, entry: MemoryEntry) -> Result<String>;

    /// Search memories by text query
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>>;

    /// Search memories by embedding similarity
    async fn search_by_embedding(
        &self,
        embedding: &[f32],
        limit: usize,
        threshold: f32,
    ) -> Result<Vec<MemoryEntry>>;

    /// Get a memory by ID
    async fn get(&self, id: &str) -> Result<Option<MemoryEntry>>;

    /// Update a memory entry
    async fn update(&self, entry: MemoryEntry) -> Result<()>;

    /// Delete a memory by ID
    async fn delete(&self, id: &str) -> Result<()>;

    /// Clear all memories
    async fn clear(&self) -> Result<()>;

    /// Get total count of memories
    async fn count(&self) -> Result<usize>;

    /// Get all memory IDs
    async fn ids(&self) -> Result<Vec<String>>;
}
