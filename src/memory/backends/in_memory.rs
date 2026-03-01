//! In-memory memory store implementation

use async_trait::async_trait;
use std::collections::HashMap;
use tokio::sync::RwLock;

use crate::Result;
use crate::memory::entry::MemoryEntry;
use crate::memory::store::MemoryStore;

/// In-memory vector store for development and testing
pub struct InMemoryStore {
    memories: RwLock<HashMap<String, MemoryEntry>>,
}

impl InMemoryStore {
    /// Create a new in-memory store
    pub fn new() -> Self {
        Self {
            memories: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new in-memory store with pre-seeded memories
    pub fn with_memories(memories: Vec<MemoryEntry>) -> Self {
        let map: HashMap<String, MemoryEntry> =
            memories.into_iter().map(|m| (m.id.clone(), m)).collect();
        Self {
            memories: RwLock::new(map),
        }
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryStore for InMemoryStore {
    async fn add(&self, entry: MemoryEntry) -> Result<String> {
        let id = entry.id.clone();
        let mut memories = self.memories.write().await;
        memories.insert(id.clone(), entry);
        Ok(id)
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        let memories = self.memories.read().await;
        let query_lower = query.to_lowercase();

        let mut results: Vec<MemoryEntry> = memories
            .values()
            .filter(|m| m.content.to_lowercase().contains(&query_lower))
            .cloned()
            .collect();

        // Sort by relevance score
        results.sort_by(|a, b| {
            b.relevance_score()
                .partial_cmp(&a.relevance_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(limit);
        Ok(results)
    }

    async fn search_by_embedding(
        &self,
        embedding: &[f32],
        limit: usize,
        threshold: f32,
    ) -> Result<Vec<MemoryEntry>> {
        let memories = self.memories.read().await;

        let mut scored: Vec<(f32, MemoryEntry)> = memories
            .values()
            .filter_map(|m| {
                let emb = m.embedding.as_ref()?;
                let score = Self::cosine_similarity(embedding, emb);
                if score >= threshold {
                    Some((score, m.clone()))
                } else {
                    None
                }
            })
            .collect();

        // Sort by similarity score (descending)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored.into_iter().take(limit).map(|(_, m)| m).collect())
    }

    async fn get(&self, id: &str) -> Result<Option<MemoryEntry>> {
        let memories = self.memories.read().await;
        Ok(memories.get(id).cloned())
    }

    async fn update(&self, entry: MemoryEntry) -> Result<()> {
        let mut memories = self.memories.write().await;
        memories.insert(entry.id.clone(), entry);
        Ok(())
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let mut memories = self.memories.write().await;
        memories.remove(id);
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let mut memories = self.memories.write().await;
        memories.clear();
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let memories = self.memories.read().await;
        Ok(memories.len())
    }

    async fn ids(&self) -> Result<Vec<String>> {
        let memories = self.memories.read().await;
        Ok(memories.keys().cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_and_get() {
        let store = InMemoryStore::new();
        let entry = MemoryEntry::new("Test memory");

        let id = store.add(entry.clone()).await.unwrap();
        let retrieved = store.get(&id).await.unwrap();

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Test memory");
    }

    #[tokio::test]
    async fn test_search() {
        let store = InMemoryStore::new();

        store
            .add(MemoryEntry::new("Rust is a programming language"))
            .await
            .unwrap();
        store
            .add(MemoryEntry::new("Python is also a programming language"))
            .await
            .unwrap();
        store
            .add(MemoryEntry::new("The weather is nice today"))
            .await
            .unwrap();

        let results = store.search("programming", 10).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_search_by_embedding() {
        let store = InMemoryStore::new();

        let mut entry1 = MemoryEntry::new("Rust programming");
        entry1.embedding = Some(vec![1.0, 0.0, 0.0]);

        let mut entry2 = MemoryEntry::new("Python programming");
        entry2.embedding = Some(vec![0.0, 1.0, 0.0]);

        store.add(entry1).await.unwrap();
        store.add(entry2).await.unwrap();

        // Search with similar embedding
        let results = store
            .search_by_embedding(&[0.9, 0.1, 0.0], 10, 0.5)
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].content, "Rust programming");
    }

    #[tokio::test]
    async fn test_delete() {
        let store = InMemoryStore::new();
        let entry = MemoryEntry::new("Test");

        let id = store.add(entry).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);

        store.delete(&id).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
    }
}
