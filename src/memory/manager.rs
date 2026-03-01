//! Memory manager - core orchestration

use std::sync::Arc;

use super::buffer::RingBuffer;
use super::embeddings::EmbeddingProvider;
use super::entry::{MemoryEntry, MemoryType};
use super::store::MemoryStore;
use crate::Result;

/// Memory configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Short-term buffer size (number of messages)
    pub short_term_size: usize,
    /// Enable long-term memory storage
    pub enable_long_term: bool,
    /// Maximum memories to retrieve
    pub retrieval_limit: usize,
    /// Similarity threshold for retrieval (0.0 - 1.0)
    pub relevance_threshold: f32,
    /// Maximum tokens for context building
    pub max_context_tokens: u64,
    /// Importance decay rate per day
    pub importance_decay: f32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            short_term_size: 20,
            enable_long_term: true,
            retrieval_limit: 5,
            relevance_threshold: 0.7,
            max_context_tokens: 2000,
            importance_decay: 0.95,
        }
    }
}

/// Memory manager orchestrates short-term and long-term memory
pub struct MemoryManager {
    config: MemoryConfig,
    short_term: RingBuffer<MemoryEntry>,
    store: Arc<dyn MemoryStore>,
    embedder: Arc<dyn EmbeddingProvider>,
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new(
        config: MemoryConfig,
        store: Arc<dyn MemoryStore>,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self {
            short_term: RingBuffer::new(config.short_term_size),
            store,
            embedder,
            config,
        }
    }

    /// Create with default configuration
    pub fn with_defaults(
        store: Arc<dyn MemoryStore>,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self::new(MemoryConfig::default(), store, embedder)
    }

    /// Store a memory
    pub async fn remember(&mut self, content: &str, memory_type: MemoryType) -> Result<String> {
        let embedding = self.embedder.embed(content).await?;

        let entry = MemoryEntry::new(content)
            .with_type(memory_type)
            .with_embedding(embedding);

        match memory_type {
            MemoryType::ShortTerm => {
                self.short_term.push(entry.clone());
                Ok(entry.id)
            }
            _ => {
                if self.config.enable_long_term {
                    self.store.add(entry).await
                } else {
                    self.short_term.push(entry.clone());
                    Ok(entry.id)
                }
            }
        }
    }

    /// Store a memory with importance
    pub async fn remember_important(
        &mut self,
        content: &str,
        memory_type: MemoryType,
        importance: f32,
    ) -> Result<String> {
        let embedding = self.embedder.embed(content).await?;

        let entry = MemoryEntry::new(content)
            .with_type(memory_type)
            .with_embedding(embedding)
            .with_importance(importance);

        match memory_type {
            MemoryType::ShortTerm => {
                self.short_term.push(entry.clone());
                Ok(entry.id)
            }
            _ => {
                if self.config.enable_long_term {
                    self.store.add(entry).await
                } else {
                    self.short_term.push(entry.clone());
                    Ok(entry.id)
                }
            }
        }
    }

    /// Recall relevant memories for a query
    pub async fn recall(&self, query: &str) -> Result<Vec<MemoryEntry>> {
        let query_embedding = self.embedder.embed(query).await?;

        // Search long-term memory
        let mut memories = if self.config.enable_long_term {
            self.store
                .search_by_embedding(
                    &query_embedding,
                    self.config.retrieval_limit,
                    self.config.relevance_threshold,
                )
                .await?
        } else {
            Vec::new()
        };

        // Add short-term memories
        for entry in self.short_term.iter_recent() {
            if let Some(ref embedding) = entry.embedding {
                let similarity = Self::cosine_similarity(&query_embedding, embedding);
                if similarity >= self.config.relevance_threshold {
                    memories.push(entry.clone());
                }
            }
        }

        // Sort by relevance score
        memories.sort_by(|a, b| {
            b.relevance_score()
                .partial_cmp(&a.relevance_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        memories.truncate(self.config.retrieval_limit);

        Ok(memories)
    }

    /// Build context string from memories
    pub fn build_context(&self, memories: &[MemoryEntry]) -> String {
        let mut context = String::new();
        let mut token_count = 0;

        for memory in memories {
            let tokens = memory.content.len() / 4; // Approximate token count
            if token_count + tokens > self.config.max_context_tokens as usize {
                break;
            }
            context.push_str(&memory.content);
            context.push_str("\n\n");
            token_count += tokens;
        }

        context
    }

    /// Recall and build context in one step
    pub async fn recall_context(&self, query: &str) -> Result<String> {
        let memories = self.recall(query).await?;
        Ok(self.build_context(&memories))
    }

    /// Get short-term memory buffer
    pub fn short_term(&self) -> &RingBuffer<MemoryEntry> {
        &self.short_term
    }

    /// Get memory store
    pub fn store(&self) -> &Arc<dyn MemoryStore> {
        &self.store
    }

    /// Clear short-term memory
    pub fn clear_short_term(&mut self) {
        self.short_term.clear();
    }

    /// Clear all memories (including long-term)
    pub async fn clear_all(&mut self) -> Result<()> {
        self.short_term.clear();
        self.store.clear().await
    }

    /// Get total memory count
    pub async fn count(&self) -> Result<usize> {
        let long_term_count = self.store.count().await?;
        Ok(self.short_term.len() + long_term_count)
    }

    /// Calculate cosine similarity
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::InMemoryStore;
    use crate::memory::embeddings::MockEmbedding;

    #[tokio::test]
    async fn test_remember_and_recall() {
        let store = Arc::new(InMemoryStore::new());
        let embedder = Arc::new(MockEmbedding::new(128));
        let mut manager = MemoryManager::with_defaults(store, embedder);

        // Store a memory
        let id = manager
            .remember("I like Rust programming", MemoryType::LongTerm)
            .await
            .unwrap();
        assert!(!id.is_empty());

        // Recall should find it
        let memories = manager.recall("programming").await.unwrap();
        assert!(!memories.is_empty());
    }

    #[tokio::test]
    async fn test_short_term_memory() {
        let store = Arc::new(InMemoryStore::new());
        let embedder = Arc::new(MockEmbedding::new(128));
        let mut manager = MemoryManager::with_defaults(store, embedder);

        // Store short-term memory
        manager
            .remember("Temporary thought", MemoryType::ShortTerm)
            .await
            .unwrap();

        assert_eq!(manager.short_term().len(), 1);
    }

    #[tokio::test]
    async fn test_build_context() {
        let store = Arc::new(InMemoryStore::new());
        let embedder = Arc::new(MockEmbedding::new(128));
        let manager = MemoryManager::with_defaults(store, embedder);

        let memories = vec![
            MemoryEntry::new("First memory"),
            MemoryEntry::new("Second memory"),
        ];

        let context = manager.build_context(&memories);
        assert!(context.contains("First memory"));
        assert!(context.contains("Second memory"));
    }

    #[tokio::test]
    async fn test_clear() {
        let store = Arc::new(InMemoryStore::new());
        let embedder = Arc::new(MockEmbedding::new(128));
        let mut manager = MemoryManager::with_defaults(store, embedder);

        manager
            .remember("Test", MemoryType::ShortTerm)
            .await
            .unwrap();

        manager.clear_short_term();
        assert!(manager.short_term().is_empty());
    }
}
