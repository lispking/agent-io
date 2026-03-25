//! Memory entry types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Memory type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    /// Recent conversation (short-term)
    #[default]
    ShortTerm,
    /// Persistent knowledge (long-term)
    LongTerm,
    /// Specific events/experiences
    Episodic,
    /// Facts and concepts
    Semantic,
}

/// Memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier
    pub id: String,
    /// Memory content
    pub content: String,
    /// Embedding vector (for similarity search)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Memory type
    #[serde(default)]
    pub memory_type: MemoryType,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last access timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_accessed: Option<DateTime<Utc>>,
    /// Importance score (0.0 - 1.0)
    #[serde(default = "default_importance")]
    pub importance: f32,
    /// Access count (for recency weighting)
    #[serde(default)]
    pub access_count: u32,
}

fn default_importance() -> f32 {
    0.5
}

impl MemoryEntry {
    /// Create a new memory entry
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.into(),
            embedding: None,
            memory_type: MemoryType::default(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            last_accessed: None,
            importance: 0.5,
            access_count: 0,
        }
    }

    /// Set memory type
    pub fn with_type(mut self, memory_type: MemoryType) -> Self {
        self.memory_type = memory_type;
        self
    }

    /// Set embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set importance
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Record access (updates access count and timestamp)
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Some(Utc::now());
    }

    /// Calculate relevance score based on importance, recency, and access frequency
    pub fn relevance_score(&self) -> f32 {
        let age_hours = (Utc::now() - self.created_at).num_hours() as f32;
        let recency_factor = (-age_hours / 24.0 / 7.0).exp(); // Decay over a week

        let access_factor = 1.0 + (self.access_count as f32).ln().max(0.0) * 0.1;

        self.importance * recency_factor * access_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_entry_creation() {
        let entry = MemoryEntry::new("Test memory content");
        assert_eq!(entry.content, "Test memory content");
        assert_eq!(entry.memory_type, MemoryType::ShortTerm);
        assert_eq!(entry.importance, 0.5);
    }

    #[test]
    fn test_memory_entry_builder() {
        let entry = MemoryEntry::new("Test")
            .with_type(MemoryType::LongTerm)
            .with_importance(0.9)
            .with_metadata("source", Value::String("user".to_string()));

        assert_eq!(entry.memory_type, MemoryType::LongTerm);
        assert_eq!(entry.importance, 0.9);
        assert!(entry.metadata.contains_key("source"));
    }

    #[test]
    fn test_record_access() {
        let mut entry = MemoryEntry::new("Test");
        assert_eq!(entry.access_count, 0);

        entry.record_access();
        assert_eq!(entry.access_count, 1);
        assert!(entry.last_accessed.is_some());
    }
}
