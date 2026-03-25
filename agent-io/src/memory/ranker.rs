//! Memory ranking and importance decay

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::entry::MemoryEntry;

/// Ranking weights for memory relevance scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingWeights {
    /// Weight for embedding similarity (0.0 - 1.0)
    pub similarity: f32,
    /// Weight for importance score (0.0 - 1.0)
    pub importance: f32,
    /// Weight for recency (0.0 - 1.0)
    pub recency: f32,
    /// Weight for access frequency (0.0 - 1.0)
    pub frequency: f32,
}

impl Default for RankingWeights {
    fn default() -> Self {
        Self {
            similarity: 0.4,
            importance: 0.25,
            recency: 0.2,
            frequency: 0.15,
        }
    }
}

/// Memory ranker for scoring and sorting memories
pub struct MemoryRanker {
    weights: RankingWeights,
    recency_half_life_hours: f32,
}

impl MemoryRanker {
    /// Create a new memory ranker with default weights
    pub fn new() -> Self {
        Self {
            weights: RankingWeights::default(),
            recency_half_life_hours: 24.0 * 7.0, // 1 week
        }
    }

    /// Create a ranker with custom weights
    pub fn with_weights(weights: RankingWeights) -> Self {
        Self {
            weights,
            recency_half_life_hours: 24.0 * 7.0,
        }
    }

    /// Set recency half-life in hours
    pub fn with_recency_half_life(mut self, hours: f32) -> Self {
        self.recency_half_life_hours = hours;
        self
    }

    /// Calculate recency score (exponential decay)
    fn recency_score(&self, created_at: DateTime<Utc>) -> f32 {
        let age_hours = (Utc::now() - created_at).num_hours() as f32;
        (-age_hours / self.recency_half_life_hours).exp()
    }

    /// Calculate frequency score (logarithmic)
    fn frequency_score(&self, access_count: u32) -> f32 {
        if access_count == 0 {
            0.0
        } else {
            (1.0 + access_count as f32).ln() / 10.0 // Normalize to roughly 0-1
        }
    }

    /// Calculate composite relevance score
    pub fn score(&self, entry: &MemoryEntry, query_embedding: &[f32]) -> f32 {
        // Similarity score
        let similarity = if let Some(ref embedding) = entry.embedding {
            cosine_similarity(query_embedding, embedding)
        } else {
            0.0
        };

        // Recency score
        let recency = self.recency_score(entry.created_at);

        // Frequency score
        let frequency = self.frequency_score(entry.access_count);

        // Weighted combination
        self.weights.similarity * similarity
            + self.weights.importance * entry.importance
            + self.weights.recency * recency
            + self.weights.frequency * frequency
    }

    /// Rank memories by relevance to query
    pub fn rank(&self, query_embedding: &[f32], memories: Vec<MemoryEntry>) -> Vec<MemoryEntry> {
        let mut scored: Vec<(f32, MemoryEntry)> = memories
            .into_iter()
            .map(|m| (self.score(&m, query_embedding), m))
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().map(|(_, m)| m).collect()
    }

    /// Rank memories and return with scores
    pub fn rank_with_scores(
        &self,
        query_embedding: &[f32],
        memories: Vec<MemoryEntry>,
    ) -> Vec<(f32, MemoryEntry)> {
        let mut scored: Vec<(f32, MemoryEntry)> = memories
            .into_iter()
            .map(|m| (self.score(&m, query_embedding), m))
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }
}

impl Default for MemoryRanker {
    fn default() -> Self {
        Self::new()
    }
}

/// Importance decay configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayConfig {
    /// Decay rate per day (0.0 - 1.0)
    pub daily_rate: f32,
    /// Minimum importance threshold (memories below this are candidates for removal)
    pub min_threshold: f32,
    /// Age in days before decay starts
    pub grace_period_days: u32,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            daily_rate: 0.01,
            min_threshold: 0.1,
            grace_period_days: 7,
        }
    }
}

impl DecayConfig {
    /// Create new decay config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set daily decay rate
    pub fn with_rate(mut self, rate: f32) -> Self {
        self.daily_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set minimum threshold
    pub fn with_min_threshold(mut self, threshold: f32) -> Self {
        self.min_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set grace period
    pub fn with_grace_period(mut self, days: u32) -> Self {
        self.grace_period_days = days;
        self
    }

    /// Apply decay to a memory entry
    pub fn apply(&self, entry: &mut MemoryEntry) -> bool {
        let age_days = (Utc::now() - entry.created_at).num_days() as u32;

        // Skip if in grace period
        if age_days < self.grace_period_days {
            return false;
        }

        // Apply exponential decay
        let days_since_grace = age_days - self.grace_period_days;
        let decay_factor = (1.0 - self.daily_rate).powi(days_since_grace as i32);
        entry.importance *= decay_factor;

        // Check if below threshold
        entry.importance < self.min_threshold
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
        (dot / (norm_a * norm_b)).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ranking_weights() {
        let weights = RankingWeights::default();
        assert!(
            (weights.similarity + weights.importance + weights.recency + weights.frequency - 1.0)
                .abs()
                < 0.01
        );
    }

    #[test]
    fn test_recency_score() {
        let ranker = MemoryRanker::new();

        // Recent memory should have high score
        let recent = Utc::now() - chrono::Duration::hours(1);
        assert!(ranker.recency_score(recent) > 0.9);

        // Old memory should have low score
        let old = Utc::now() - chrono::Duration::hours(24 * 30);
        assert!(ranker.recency_score(old) < 0.5);
    }

    #[test]
    fn test_frequency_score() {
        let ranker = MemoryRanker::new();

        assert_eq!(ranker.frequency_score(0), 0.0);
        assert!(ranker.frequency_score(10) > ranker.frequency_score(1));
    }

    #[test]
    fn test_rank_memories() {
        let ranker = MemoryRanker::new();

        let mut entry1 = MemoryEntry::new("First");
        entry1.embedding = Some(vec![1.0, 0.0, 0.0]);
        entry1.importance = 0.9;

        let mut entry2 = MemoryEntry::new("Second");
        entry2.embedding = Some(vec![0.0, 1.0, 0.0]);
        entry2.importance = 0.5;

        let ranked = ranker.rank(&[0.9, 0.1, 0.0], vec![entry1.clone(), entry2.clone()]);
        assert_eq!(ranked[0].content, "First");
    }

    #[test]
    fn test_decay_config() {
        let config = DecayConfig::new()
            .with_rate(0.1)
            .with_min_threshold(0.2)
            .with_grace_period(3);

        assert_eq!(config.daily_rate, 0.1);
        assert_eq!(config.min_threshold, 0.2);
        assert_eq!(config.grace_period_days, 3);
    }

    #[test]
    fn test_decay_apply() {
        let config = DecayConfig::default();
        let mut entry = MemoryEntry::new("Test");
        entry.importance = 0.5;
        entry.created_at = Utc::now() - chrono::Duration::days(10);

        let _below_threshold = config.apply(&mut entry);
        assert!(entry.importance < 0.5);
    }
}
