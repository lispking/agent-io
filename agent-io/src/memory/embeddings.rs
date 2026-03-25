//! Embedding provider trait and implementations

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use crate::Result;

/// Embedding provider trait
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embedding for a single text
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for multiple texts
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Get the embedding dimension
    fn dimension(&self) -> usize;
}

/// OpenAI embedding provider
pub struct OpenAIEmbedding {
    client: Client,
    api_key: String,
    model: String,
    dimension: usize,
}

impl OpenAIEmbedding {
    /// Create a new OpenAI embedding provider
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: "text-embedding-3-small".to_string(),
            dimension: 1536,
        }
    }

    /// Create from environment variable
    pub fn from_env() -> crate::Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| crate::Error::Config("OPENAI_API_KEY not set".into()))?;
        Ok(Self::new(api_key))
    }

    /// Use a specific model
    pub fn with_model(mut self, model: impl Into<String>, dimension: usize) -> Self {
        self.model = model.into();
        self.dimension = dimension;
        self
    }

    /// Use text-embedding-3-large model
    pub fn large() -> crate::Result<Self> {
        Ok(Self::from_env()?.with_model("text-embedding-3-large", 3072))
    }

    /// Use text-embedding-ada-002 model
    pub fn ada() -> crate::Result<Self> {
        Ok(Self::from_env()?.with_model("text-embedding-ada-002", 1536))
    }
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingProvider for OpenAIEmbedding {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| crate::Error::Agent("No embedding returned".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": self.model,
                "input": texts,
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(crate::Error::Agent(format!(
                "OpenAI embedding error ({}): {}",
                status, body
            )));
        }

        let data: EmbeddingResponse = response.json().await?;
        Ok(data.data.into_iter().map(|e| e.embedding).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Mock embedding provider for testing
#[allow(dead_code)]
pub struct MockEmbedding {
    dimension: usize,
}

#[allow(dead_code)]
impl MockEmbedding {
    /// Create a new mock embedding provider
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Default for MockEmbedding {
    fn default() -> Self {
        Self::new(384)
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbedding {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        // Return a deterministic embedding based on text length
        Ok(vec![0.1; self.dimension])
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.1; self.dimension]).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_embedding() {
        let embedder = MockEmbedding::new(128);

        let embedding = embedder.embed("test").await.unwrap();
        assert_eq!(embedding.len(), 128);

        let batch = embedder.embed_batch(&["a", "b", "c"]).await.unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].len(), 128);
    }
}
