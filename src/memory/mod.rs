//! Memory system for AI agents

mod backends;
mod buffer;
mod embeddings;
mod entry;
mod manager;
mod ranker;
mod store;

pub use backends::in_memory::InMemoryStore;
pub use backends::sqlite::SqliteStore;
pub use buffer::RingBuffer;
pub use embeddings::{EmbeddingProvider, MockEmbedding, OpenAIEmbedding};
pub use entry::{MemoryEntry, MemoryType};
pub use manager::{MemoryConfig, MemoryManager};
pub use ranker::{DecayConfig, MemoryRanker, RankingWeights};
pub use store::MemoryStore;
