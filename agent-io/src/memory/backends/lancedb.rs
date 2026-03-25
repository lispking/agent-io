//! LanceDB-based memory store for persistent vector storage

use arrow::array::{Array, FixedSizeListArray, Float32Array, Int64Array, StringArray, UInt32Array};
use arrow::record_batch::RecordBatch;
use arrow_array::RecordBatchIterator;
use arrow_array::types::Float32Type;
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{Table, connect};
use std::path::PathBuf;
use std::sync::Arc;

use crate::Result;
use crate::memory::entry::{MemoryEntry, MemoryType};
use crate::memory::store::MemoryStore;

const TABLE_NAME: &str = "memories";
const EMBEDDING_DIM: usize = 1536; // OpenAI embedding dimension

/// LanceDB memory store for persistent vector storage with FTS support
pub struct LanceDbStore {
    table: Arc<Table>,
}

impl LanceDbStore {
    /// Create a new LanceDB store with an in-memory database
    pub async fn new() -> Result<Self> {
        Self::open_uri("memory://agent_io_memories").await
    }

    /// Create a new LanceDB store with a file database
    pub async fn open<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let path = path.into();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| crate::Error::Agent(format!("Failed to create directory: {}", e)))?;
        }

        let uri = path.to_string_lossy().to_string();
        Self::open_uri(&uri).await
    }

    async fn open_uri(uri: &str) -> Result<Self> {
        let db = connect(uri)
            .execute()
            .await
            .map_err(|e| crate::Error::Agent(format!("Failed to connect to LanceDB: {}", e)))?;

        let table_names = db
            .table_names()
            .execute()
            .await
            .map_err(|e| crate::Error::Agent(format!("Failed to list tables: {}", e)))?;

        let table = if table_names.contains(&TABLE_NAME.to_string()) {
            db.open_table(TABLE_NAME)
                .execute()
                .await
                .map_err(|e| crate::Error::Agent(format!("Failed to open table: {}", e)))?
        } else {
            // Create an empty table with schema
            let schema = Self::schema();
            db.create_empty_table(TABLE_NAME, schema)
                .execute()
                .await
                .map_err(|e| crate::Error::Agent(format!("Failed to create table: {}", e)))?
        };

        Ok(Self {
            table: Arc::new(table),
        })
    }

    /// Get the table schema
    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    EMBEDDING_DIM as i32,
                ),
                true,
            ),
            Field::new("memory_type", DataType::Utf8, false),
            Field::new("metadata", DataType::Utf8, true),
            Field::new("created_at", DataType::Int64, false),
            Field::new("last_accessed", DataType::Int64, true),
            Field::new("importance", DataType::Float32, false),
            Field::new("access_count", DataType::UInt32, false),
        ]))
    }

    /// Convert memory type to string
    fn memory_type_to_string(t: &MemoryType) -> &'static str {
        match t {
            MemoryType::ShortTerm => "short_term",
            MemoryType::LongTerm => "long_term",
            MemoryType::Episodic => "episodic",
            MemoryType::Semantic => "semantic",
        }
    }

    /// Convert string to memory type
    fn string_to_memory_type(s: &str) -> MemoryType {
        match s {
            "long_term" => MemoryType::LongTerm,
            "episodic" => MemoryType::Episodic,
            "semantic" => MemoryType::Semantic,
            _ => MemoryType::ShortTerm,
        }
    }

    /// Convert MemoryEntry to RecordBatch
    fn entry_to_batch(entry: &MemoryEntry) -> Result<RecordBatch> {
        let schema = Self::schema();

        let id_array = StringArray::from(vec![entry.id.clone()]);
        let content_array = StringArray::from(vec![entry.content.clone()]);

        // Handle embedding as FixedSizeList
        let embedding_array = if let Some(ref embedding) = entry.embedding {
            FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                vec![Some(embedding.iter().map(|&v| Some(v)).collect::<Vec<_>>())],
                EMBEDDING_DIM as i32,
            )
        } else {
            // Create a null array with the correct type
            FixedSizeListArray::from_iter_primitive::<Float32Type, Option<Option<f32>>, _>(
                vec![None],
                EMBEDDING_DIM as i32,
            )
        };

        let memory_type_array =
            StringArray::from(vec![Self::memory_type_to_string(&entry.memory_type)]);

        let metadata_array = if entry.metadata.is_empty() {
            StringArray::from(vec![None::<String>])
        } else {
            StringArray::from(vec![Some(
                serde_json::to_string(&entry.metadata).unwrap_or_default(),
            )])
        };

        let created_at_array = Int64Array::from(vec![entry.created_at.timestamp()]);
        let last_accessed_array =
            Int64Array::from(vec![entry.last_accessed.map(|la| la.timestamp())]);
        let importance_array = Float32Array::from(vec![entry.importance]);
        let access_count_array = UInt32Array::from(vec![entry.access_count]);

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(content_array),
                Arc::new(embedding_array),
                Arc::new(memory_type_array),
                Arc::new(metadata_array),
                Arc::new(created_at_array),
                Arc::new(last_accessed_array),
                Arc::new(importance_array),
                Arc::new(access_count_array),
            ],
        )
        .map_err(|e| crate::Error::Agent(format!("Failed to create record batch: {}", e)))
    }

    fn parse_batch_row(batch: &RecordBatch, i: usize) -> Result<MemoryEntry> {
        let id = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|arr| arr.value(i).to_string())
            .unwrap_or_default();

        let content = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|arr| arr.value(i).to_string())
            .unwrap_or_default();

        let embedding = batch
            .column(2)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .and_then(|arr| {
                if arr.is_null(i) {
                    return None;
                }
                let values = arr.value(i);
                values
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .map(|v| v.values().to_vec())
            });

        let memory_type = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|arr| arr.value(i).to_string())
            .unwrap_or_default();

        let metadata = batch
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .and_then(|arr| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i).to_string())
                }
            });

        let created_at = batch
            .column(5)
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|arr| arr.value(i))
            .unwrap_or(0);

        let last_accessed = batch
            .column(6)
            .as_any()
            .downcast_ref::<Int64Array>()
            .and_then(|arr| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i))
                }
            });

        let importance = batch
            .column(7)
            .as_any()
            .downcast_ref::<Float32Array>()
            .map(|arr| arr.value(i))
            .unwrap_or(0.5);

        let access_count = batch
            .column(8)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .map(|arr| arr.value(i))
            .unwrap_or(0);

        let metadata_map: std::collections::HashMap<String, serde_json::Value> = metadata
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();

        Ok(MemoryEntry {
            id,
            content,
            embedding,
            memory_type: Self::string_to_memory_type(&memory_type),
            metadata: metadata_map,
            created_at: chrono::DateTime::from_timestamp(created_at, 0)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(chrono::Utc::now),
            last_accessed: last_accessed
                .and_then(|ts| chrono::DateTime::from_timestamp(ts, 0))
                .map(|dt| dt.with_timezone(&chrono::Utc)),
            importance,
            access_count,
        })
    }
}

#[async_trait]
impl MemoryStore for LanceDbStore {
    async fn add(&self, entry: MemoryEntry) -> Result<String> {
        let id = entry.id.clone();
        let batch = Self::entry_to_batch(&entry)?;

        self.table
            .add(RecordBatchIterator::new(
                vec![Ok(batch.clone())],
                batch.schema(),
            ))
            .execute()
            .await
            .map_err(|e| crate::Error::Agent(format!("Failed to add memory: {}", e)))?;

        Ok(id)
    }

    async fn get(&self, id: &str) -> Result<Option<MemoryEntry>> {
        let batches = self
            .table
            .query()
            .only_if(format!("id = '{}'", id.replace('\'', "''")))
            .execute()
            .await
            .map_err(|e| crate::Error::Agent(format!("Failed to query: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| crate::Error::Agent(format!("Failed to collect batches: {}", e)))?;

        if let Some(batch) = batches.first()
            && batch.num_rows() > 0
        {
            return Ok(Some(Self::parse_batch_row(batch, 0)?));
        }

        Ok(None)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        self.table
            .delete(&format!("id = '{}'", id.replace('\'', "''")))
            .await
            .map_err(|e| crate::Error::Agent(format!("Failed to delete memory: {}", e)))?;

        Ok(())
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        let batches = self
            .table
            .query()
            .only_if(format!("content LIKE '%{}%'", query.replace('\'', "''")))
            .limit(limit)
            .execute()
            .await
            .map_err(|e| crate::Error::Agent(format!("Failed to search: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| crate::Error::Agent(format!("Failed to collect batches: {}", e)))?;

        let mut entries = Vec::new();
        for batch in batches {
            for i in 0..batch.num_rows() {
                entries.push(Self::parse_batch_row(&batch, i)?);
            }
        }

        Ok(entries)
    }

    async fn search_by_embedding(
        &self,
        embedding: &[f32],
        limit: usize,
        threshold: f32,
    ) -> Result<Vec<MemoryEntry>> {
        let batches = self
            .table
            .query()
            .limit(limit * 2) // Fetch more to filter by threshold
            .nearest_to(embedding)
            .map_err(|e| crate::Error::Agent(format!("Failed to create vector search: {}", e)))?
            .execute()
            .await
            .map_err(|e| crate::Error::Agent(format!("Failed to search by embedding: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| crate::Error::Agent(format!("Failed to collect batches: {}", e)))?;

        let mut entries_with_score = Vec::new();
        for batch in batches {
            for i in 0..batch.num_rows() {
                let entry = Self::parse_batch_row(&batch, i)?;

                // Get similarity from _distance column if present
                let similarity = if let Some(distance_col) = batch.column_by_name("_distance") {
                    let dist = distance_col
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .map(|arr| arr.value(i))
                        .unwrap_or(1.0);
                    1.0 - dist // Convert distance to similarity
                } else if let Some(ref entry_embedding) = entry.embedding {
                    cosine_similarity(embedding, entry_embedding)
                } else {
                    0.0
                };

                if similarity >= threshold {
                    entries_with_score.push((entry, similarity));
                }
            }
        }

        // Sort by similarity descending
        entries_with_score
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries_with_score.truncate(limit);

        Ok(entries_with_score.into_iter().map(|(e, _)| e).collect())
    }

    async fn ids(&self) -> Result<Vec<String>> {
        let batches = self
            .table
            .query()
            .select(Select::columns(&["id"]))
            .execute()
            .await
            .map_err(|e| crate::Error::Agent(format!("Failed to query ids: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| crate::Error::Agent(format!("Failed to collect batches: {}", e)))?;

        let mut ids = Vec::new();
        for batch in batches {
            if let Some(id_array) = batch
                .column_by_name("id")
                .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            {
                for i in 0..id_array.len() {
                    ids.push(id_array.value(i).to_string());
                }
            }
        }

        Ok(ids)
    }

    async fn count(&self) -> Result<usize> {
        let batches = self
            .table
            .query()
            .select(Select::columns(&["id"]))
            .execute()
            .await
            .map_err(|e| crate::Error::Agent(format!("Failed to count: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| crate::Error::Agent(format!("Failed to collect batches: {}", e)))?;

        let mut count = 0;
        for batch in batches {
            count += batch.num_rows();
        }

        Ok(count)
    }

    async fn update(&self, entry: MemoryEntry) -> Result<()> {
        // LanceDB doesn't have a native update, so we delete and re-add
        self.delete(&entry.id).await?;
        self.add(entry).await?;
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        self.table
            .delete("true")
            .await
            .map_err(|e| crate::Error::Agent(format!("Failed to clear memories: {}", e)))?;

        Ok(())
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lancedb_store_basic() {
        let store = LanceDbStore::new().await.expect("Failed to create store");

        let entry = MemoryEntry::new("This is a test memory");
        let id = store.add(entry.clone()).await.expect("Failed to add");

        let retrieved = store.get(&id).await.expect("Failed to get");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "This is a test memory");
    }

    #[tokio::test]
    async fn test_lancedb_store_delete() {
        let store = LanceDbStore::new().await.expect("Failed to create store");

        let entry = MemoryEntry::new("Memory to delete");
        let id = store.add(entry).await.expect("Failed to add");

        store.delete(&id).await.expect("Failed to delete");

        let retrieved = store.get(&id).await.expect("Failed to get");
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_lancedb_store_search() {
        let store = LanceDbStore::new().await.expect("Failed to create store");

        store
            .add(MemoryEntry::new("Rust programming language"))
            .await
            .ok();
        store
            .add(MemoryEntry::new("Python machine learning"))
            .await
            .ok();
        store
            .add(MemoryEntry::new("Rust async programming"))
            .await
            .ok();

        let results = store.search("Rust", 10).await.expect("Failed to search");
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_lancedb_store_count() {
        let store = LanceDbStore::new().await.expect("Failed to create store");

        store.clear().await.ok();

        store.add(MemoryEntry::new("Test 1")).await.ok();
        store.add(MemoryEntry::new("Test 2")).await.ok();

        let count = store.count().await.expect("Failed to count");
        assert_eq!(count, 2);
    }
}
