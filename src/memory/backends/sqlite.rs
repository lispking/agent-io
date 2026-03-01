//! SQLite-based memory store for persistent storage

use async_trait::async_trait;
use rusqlite::{Connection, params};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::Result;
use crate::memory::entry::{MemoryEntry, MemoryType};
use crate::memory::store::MemoryStore;

/// SQLite memory store for persistent storage
pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteStore {
    /// Create a new SQLite store with an in-memory database
    pub fn new() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(|e| {
            crate::Error::Agent(format!("Failed to create in-memory SQLite: {}", e))
        })?;

        Self::initialize_schema(&conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Create a new SQLite store with a file database
    pub fn open<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let path = path.into();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| crate::Error::Agent(format!("Failed to create directory: {}", e)))?;
        }

        let conn = Connection::open(&path)
            .map_err(|e| crate::Error::Agent(format!("Failed to open SQLite database: {}", e)))?;

        Self::initialize_schema(&conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Initialize database schema (synchronous)
    fn initialize_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB,
                memory_type TEXT NOT NULL DEFAULT 'short_term',
                metadata TEXT,
                created_at TEXT NOT NULL,
                last_accessed TEXT,
                importance REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0
            );
            
            CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at);
            
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id UNINDEXED,
                content,
                content='memories',
                content_rowid='rowid'
            );
            
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, id, content) 
                VALUES (new.rowid, new.id, new.content);
            END;
            
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, id, content) 
                VALUES('delete', old.rowid, old.id, old.content);
            END;
            
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, id, content) 
                VALUES('delete', old.rowid, old.id, old.content);
                INSERT INTO memories_fts(rowid, id, content) 
                VALUES (new.rowid, new.id, new.content);
            END;
            "#,
        )
        .map_err(|e| crate::Error::Agent(format!("Failed to initialize schema: {}", e)))?;

        Ok(())
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
}

impl Default for SqliteStore {
    fn default() -> Self {
        Self::new().expect("Failed to create default SqliteStore")
    }
}

#[async_trait]
impl MemoryStore for SqliteStore {
    async fn add(&self, entry: MemoryEntry) -> Result<String> {
        let conn = self.conn.clone();
        let id = entry.id.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn.blocking_lock();

            let embedding_bytes = entry.embedding.as_ref().map(|v| {
                let len = v.len() * std::mem::size_of::<f32>();
                let mut bytes = Vec::with_capacity(len);
                for &f in v {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                bytes
            });

            conn.execute(
                r#"
                INSERT INTO memories (id, content, embedding, memory_type, metadata, created_at, 
                                     last_accessed, importance, access_count)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                "#,
                params![
                    entry.id,
                    entry.content,
                    embedding_bytes,
                    Self::memory_type_to_string(&entry.memory_type),
                    if entry.metadata.is_empty() {
                        None::<String>
                    } else {
                        Some(serde_json::to_string(&entry.metadata).unwrap_or_default())
                    },
                    entry.created_at.to_rfc3339(),
                    entry.last_accessed.map(|t| t.to_rfc3339()),
                    entry.importance,
                    entry.access_count,
                ],
            )
            .map_err(|e| crate::Error::Agent(format!("Failed to insert memory: {}", e)))?;

            Ok(id)
        })
        .await
        .map_err(|e| crate::Error::Agent(format!("Task join error: {}", e)))?
    }

    async fn get(&self, id: &str) -> Result<Option<MemoryEntry>> {
        let conn = self.conn.clone();
        let id = id.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = conn.blocking_lock();

            let result = conn.query_row(
                "SELECT id, content, embedding, memory_type, metadata, created_at, 
                        last_accessed, importance, access_count 
                 FROM memories WHERE id = ?1",
                params![id],
                |row| {
                    let embedding_blob: Option<Vec<u8>> = row.get(2)?;
                    let embedding = embedding_blob.as_ref().map(|blob| {
                        let len = blob.len() / std::mem::size_of::<f32>();
                        let mut vec = Vec::with_capacity(len);
                        for chunk in blob.chunks(std::mem::size_of::<f32>()) {
                            let bytes: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                            vec.push(f32::from_le_bytes(bytes));
                        }
                        vec
                    });

                    let metadata_str: Option<String> = row.get(4)?;
                    let metadata: std::collections::HashMap<String, serde_json::Value> =
                        metadata_str
                            .and_then(|s| serde_json::from_str(&s).ok())
                            .unwrap_or_default();

                    let created_at_str: String = row.get(5)?;
                    let last_accessed_str: Option<String> = row.get(6)?;

                    Ok(MemoryEntry {
                        id: row.get(0)?,
                        content: row.get(1)?,
                        embedding,
                        memory_type: Self::string_to_memory_type(&row.get::<_, String>(3)?),
                        metadata,
                        created_at: chrono::DateTime::parse_from_rfc3339(&created_at_str)
                            .map(|dt| dt.with_timezone(&chrono::Utc))
                            .unwrap_or_else(|_| chrono::Utc::now()),
                        last_accessed: last_accessed_str
                            .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                            .map(|dt| dt.with_timezone(&chrono::Utc)),
                        importance: row.get(7)?,
                        access_count: row.get(8)?,
                    })
                },
            );

            match result {
                Ok(entry) => Ok(Some(entry)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(crate::Error::Agent(format!("Failed to get memory: {}", e))),
            }
        })
        .await
        .map_err(|e| crate::Error::Agent(format!("Task join error: {}", e)))?
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let conn = self.conn.clone();
        let id = id.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = conn.blocking_lock();

            conn.execute("DELETE FROM memories WHERE id = ?1", params![id])
                .map_err(|e| crate::Error::Agent(format!("Failed to delete memory: {}", e)))?;

            Ok(())
        })
        .await
        .map_err(|e| crate::Error::Agent(format!("Task join error: {}", e)))?
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        let conn = self.conn.clone();
        let query = query.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = conn.blocking_lock();

            let mut stmt = conn
                .prepare(
                    r#"
                SELECT m.id, m.content, m.embedding, m.memory_type, m.metadata, 
                       m.created_at, m.last_accessed, m.importance, m.access_count
                FROM memories m
                JOIN memories_fts fts ON m.id = fts.id
                WHERE memories_fts MATCH ?1
                ORDER BY m.importance DESC
                LIMIT ?2
                "#,
                )
                .map_err(|e| crate::Error::Agent(format!("Failed to prepare search: {}", e)))?;

            let entries = stmt
                .query_map(params![query, limit as i64], |row| {
                    let embedding_blob: Option<Vec<u8>> = row.get(2)?;
                    let embedding = embedding_blob.as_ref().map(|blob| {
                        let len = blob.len() / std::mem::size_of::<f32>();
                        let mut vec = Vec::with_capacity(len);
                        for chunk in blob.chunks(std::mem::size_of::<f32>()) {
                            let bytes: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                            vec.push(f32::from_le_bytes(bytes));
                        }
                        vec
                    });

                    let metadata_str: Option<String> = row.get(4)?;
                    let metadata: std::collections::HashMap<String, serde_json::Value> =
                        metadata_str
                            .and_then(|s| serde_json::from_str(&s).ok())
                            .unwrap_or_default();

                    let created_at_str: String = row.get(5)?;
                    let last_accessed_str: Option<String> = row.get(6)?;

                    Ok(MemoryEntry {
                        id: row.get(0)?,
                        content: row.get(1)?,
                        embedding,
                        memory_type: Self::string_to_memory_type(&row.get::<_, String>(3)?),
                        metadata,
                        created_at: chrono::DateTime::parse_from_rfc3339(&created_at_str)
                            .map(|dt| dt.with_timezone(&chrono::Utc))
                            .unwrap_or_else(|_| chrono::Utc::now()),
                        last_accessed: last_accessed_str
                            .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                            .map(|dt| dt.with_timezone(&chrono::Utc)),
                        importance: row.get(7)?,
                        access_count: row.get(8)?,
                    })
                })
                .map_err(|e| crate::Error::Agent(format!("Failed to search memories: {}", e)))?;

            let mut results = Vec::new();
            for entry in entries {
                results.push(
                    entry.map_err(|e| {
                        crate::Error::Agent(format!("Failed to parse entry: {}", e))
                    })?,
                );
            }

            Ok(results)
        })
        .await
        .map_err(|e| crate::Error::Agent(format!("Task join error: {}", e)))?
    }

    async fn search_by_embedding(
        &self,
        embedding: &[f32],
        limit: usize,
        threshold: f32,
    ) -> Result<Vec<MemoryEntry>> {
        // For SQLite, we need to compute similarity in memory
        // This is a simplified implementation - for production, consider using a vector database
        let conn = self.conn.clone();
        let embedding = embedding.to_vec();

        tokio::task::spawn_blocking(move || {
            let conn = conn.blocking_lock();

            let mut stmt = conn
                .prepare(
                    "SELECT id, content, embedding, memory_type, metadata, created_at, 
                        last_accessed, importance, access_count 
                 FROM memories 
                 WHERE embedding IS NOT NULL
                 ORDER BY importance DESC",
                )
                .map_err(|e| {
                    crate::Error::Agent(format!("Failed to prepare embedding search: {}", e))
                })?;

            let entries = stmt
                .query_map([], |row| {
                    let embedding_blob: Vec<u8> = row.get(2)?;
                    let stored_embedding: Vec<f32> = {
                        let len = embedding_blob.len() / std::mem::size_of::<f32>();
                        let mut vec = Vec::with_capacity(len);
                        for chunk in embedding_blob.chunks(std::mem::size_of::<f32>()) {
                            let bytes: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                            vec.push(f32::from_le_bytes(bytes));
                        }
                        vec
                    };

                    let metadata_str: Option<String> = row.get(4)?;
                    let metadata: std::collections::HashMap<String, serde_json::Value> =
                        metadata_str
                            .and_then(|s| serde_json::from_str(&s).ok())
                            .unwrap_or_default();

                    let created_at_str: String = row.get(5)?;
                    let last_accessed_str: Option<String> = row.get(6)?;

                    let entry = MemoryEntry {
                        id: row.get(0)?,
                        content: row.get(1)?,
                        embedding: Some(stored_embedding.clone()),
                        memory_type: Self::string_to_memory_type(&row.get::<_, String>(3)?),
                        metadata,
                        created_at: chrono::DateTime::parse_from_rfc3339(&created_at_str)
                            .map(|dt| dt.with_timezone(&chrono::Utc))
                            .unwrap_or_else(|_| chrono::Utc::now()),
                        last_accessed: last_accessed_str
                            .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                            .map(|dt| dt.with_timezone(&chrono::Utc)),
                        importance: row.get(7)?,
                        access_count: row.get(8)?,
                    };

                    Ok((entry, stored_embedding))
                })
                .map_err(|e| {
                    crate::Error::Agent(format!("Failed to search by embedding: {}", e))
                })?;

            // Compute cosine similarity and filter by threshold
            let mut results: Vec<_> = entries
                .filter_map(|r| r.ok())
                .map(|(entry, stored)| {
                    let similarity = cosine_similarity(&embedding, &stored);
                    (entry, similarity)
                })
                .filter(|(_, sim)| *sim >= threshold)
                .collect();

            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(limit);

            Ok(results.into_iter().map(|(entry, _)| entry).collect())
        })
        .await
        .map_err(|e| crate::Error::Agent(format!("Task join error: {}", e)))?
    }

    async fn ids(&self) -> Result<Vec<String>> {
        let conn = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn.blocking_lock();

            let mut stmt = conn
                .prepare("SELECT id FROM memories ORDER BY created_at DESC")
                .map_err(|e| crate::Error::Agent(format!("Failed to prepare ids: {}", e)))?;

            let ids = stmt
                .query_map([], |row| row.get(0))
                .map_err(|e| crate::Error::Agent(format!("Failed to get ids: {}", e)))?;

            let mut results = Vec::new();
            for id in ids {
                results.push(
                    id.map_err(|e| crate::Error::Agent(format!("Failed to parse id: {}", e)))?,
                );
            }

            Ok(results)
        })
        .await
        .map_err(|e| crate::Error::Agent(format!("Task join error: {}", e)))?
    }

    async fn count(&self) -> Result<usize> {
        let conn = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn.blocking_lock();

            let count: i64 = conn
                .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
                .map_err(|e| crate::Error::Agent(format!("Failed to count memories: {}", e)))?;

            Ok(count as usize)
        })
        .await
        .map_err(|e| crate::Error::Agent(format!("Task join error: {}", e)))?
    }

    async fn update(&self, entry: MemoryEntry) -> Result<()> {
        let conn = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn.blocking_lock();

            let embedding_bytes = entry.embedding.as_ref().map(|v| {
                let len = v.len() * std::mem::size_of::<f32>();
                let mut bytes = Vec::with_capacity(len);
                for &f in v {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                bytes
            });

            conn.execute(
                r#"
                UPDATE memories SET 
                    content = ?2,
                    embedding = ?3,
                    memory_type = ?4,
                    metadata = ?5,
                    last_accessed = ?6,
                    importance = ?7,
                    access_count = ?8
                WHERE id = ?1
                "#,
                params![
                    entry.id,
                    entry.content,
                    embedding_bytes,
                    Self::memory_type_to_string(&entry.memory_type),
                    if entry.metadata.is_empty() {
                        None::<String>
                    } else {
                        Some(serde_json::to_string(&entry.metadata).unwrap_or_default())
                    },
                    entry.last_accessed.map(|t| t.to_rfc3339()),
                    entry.importance,
                    entry.access_count,
                ],
            )
            .map_err(|e| crate::Error::Agent(format!("Failed to update memory: {}", e)))?;

            Ok(())
        })
        .await
        .map_err(|e| crate::Error::Agent(format!("Task join error: {}", e)))?
    }

    async fn clear(&self) -> Result<()> {
        let conn = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn.blocking_lock();

            conn.execute("DELETE FROM memories", [])
                .map_err(|e| crate::Error::Agent(format!("Failed to clear memories: {}", e)))?;

            Ok(())
        })
        .await
        .map_err(|e| crate::Error::Agent(format!("Task join error: {}", e)))?
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
    async fn test_sqlite_store_basic() {
        let store = SqliteStore::new().expect("Failed to create store");

        let entry = MemoryEntry::new("This is a test memory");
        let id = store.add(entry.clone()).await.expect("Failed to add");

        let retrieved = store.get(&id).await.expect("Failed to get");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "This is a test memory");
    }

    #[tokio::test]
    async fn test_sqlite_store_delete() {
        let store = SqliteStore::new().expect("Failed to create store");

        let entry = MemoryEntry::new("Memory to delete");
        let id = store.add(entry).await.expect("Failed to add");

        store.delete(&id).await.expect("Failed to delete");

        let retrieved = store.get(&id).await.expect("Failed to get");
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_sqlite_store_search() {
        let store = SqliteStore::new().expect("Failed to create store");

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
    async fn test_sqlite_store_count() {
        let store = SqliteStore::new().expect("Failed to create store");

        store.clear().await.ok();

        store.add(MemoryEntry::new("Test 1")).await.ok();
        store.add(MemoryEntry::new("Test 2")).await.ok();

        let count = store.count().await.expect("Failed to count");
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_sqlite_store_embedding() {
        let store = SqliteStore::new().expect("Failed to create store");

        let entry =
            MemoryEntry::new("Test with embedding").with_embedding(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        store.add(entry).await.expect("Failed to add");

        let query_embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let results = store
            .search_by_embedding(&query_embedding, 10, 0.9)
            .await
            .expect("Failed to search by embedding");

        assert!(!results.is_empty());
        assert!(results[0].embedding.is_some());
    }
}
