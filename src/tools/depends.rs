//! Dependency injection system

use std::any::Any;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Dependency resolver trait
pub trait Dependency: Clone + Send + Sync + 'static {}

impl<T: Clone + Send + Sync + 'static> Dependency for T {}

/// A dependency that can be resolved at runtime
#[derive(Clone)]
pub struct Depends<T>
where
    T: Dependency,
{
    factory: Arc<dyn Fn() -> Pin<Box<dyn Future<Output = T> + Send>> + Send + Sync>,
}

impl<T: Dependency> Depends<T> {
    /// Create a new dependency with a factory function
    pub fn new<F, Fut>(factory: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = T> + Send + 'static,
    {
        Self {
            factory: Arc::new(move || Box::pin(factory())),
        }
    }

    /// Create a dependency with a static value
    pub fn with_value(value: T) -> Self {
        Self::new(move || {
            let v = value.clone();
            async move { v }
        })
    }

    /// Resolve the dependency
    pub async fn resolve(&self) -> T {
        (self.factory)().await
    }
}

/// Dependency container for managing shared dependencies
#[derive(Default)]
pub struct DependencyContainer {
    dependencies: HashMap<std::any::TypeId, Box<dyn Any + Send + Sync>>,
}

impl DependencyContainer {
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
        }
    }

    /// Register a dependency
    pub fn register<T: 'static + Send + Sync>(&mut self, value: T) {
        self.dependencies
            .insert(std::any::TypeId::of::<T>(), Box::new(value));
    }

    /// Get a dependency
    pub fn get<T: 'static + Clone + Send + Sync>(&self) -> Option<T> {
        self.dependencies
            .get(&std::any::TypeId::of::<T>())
            .and_then(|v| v.downcast_ref::<T>())
            .cloned()
    }

    /// Check if a dependency exists
    pub fn contains<T: 'static>(&self) -> bool {
        self.dependencies.contains_key(&std::any::TypeId::of::<T>())
    }
}

/// Wrapper for dependency overrides
#[derive(Default)]
pub struct DependencyOverrides {
    inner: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl DependencyOverrides {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn insert<T: 'static + Send + Sync>(&mut self, key: &str, value: T) {
        self.inner.insert(key.to_string(), Box::new(value));
    }

    pub fn get<T: 'static + Clone + Send + Sync>(&self, key: &str) -> Option<T> {
        self.inner
            .get(key)
            .and_then(|v| v.downcast_ref::<T>())
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct Database {
        url: String,
    }

    #[tokio::test]
    async fn test_depends_with_value() {
        let db = Database {
            url: "postgresql://localhost".to_string(),
        };
        let depends = Depends::with_value(db.clone());

        let resolved = depends.resolve().await;
        assert_eq!(resolved, db);
    }

    #[tokio::test]
    async fn test_depends_with_factory() {
        let depends = Depends::new(|| async {
            Database {
                url: "postgresql://localhost".to_string(),
            }
        });

        let resolved = depends.resolve().await;
        assert_eq!(resolved.url, "postgresql://localhost");
    }

    #[test]
    fn test_dependency_container() {
        let mut container = DependencyContainer::new();
        let db = Database {
            url: "postgresql://localhost".to_string(),
        };

        container.register(db.clone());

        assert!(container.contains::<Database>());
        assert_eq!(container.get::<Database>(), Some(db));
    }
}
