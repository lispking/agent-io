//! Tool system with dependency injection

mod builder;
mod depends;
mod function;
mod simple;
mod tool;

// Re-export core types
pub use tool::{EphemeralConfig, Tool, ToolResult};

// Re-export tool implementations
pub use builder::ToolBuilder;
pub use function::FunctionTool;
pub use simple::SimpleTool;

// Re-export from depends module
pub use depends::{Dependency, DependencyContainer, Depends};
