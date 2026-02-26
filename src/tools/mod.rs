//! Tool system with dependency injection

mod depends;
mod tool;

// Re-export from tool module
pub use tool::{EphemeralConfig, FunctionTool, SimpleTool, Tool, ToolBuilder, ToolResult};

// Re-export from depends module
pub use depends::{Dependency, DependencyContainer, DependencyOverrides, Depends};
