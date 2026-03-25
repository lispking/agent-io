//! Procedural macros for the agent-io SDK.
//!
//! # `#[tool]` attribute macro
//!
//! Annotate an `async fn` to automatically generate a `Tool` implementation.
//! The function's doc comment becomes the tool description.
//! Parameter descriptions are provided as key=value pairs in the attribute.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use agent_io::tool;
//!
//! /// Get the current weather for a location
//! #[tool(location = "The city name to get weather for")]
//! async fn get_weather(location: String) -> agent_io::Result<String> {
//!     Ok(format!("Weather in {location}: Sunny, 25\u{00B0}C"))
//! }
//!
//! // Use in agent:
//! let agent = Agent::builder()
//!     .with_llm(Arc::new(llm))
//!     .tool(get_weather())
//!     .build()?;
//! ```

mod attr;
mod codegen;
mod params;
mod utils;

use proc_macro::TokenStream;
use syn::parse_macro_input;

use attr::ToolAttr;

/// Derive a `Tool` implementation from an annotated async function.
///
/// See the [crate-level documentation](self) for full usage.
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let tool_attr = parse_macro_input!(attr as ToolAttr);
    let input = parse_macro_input!(item as syn::ItemFn);
    match codegen::expand_tool(tool_attr, input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}
