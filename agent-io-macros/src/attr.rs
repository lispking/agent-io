//! Attribute argument parser for `#[tool(...)]`.
//!
//! Parses key=value pairs of the form:
//!   `#[tool(param1 = "desc1", param2 = "desc2")]`

use std::collections::HashMap;

use syn::{LitStr, parse::ParseStream};

/// Parsed arguments from the `#[tool(...)]` attribute.
pub struct ToolAttr {
    /// Maps parameter name → description string.
    pub descriptions: HashMap<String, String>,
}

impl syn::parse::Parse for ToolAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut descriptions = HashMap::new();
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            let _eq: syn::Token![=] = input.parse()?;
            let val: LitStr = input.parse()?;
            descriptions.insert(key.to_string(), val.value());
            if input.peek(syn::Token![,]) {
                let _: syn::Token![,] = input.parse()?;
            }
        }
        Ok(ToolAttr { descriptions })
    }
}
