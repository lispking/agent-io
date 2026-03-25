//! Code generation for the `#[tool]` macro.
//!
//! `expand_tool` takes the parsed attribute + function and emits a constructor
//! function that returns `Arc<dyn agent_io::tools::Tool>`.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{ItemFn, LitStr};

use crate::{
    attr::ToolAttr,
    params::{collect_params, validate_async, validate_return_type},
    utils::{extract_doc, rust_type_to_json_schema, to_pascal_case},
};

pub fn expand_tool(attr: ToolAttr, input: ItemFn) -> syn::Result<TokenStream2> {
    let fn_name = &input.sig.ident;
    let fn_name_str = fn_name.to_string();

    // Validate
    validate_async(&input)?;
    validate_return_type(&input.sig.output)?;

    let tool_description = extract_doc(&input.attrs);
    if tool_description.is_empty() {
        return Err(syn::Error::new_spanned(
            fn_name,
            "#[tool] functions must have a doc comment describing what the tool does",
        ));
    }

    let params = collect_params(&input.sig.inputs)?;

    // Derived identifiers
    let pascal = to_pascal_case(&fn_name_str);
    let tool_struct = quote::format_ident!("__ToolImpl_{}", pascal);
    let args_struct = quote::format_ident!("__ToolArgs_{}", pascal);

    // Schema property entries
    let schema_entries = params.iter().map(|p| {
        let name = &p.name;
        let desc = attr
            .descriptions
            .get(&p.name)
            .cloned()
            .unwrap_or_else(|| p.name.clone());
        let type_str = rust_type_to_json_schema(&p.ty);
        quote! {
            __map.insert(
                #name.to_string(),
                ::agent_io::__macro_support::serde_json::json!({
                    "type": #type_str,
                    "description": #desc
                })
            );
        }
    });

    let required_names: Vec<LitStr> = params
        .iter()
        .map(|p| LitStr::new(&p.name, proc_macro2::Span::call_site()))
        .collect();

    // Args struct fields
    let args_fields = params.iter().map(|p| {
        let ident = quote::format_ident!("{}", p.name);
        let ty = &p.ty;
        quote! { #ident: #ty }
    });

    // Arguments forwarded to the inner function
    let call_args = params.iter().map(|p| {
        let ident = quote::format_ident!("{}", p.name);
        quote! { __args.#ident }
    });

    let has_params = !params.is_empty();

    // Inner function: same body, renamed to `__inner`, #[tool] attr stripped
    let inner_fn = {
        let mut f = input.clone();
        f.sig.ident = quote::format_ident!("__inner");
        f.attrs.retain(|a| !a.path().is_ident("tool"));
        f
    };

    let execute_body = if has_params {
        quote! {
            let __args: #args_struct =
                ::agent_io::__macro_support::serde_json::from_value(__json_args)
                    .map_err(|e| ::agent_io::Error::Tool(format!("Failed to parse args: {e}")))?;
            __inner(#(#call_args),*).await
        }
    } else {
        quote! {
            let _ = __json_args;
            __inner().await
        }
    };

    let description_lit = LitStr::new(&tool_description, proc_macro2::Span::call_site());
    let fn_name_lit = LitStr::new(&fn_name_str, proc_macro2::Span::call_site());

    Ok(quote! {
        #[allow(non_snake_case, dead_code)]
        fn #fn_name() -> ::std::sync::Arc<dyn ::agent_io::tools::Tool> {
            #[allow(unused_imports)]
            use ::agent_io::__macro_support::serde_json;

            #[derive(::agent_io::__macro_support::serde::Deserialize)]
            #[allow(non_camel_case_types, dead_code)]
            struct #args_struct {
                #(#args_fields,)*
            }

            #[allow(non_camel_case_types)]
            struct #tool_struct;

            #[::agent_io::__macro_support::async_trait]
            impl ::agent_io::tools::Tool for #tool_struct {
                fn name(&self) -> &str {
                    #fn_name_lit
                }

                fn description(&self) -> &str {
                    #description_lit
                }

                fn definition(&self) -> ::agent_io::llm::ToolDefinition {
                    let mut __map = serde_json::Map::new();
                    #(#schema_entries)*
                    let mut __schema = serde_json::Map::new();
                    __schema.insert("type".to_string(), serde_json::json!("object"));
                    __schema.insert("properties".to_string(), serde_json::Value::Object(__map));
                    __schema.insert("required".to_string(), serde_json::json!([#(#required_names),*]));
                    ::agent_io::llm::ToolDefinition::new(#fn_name_lit, #description_lit, __schema)
                }

                async fn execute(
                    &self,
                    __json_args: ::agent_io::__macro_support::serde_json::Value,
                ) -> ::agent_io::Result<::agent_io::tools::ToolResult> {
                    #inner_fn
                    let content = { #execute_body }?;
                    Ok(::agent_io::tools::ToolResult::new("", content))
                }
            }

            ::std::sync::Arc::new(#tool_struct)
        }
    })
}
