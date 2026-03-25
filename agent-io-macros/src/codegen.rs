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
    utils::{extract_doc, json_schema_for_type, object_schema, to_pascal_case},
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

    let schema_properties = params
        .iter()
        .map(|p| {
            let name = p.name.clone();
            let desc = attr
                .descriptions
                .get(&p.name)
                .cloned()
                .unwrap_or_else(|| p.name.clone());
            let mut schema = json_schema_for_type(&p.ty);
            if let serde_json::Value::Object(ref mut obj) = schema {
                obj.insert("description".to_string(), serde_json::Value::String(desc));
            }
            (name, schema)
        })
        .collect::<serde_json::Map<String, serde_json::Value>>();

    let required_names: Vec<String> = params
        .iter()
        .filter(|p| !p.optional)
        .map(|p| p.name.clone())
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

    let schema_json = object_schema(schema_properties, required_names);
    let schema_lit = LitStr::new(&schema_json.to_string(), proc_macro2::Span::call_site());
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
                    let __schema: serde_json::Value = serde_json::from_str(#schema_lit)
                        .expect("macro-generated schema must be valid JSON");
                    ::agent_io::llm::SchemaOptimizer::create_tool_definition(
                        #fn_name_lit,
                        #description_lit,
                        __schema,
                    )
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
