//! Tool parameter extraction from function signatures.

use syn::{FnArg, ItemFn, Pat, PatType, ReturnType, Type, punctuated::Punctuated, token::Comma};

use crate::utils::is_optional_type;

/// A single parsed parameter of a `#[tool]` function.
pub struct ToolParam {
    pub name: String,
    pub ty: Type,
    pub optional: bool,
}

/// Extract `ToolParam`s from a function's input list.
///
/// Returns an error if:
/// - Any parameter uses `self`
/// - Any parameter uses a non-ident pattern (e.g. destructuring)
pub fn collect_params(inputs: &Punctuated<FnArg, Comma>) -> syn::Result<Vec<ToolParam>> {
    let mut params = Vec::new();
    for arg in inputs {
        match arg {
            FnArg::Receiver(_) => {
                return Err(syn::Error::new_spanned(
                    arg,
                    "#[tool] functions cannot take `self`",
                ));
            }
            FnArg::Typed(PatType { pat, ty, .. }) => {
                let name = match pat.as_ref() {
                    Pat::Ident(pi) => pi.ident.to_string(),
                    other => {
                        return Err(syn::Error::new_spanned(
                            other,
                            "#[tool] parameters must be simple identifiers",
                        ));
                    }
                };
                let ty = *ty.clone();
                params.push(ToolParam {
                    name,
                    optional: is_optional_type(&ty),
                    ty,
                });
            }
        }
    }
    Ok(params)
}

/// Validate that the function return type is explicit and plausibly `Result<String, _>`.
pub fn validate_return_type(ret: &ReturnType) -> syn::Result<()> {
    match ret {
        ReturnType::Default => Err(syn::Error::new_spanned(
            ret,
            "#[tool] functions must return `agent_io::Result<String>`",
        )),
        ReturnType::Type(_, ty) => {
            let Type::Path(tp) = ty.as_ref() else {
                return Err(syn::Error::new_spanned(
                    ty,
                    "#[tool] functions must return `agent_io::Result<String>`",
                ));
            };
            let Some(seg) = tp.path.segments.last() else {
                return Err(syn::Error::new_spanned(
                    ty,
                    "#[tool] functions must return `agent_io::Result<String>`",
                ));
            };
            if seg.ident != "Result" {
                return Err(syn::Error::new_spanned(
                    ty,
                    "#[tool] functions must return `agent_io::Result<String>`",
                ));
            }
            Ok(())
        }
    }
}

/// Validate that the function is async.
pub fn validate_async(item: &ItemFn) -> syn::Result<()> {
    if item.sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            item.sig.fn_token,
            "#[tool] functions must be async",
        ));
    }
    Ok(())
}
