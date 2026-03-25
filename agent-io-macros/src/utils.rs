//! Shared utility functions.

use serde_json::{Map, Value, json};
use syn::{GenericArgument, PathArguments, Type};

/// Extract concatenated `///` doc comments from an attribute list.
pub fn extract_doc(attrs: &[syn::Attribute]) -> String {
    attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .filter_map(|a| {
            if let syn::Meta::NameValue(nv) = &a.meta
                && let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) = &nv.value
            {
                return Some(s.value().trim().to_string());
            }
            None
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn path_segment_ident(ty: &Type) -> Option<String> {
    if let Type::Path(tp) = ty {
        tp.path.segments.last().map(|seg| seg.ident.to_string())
    } else {
        None
    }
}

fn first_generic_type(ty: &Type) -> Option<&Type> {
    let Type::Path(tp) = ty else {
        return None;
    };
    let seg = tp.path.segments.last()?;
    let PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    args.args.iter().find_map(|arg| match arg {
        GenericArgument::Type(inner) => Some(inner),
        _ => None,
    })
}

pub fn is_optional_type(ty: &Type) -> bool {
    matches!(path_segment_ident(ty).as_deref(), Some("Option"))
}

pub fn json_schema_for_type(ty: &Type) -> Value {
    if is_optional_type(ty) {
        return first_generic_type(ty)
            .map(json_schema_for_type)
            .unwrap_or_else(|| json!({ "type": "string" }));
    }

    if matches!(path_segment_ident(ty).as_deref(), Some("Vec")) {
        let items = first_generic_type(ty)
            .map(json_schema_for_type)
            .unwrap_or_else(|| json!({ "type": "string" }));
        return json!({
            "type": "array",
            "items": items,
        });
    }

    let schema_type = match path_segment_ident(ty).as_deref() {
        Some("String") | Some("str") => "string",
        Some("bool") => "boolean",
        Some("f32") | Some("f64") => "number",
        Some("i8") | Some("i16") | Some("i32") | Some("i64") | Some("i128") | Some("isize")
        | Some("u8") | Some("u16") | Some("u32") | Some("u64") | Some("u128") | Some("usize") => {
            "integer"
        }
        _ => "string",
    };

    json!({ "type": schema_type })
}

pub fn object_schema(properties: Map<String, Value>, required: Vec<String>) -> Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
    })
}

/// Convert `snake_case` to `PascalCase`.
pub fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}
