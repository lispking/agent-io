//! Shared utility functions.

use syn::Type;

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

/// Map a Rust type to its JSON Schema `type` string.
///
/// | Rust type | JSON Schema |
/// |-----------|-------------|
/// | `String` / `str` | `"string"` |
/// | `bool` | `"boolean"` |
/// | `f32` / `f64` | `"number"` |
/// | integer types | `"integer"` |
/// | anything else | `"string"` |
pub fn rust_type_to_json_schema(ty: &Type) -> &'static str {
    if let Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last()
    {
        return match seg.ident.to_string().as_str() {
            "String" | "str" => "string",
            "bool" => "boolean",
            "f32" | "f64" => "number",
            "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64"
            | "u128" | "usize" => "integer",
            _ => "string",
        };
    }
    "string"
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
