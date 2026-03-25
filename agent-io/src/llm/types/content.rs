//! Content part types for messages

use serde::{Deserialize, Serialize};

/// JSON Schema for tool parameters
pub type JsonSchema = serde_json::Map<String, serde_json::Value>;

/// Text content part
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContentPartText {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

impl ContentPartText {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            content_type: "text".to_string(),
            text: text.into(),
        }
    }
}

impl From<String> for ContentPartText {
    fn from(text: String) -> Self {
        Self::new(text)
    }
}

impl From<&str> for ContentPartText {
    fn from(text: &str) -> Self {
        Self::new(text)
    }
}

/// Image content part
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPartImage {
    #[serde(rename = "type")]
    pub content_type: String,
    pub image_url: ImageUrl,
}

/// Image URL structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl ContentPartImage {
    /// Create from URL
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            content_type: "image_url".to_string(),
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

    /// Create from base64 data
    pub fn from_base64(media_type: &str, data: &str) -> Self {
        Self {
            content_type: "image_url".to_string(),
            image_url: ImageUrl {
                url: format!("data:{};base64,{}", media_type, data),
                detail: None,
            },
        }
    }
}

/// Document content part (for PDFs etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPartDocument {
    #[serde(rename = "type")]
    pub content_type: String,
    pub source: DocumentSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

impl ContentPartDocument {
    pub fn from_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            content_type: "document".to_string(),
            source: DocumentSource {
                source_type: "base64".to_string(),
                media_type: media_type.into(),
                data: data.into(),
            },
        }
    }
}

/// Thinking content part
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPartThinking {
    #[serde(rename = "type")]
    pub content_type: String,
    pub thinking: String,
}

impl ContentPartThinking {
    pub fn new(thinking: impl Into<String>) -> Self {
        Self {
            content_type: "thinking".to_string(),
            thinking: thinking.into(),
        }
    }
}

/// Redacted thinking content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPartRedactedThinking {
    #[serde(rename = "type")]
    pub content_type: String,
    pub data: String,
}

/// Refusal content part
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPartRefusal {
    #[serde(rename = "type")]
    pub content_type: String,
    pub refusal: String,
}

/// Union type for all content parts
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContentPart {
    Text(ContentPartText),
    Image(ContentPartImage),
    Document(ContentPartDocument),
    Thinking(ContentPartThinking),
    RedactedThinking(ContentPartRedactedThinking),
    Refusal(ContentPartRefusal),
}

impl ContentPart {
    pub fn text(content: impl Into<String>) -> Self {
        ContentPart::Text(ContentPartText::new(content))
    }

    pub fn is_text(&self) -> bool {
        matches!(self, ContentPart::Text(_))
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentPart::Text(t) => Some(&t.text),
            _ => None,
        }
    }
}

impl From<String> for ContentPart {
    fn from(text: String) -> Self {
        ContentPart::text(text)
    }
}

impl From<&str> for ContentPart {
    fn from(text: &str) -> Self {
        ContentPart::text(text)
    }
}
