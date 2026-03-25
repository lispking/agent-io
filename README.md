# Agent IO

[![Crates.io](https://img.shields.io/crates/v/agent-io.svg)](https://crates.io/crates/agent-io)
[![Documentation](https://docs.rs/agent-io/badge.svg)](https://docs.rs/agent-io)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[<img alt="build status" src="https://img.shields.io/github/actions/workflow/status/lispking/agent-io/ci.yml?branch=main&style=for-the-badge" height="20">](https://github.com/lispking/agent-io/actions?query=branch%3Amain)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/lispking/agent-io)

A Rust SDK for building AI agents with multi-provider LLM support.

## Features

- **Multi-provider LLM support**: OpenAI, Anthropic, Google Gemini, and OpenAI-compatible providers
- **Tool/Function calling**: Built-in tool system — define tools with a simple `#[tool]` macro
- **Streaming responses**: Event-based real-time response handling
- **Context compaction**: Automatic management of long conversation context
- **Token tracking**: Usage tracking and cost calculation across providers
- **Retry mechanism**: Built-in exponential backoff retry for rate limit handling
- **Memory system**: Pluggable memory backends (in-memory, LanceDB vector store)

## Installation

```toml
[dependencies]
agent-io = "0.3"
tokio = { version = "1", features = ["full"] }
```

## Quick Start

### Basic Usage

```rust,no_run
use std::sync::Arc;
use agent_io::{Agent, llm::ChatOpenAI};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = ChatOpenAI::new("gpt-4o")?;
    let agent = Agent::builder()
        .with_llm(Arc::new(llm))
        .build()?;

    let response = agent.query("Hello!").await?;
    println!("{}", response);
    Ok(())
}
```

### Defining Tools with `#[tool]`

The `#[tool]` macro eliminates boilerplate — just write a plain `async fn`:

```rust,no_run
use std::sync::Arc;
use agent_io::{Agent, llm::ChatOpenAI, tool};

/// Get the current weather for a city
#[tool(location = "The city name to fetch weather for")]
async fn get_weather(location: String) -> agent_io::Result<String> {
    Ok(format!("Weather in {location}: Sunny, 25°C"))
}

/// Evaluate a simple arithmetic expression
#[tool(expression = "The expression to evaluate, e.g. '15 * 7'")]
async fn calculator(expression: String) -> agent_io::Result<String> {
    // ... implementation
    Ok("Result: 105".to_string())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = ChatOpenAI::new("gpt-5.4")?;
    let agent = Agent::builder()
        .with_llm(Arc::new(llm))
        .tool(get_weather())   // macro generates Arc<dyn Tool> constructor
        .tool(calculator())
        .system_prompt("You are a helpful assistant.")
        .build()?;

    let response = agent.query("What's the weather in Tokyo and 15 * 7?").await?;
    println!("{}", response);
    Ok(())
}
```

The macro automatically:
- Uses the function doc comment as the tool description
- Maps Rust types to JSON Schema types (`String` → `"string"`, `f64` → `"number"`, etc.)
- Uses the attribute key=value pairs as parameter descriptions
- Generates a zero-arg constructor returning `Arc<dyn Tool>`

### Manual Tool Definition

For more control, use `FunctionTool` or `ToolBuilder` directly:

```rust,no_run
use std::sync::Arc;
use agent_io::tools::{ToolBuilder, Tool};
use serde::Deserialize;

#[derive(Deserialize)]
struct WeatherArgs { location: String }

let tool: Box<dyn Tool> = ToolBuilder::new("get_weather")
    .description("Get weather for a location")
    .string_param("location", "The city name")
    .build(|args: WeatherArgs| Box::pin(async move {
        Ok(format!("Sunny in {}", args.location))
    }));
```

### Supported Providers

| Provider | Type | Environment Variable |
|----------|------|---------------------|
| OpenAI | `ChatOpenAI` | `OPENAI_API_KEY` |
| Anthropic | `ChatAnthropic` | `ANTHROPIC_API_KEY` |
| Google Gemini | `ChatGoogle` | `GOOGLE_API_KEY` |
| DeepSeek | `ChatDeepSeek` | `DEEPSEEK_API_KEY` |
| Groq | `ChatGroq` | `GROQ_API_KEY` |
| Mistral | `ChatMistral` | `MISTRAL_API_KEY` |
| Ollama | `ChatOllama` | — (local) |
| OpenRouter | `ChatOpenRouter` | `OPENROUTER_API_KEY` |
| OpenAI-compatible | `ChatOpenAICompatible` | configurable |

## Feature Flags

```toml
[dependencies.agent-io]
version = "0.3"
features = ["openai", "anthropic", "google"]
# Or enable all:
features = ["full"]
```

## Examples

```bash
# Basic example (manual tool definition)
cargo run --example basic

# Macro-based tools (zero boilerplate)
cargo run --example macro_tools

# Multi-provider
cargo run --example multi_provider --features full
```

## Workspace

This repository is a Cargo workspace:

```
agent-io/              # main SDK crate
agent-io-macros/       # proc-macro crate (#[tool])
```

## License

Licensed under the [Apache License 2.0](agent-io/LICENSE).
