# Agent IO

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A Rust SDK for building AI agents with multi-provider LLM support.

## Features

- **Multi-provider LLM support**: OpenAI, Anthropic, Google Gemini, and OpenAI-compatible providers
- **Tool/Function calling**: Built-in tool system with dependency injection
- **Streaming responses**: Event-based real-time response handling
- **Context compaction**: Automatic management of long conversation context
- **Token tracking**: Usage tracking and cost calculation across providers
- **Retry mechanism**: Built-in exponential backoff retry for rate limit handling

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
agent-io = "0.1"
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

### Using Tools

```rust,no_run
use std::sync::Arc;
use agent_io::{
    Agent, AgentEvent,
    llm::ChatOpenAI,
    tools::{FunctionTool, Tool, EphemeralConfig},
};
use futures::{StreamExt, pin_mut};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = ChatOpenAI::new("gpt-4o")?;

    // Create a weather tool
    let weather_tool = Arc::new(FunctionTool::new(
        "get_weather",
        "Get current weather for a location",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'Tokyo'"
                }
            },
            "required": ["location"]
        })
        .as_object()
        .unwrap()
        .clone(),
        |args: WeatherArgs| {
            Box::pin(async move {
                Ok(format!("Weather in {}: Sunny, 25°C", args.location))
            })
        },
    ));

    // Create agent with tools
    let agent = Agent::builder()
        .with_llm(Arc::new(llm))
        .tool(weather_tool)
        .system_prompt("You are a helpful assistant. Use tools when appropriate.")
        .build()?;

    // Stream query
    let stream = agent.query_stream("What's the weather in Tokyo?").await?;
    pin_mut!(stream);

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Text(e) => print!("{}", e.content),
            AgentEvent::ToolCall(e) => println!("\n[Calling tool: {}]", e.name),
            AgentEvent::ToolResult(e) => println!("[Result: {}]", e.result),
            AgentEvent::FinalResponse(e) => println!("\n{}", e.content),
            _ => {}
        }
    }

    Ok(())
}

#[derive(serde::Deserialize)]
struct WeatherArgs {
    location: String,
}
```

## Supported LLM Providers

| Provider | Model Type | Feature Flag | Environment Variable |
|----------|-----------|--------------|---------------------|
| OpenAI | `ChatOpenAI` | `openai` | `OPENAI_API_KEY` |
| Anthropic | `ChatAnthropic` | `anthropic` | `ANTHROPIC_API_KEY` |
| Google Gemini | `ChatGoogle` | `google` | `GOOGLE_API_KEY` |
| OpenRouter | `ChatOpenRouter` | - | `OPENROUTER_API_KEY` |
| Groq | `ChatGroq` | - | `GROQ_API_KEY` |
| Mistral | `ChatMistral` | - | `MISTRAL_API_KEY` |
| DeepSeek | `ChatDeepSeek` | - | `DEEPSEEK_API_KEY` |
| Ollama | `ChatOllama` | - | - |

## Configuration

### Agent Builder

```rust
let agent = Agent::builder()
    .with_llm(Arc::new(llm))
    .system_prompt("You are a professional assistant")  // System prompt
    .max_iterations(100)                                 // Max iterations
    .build()?;
```

### Tool Configuration

```rust
// Create an ephemeral tool (results removed from context after use)
let tool = FunctionTool::new(
    "search",
    "Search for relevant information",
    schema,
    |args| Box::pin(async move { Ok("Search results".to_string()) }),
)
.with_ephemeral(EphemeralConfig::Single);  // Remove after use
```

## Event Types

| Event | Description |
|-------|-------------|
| `AgentEvent::Text` | Text content chunk |
| `AgentEvent::Thinking` | Model thinking process (Claude support) |
| `AgentEvent::ToolCall` | Tool call request |
| `AgentEvent::ToolResult` | Tool execution result |
| `AgentEvent::FinalResponse` | Final response |
| `AgentEvent::Error` | Error message |
| `AgentEvent::StepStart` | Step started |
| `AgentEvent::StepComplete` | Step completed |

## Usage Tracking

```rust
// Get session usage statistics
let usage = agent.get_usage().await;
println!("Total tokens: {}", usage.total_tokens);
println!("By model: {:?}", usage.by_model);
```

## OpenAI-Compatible Providers

For providers that support OpenAI API format, use `ChatOpenAICompatible`:

```rust,no_run
use agent_io::llm::ChatOpenAICompatible;

let llm = ChatOpenAICompatible::new("your-model")
    .with_base_url("https://your-api-endpoint.com/v1")
    .with_api_key("your-api-key");
```

## Feature Flags

```toml
[dependencies.agent-io]
version = "0.1"
features = ["openai", "anthropic", "google"]
# Or use "full" to enable all major providers
features = ["full"]
```

## Examples

Run the examples:

```bash
# Basic example
cargo run --example basic

# Multi-provider example
cargo run --example multi_provider --features full
```

## License

Licensed under the [Apache License 2.0](LICENSE).
