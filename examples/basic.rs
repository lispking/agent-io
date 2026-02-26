//! Basic agent example
//!
//! Run with: cargo run --example basic

use std::sync::Arc;

use agent_io::{
    Agent, AgentEvent,
    llm::ChatOpenAI,
    tools::{FunctionTool, Tool},
};
use futures::{StreamExt, pin_mut};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Create LLM
    let llm = ChatOpenAI::new("gpt-4o-mini")?;

    // Create tools
    let weather_tool = create_weather_tool();
    let calculator_tool = create_calculator_tool();

    // Create agent
    let agent = Agent::builder()
        .with_llm(Arc::new(llm))
        .tool(weather_tool as Arc<dyn Tool>)
        .tool(calculator_tool as Arc<dyn Tool>)
        .system_prompt("You are a helpful assistant. Use tools when appropriate.")
        .build()?;

    // Query the agent with streaming
    println!("User: What's the weather in Tokyo and what's 15 * 7?\n");

    let stream = agent
        .query_stream("What's the weather in Tokyo and what's 15 * 7?")
        .await?;
    pin_mut!(stream);

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Text(e) => {
                if e.delta {
                    print!("{}", e.content);
                } else {
                    println!("Text: {}", e.content);
                }
            }
            AgentEvent::Thinking(e) => {
                println!("[Thinking: {}]", e.content);
            }
            AgentEvent::ToolCall(e) => {
                println!("\n[Calling tool: {} with args: {}]", e.name, e.arguments);
            }
            AgentEvent::ToolResult(e) => {
                println!("[Tool result: {}]", e.result);
            }
            AgentEvent::FinalResponse(e) => {
                println!("\n\n=== Final Response ===");
                println!("{}", e.content);
                if let Some(usage) = e.usage {
                    println!("\n--- Usage ---");
                    println!("Prompt tokens: {}", usage.total_prompt_tokens);
                    println!("Completion tokens: {}", usage.total_completion_tokens);
                    println!("Total tokens: {}", usage.total_tokens);
                }
            }
            AgentEvent::Error(e) => {
                eprintln!("Error: {}", e.message);
            }
            _ => {}
        }
    }

    // Get usage summary
    let usage = agent.get_usage().await;
    println!("\n=== Session Usage ===");
    println!("Total tokens: {}", usage.total_tokens);
    println!("By model: {:?}", usage.by_model);

    Ok(())
}

/// Create a weather tool
fn create_weather_tool() -> Arc<dyn Tool> {
    Arc::new(FunctionTool::new(
        "get_weather",
        "Get the current weather for a location",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g. 'Tokyo, Japan'"
                }
            },
            "required": ["location"]
        })
        .as_object()
        .unwrap()
        .clone(),
        |args: WeatherArgs| {
            Box::pin(async move {
                // Simulated weather response
                Ok(format!(
                    "Weather in {}: Sunny, 25°C, humidity 60%",
                    args.location
                ))
            })
        },
    ))
}

/// Create a calculator tool
fn create_calculator_tool() -> Arc<dyn Tool> {
    Arc::new(FunctionTool::new(
        "calculate",
        "Perform basic arithmetic calculations",
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '15 * 7'"
                }
            },
            "required": ["expression"]
        })
        .as_object()
        .unwrap()
        .clone(),
        |args: CalculatorArgs| {
            Box::pin(async move {
                // Simple expression parser (just handles multiplication for demo)
                let parts: Vec<&str> = args.expression.split('*').collect();
                if parts.len() == 2 {
                    let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
                    let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
                    Ok(format!("Result: {}", a * b))
                } else {
                    Ok("Could not parse expression".to_string())
                }
            })
        },
    ))
}

#[derive(serde::Deserialize)]
struct WeatherArgs {
    location: String,
}

#[derive(serde::Deserialize)]
struct CalculatorArgs {
    expression: String,
}
