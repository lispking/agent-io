//! Example: using the `#[tool]` macro for zero-boilerplate tool definitions
//!
//! Compare this to `basic.rs` which requires manual JSON schema, Box::pin,
//! and separate args structs. With `#[tool]` you just write a plain async fn.
//!
//! Run with: cargo run --example macro_tools

use std::sync::Arc;

use agent_io::{Agent, AgentEvent, llm::ChatOpenAI, tool};
use futures::{StreamExt, pin_mut};

// ── Tool definitions ─────────────────────────────────────────────────────────
//
// Just write a normal async fn with doc comments.
// The #[tool] macro generates the Tool impl automatically:
//   - fn doc comment     -> tool description sent to the LLM
//   - param doc comments -> JSON schema "description" for each parameter
//   - param types        -> JSON schema types (String->"string", f64->"number", etc.)
//
// Calling `get_weather()` returns Arc<dyn Tool> ready to hand to the agent.

/// Get the current weather for a city
#[tool(location = "The name of the city to fetch weather for")]
async fn get_weather(location: String) -> agent_io::Result<String> {
    // In a real implementation this would call a weather API
    Ok(format!("Weather in {location}: Sunny, 25\u{00B0}C"))
}

/// Evaluate a simple arithmetic expression
#[tool(expression = "The arithmetic expression to evaluate, e.g. '15 * 7' or '100 / 4'")]
async fn calculator(expression: String) -> agent_io::Result<String> {
    // Minimal parser supporting +, -, *, /
    let expression = expression.trim();
    for (op, sym) in [('*', "*"), ('/', "/"), ('+', "+"), ('-', "-")] {
        if let Some(pos) = expression.find(sym) {
            let lhs: f64 = expression[..pos].trim().parse().unwrap_or(0.0);
            let rhs: f64 = expression[pos + 1..].trim().parse().unwrap_or(0.0);
            let result = match op {
                '*' => lhs * rhs,
                '/' => lhs / rhs,
                '+' => lhs + rhs,
                '-' => lhs - rhs,
                _ => unreachable!(),
            };
            return Ok(format!("Result: {result}"));
        }
    }
    Ok("Could not parse expression".to_string())
}

// ── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let llm = ChatOpenAI::new("gpt-5.4-mini")?;

    // Note: tools are created by calling the function name — no Arc::new, no
    // FunctionTool::new, no manual JSON schema needed.
    let agent = Agent::builder()
        .with_llm(Arc::new(llm))
        .tool(get_weather()) // <- macro-generated constructor
        .tool(calculator()) // <- macro-generated constructor
        .system_prompt("You are a helpful assistant. Use tools when appropriate.")
        .build()?;

    println!("User: What's the weather in Tokyo and what's 15 * 7?\n");

    let stream = agent
        .query_stream("What's the weather in Tokyo and what's 15 * 7?")
        .await?;
    pin_mut!(stream);

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Text(e) if e.delta => print!("{}", e.content),
            AgentEvent::Text(e) => println!("Text: {}", e.content),
            AgentEvent::Thinking(e) => println!("[Thinking: {}]", e.content),
            AgentEvent::ToolCall(e) => {
                println!("\n[Calling tool: {} with args: {}]", e.name, e.arguments);
            }
            AgentEvent::ToolResult(e) => println!("[Tool result: {}]", e.result),
            AgentEvent::FinalResponse(e) => {
                println!("\n\n=== Final Response ===");
                println!("{}", e.content);
                if let Some(usage) = e.usage {
                    println!(
                        "\nTokens used: {} in, {} out",
                        usage.total_prompt_tokens, usage.total_completion_tokens
                    );
                }
            }
            AgentEvent::Error(e) => eprintln!("[Error]: {}", e.message),
            _ => {}
        }
    }

    Ok(())
}
