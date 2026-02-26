//! Multi-provider example
//!
//! Run with:
//!   cargo run --example multi_provider --features full
//!
//! Or with specific provider:
//!   cargo run --example multi_provider --features openai

#[cfg(feature = "openai")]
use std::sync::Arc;

#[cfg(feature = "openai")]
use agent_io::{Agent, llm::ChatOpenAI};

#[cfg(feature = "anthropic")]
use agent_io::{Agent, llm::ChatAnthropic};

#[cfg(feature = "google")]
use agent_io::{Agent, llm::ChatGoogle};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(any(feature = "openai", feature = "anthropic", feature = "google"))]
    let prompt = "What is 2 + 2? Answer briefly.";

    // OpenAI
    #[cfg(feature = "openai")]
    {
        println!("=== OpenAI ===");
        let llm = ChatOpenAI::new("gpt-4o-mini")?;
        let agent = Agent::builder().with_llm(Arc::new(llm)).build()?;

        let response = agent.query(prompt).await?;
        println!("Response: {}\n", response);
    }

    // Anthropic
    #[cfg(feature = "anthropic")]
    {
        println!("=== Anthropic ===");
        let llm = ChatAnthropic::new("claude-3-5-sonnet-20241022")?;
        let agent = Agent::builder().with_llm(Arc::new(llm)).build()?;

        let response = agent.query(prompt).await?;
        println!("Response: {}\n", response);
    }

    // Google
    #[cfg(feature = "google")]
    {
        println!("=== Google ===");
        let llm = ChatGoogle::new("gemini-2.0-flash")?;
        let agent = Agent::builder().with_llm(Arc::new(llm)).build()?;

        let response = agent.query(prompt).await?;
        println!("Response: {}\n", response);
    }

    #[cfg(not(any(feature = "openai", feature = "anthropic", feature = "google")))]
    println!("No provider feature enabled. Run with --features openai|anthropic|google|full");

    Ok(())
}
