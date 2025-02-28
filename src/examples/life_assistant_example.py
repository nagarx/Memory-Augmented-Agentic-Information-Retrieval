#!/usr/bin/env python3
"""
Example Life Assistant application using the Agentic IR framework.

This script demonstrates how to use the Agentic IR framework to build a life assistant
that can respond to user queries about weather, calendar, restaurants, and more.
"""

import os
import logging
import argparse
from typing import Optional

from src.core.agent import Agent
from src.core.memory import InMemoryStorage
from src.core.thought import ChainOfThoughtGenerator, ThoughtManager
from src.core.reward import SimpleRewardModel
from src.core.policy import LLMPolicy
from src.llm.ollama import create_ollama_client, create_completion_function
from src.environments.life_assistant import LifeAssistantEnvironment
from src.tools.search import WebSearchTool, WebContentTool

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_agent(model_name: str = "deepseek-coder:latest", verbose: bool = False) -> Agent:
    """
    Set up the agent with all necessary components.
    
    Args:
        model_name: The name of the Ollama model to use
        verbose: Whether to print verbose output
        
    Returns:
        An initialized agent
    """
    # Set up LLM client
    ollama_client = create_ollama_client(model_name=model_name)
    completion_fn = ollama_client.create_completion_function()
    
    # Set up memory
    memory = InMemoryStorage()
    
    # Set up thought manager
    thought_manager = ThoughtManager()
    
    # Set up thought generator
    thought_generator = ChainOfThoughtGenerator(
        llm_fn=completion_fn,
        max_thoughts=3,
        verbose=verbose
    )
    
    # Set up reward model
    reward_model = SimpleRewardModel(
        weights={
            "similarity": 0.7,
            "step_cost": 0.05,
            "time_cost": 0.05
        }
    )
    
    # Set up policy
    policy = LLMPolicy(
        llm_fn=completion_fn,
        thought_generator=thought_generator,
        thought_manager=thought_manager,
        memory=memory,
        verbose=verbose
    )
    
    # Create agent
    agent = Agent(
        memory=memory,
        thought_generator=thought_generator,
        thought_manager=thought_manager,
        policy=policy,
        reward_model=reward_model,
        verbose=verbose
    )
    
    # Add tools to the agent
    agent.add_tool(WebSearchTool())
    agent.add_tool(WebContentTool())
    
    return agent

def run_interactive_session(agent: Agent, env: LifeAssistantEnvironment):
    """
    Run an interactive session with the agent.
    
    Args:
        agent: The initialized agent
        env: The life assistant environment
    """
    print("\n🤖 Welcome to the Life Assistant! (powered by Agentic IR)\n")
    print("You can ask about weather, calendar, restaurants, set reminders, plan trips, and more.")
    print("Type 'quit', 'exit', or 'bye' to end the session.\n")
    
    # Initialize the environment with a random query
    current_state = env.reset()
    print(f"🧠 Initial query: {current_state.text}\n")
    
    # Set the agent's current state
    agent.set_current_state(current_state)
    
    # Process the initial query
    print("🧠 Thinking...")
    result = agent.run(env, max_steps=10)
    print(f"🤖 Response: {result.final_state.text}\n")
    
    # Interactive loop
    while True:
        # Get user input
        user_input = input("👤 You: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\n🤖 Thank you for using Life Assistant. Goodbye!")
            break
        
        # Create a new state with the user input
        current_state = env.reset()
        current_state.text = user_input
        
        # Set the agent's current state
        agent.set_current_state(current_state)
        
        # Process the user input
        print("\n🧠 Thinking...")
        result = agent.run(env, max_steps=10)
        
        # Display the response
        print(f"\n🤖 Response: {result.final_state.text}\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Life Assistant using Agentic IR")
    parser.add_argument("--model", type=str, default="deepseek-coder:latest",
                        help="Ollama model name to use")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    args = parser.parse_args()
    
    try:
        # Set up the agent
        print("📝 Setting up the Life Assistant...")
        agent = setup_agent(model_name=args.model, verbose=args.verbose)
        
        # Create environment
        env = LifeAssistantEnvironment()
        
        # Run interactive session
        run_interactive_session(agent, env)
        
    except KeyboardInterrupt:
        print("\n\n🤖 Session interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main() 