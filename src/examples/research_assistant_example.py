#!/usr/bin/env python3
"""
Example Research Assistant application using the Agentic IR framework.

This script demonstrates how to use the Agentic IR framework to build a research assistant
that can answer questions based on a collection of research papers.
"""

import os
import logging
import argparse
import glob
from typing import Optional, List

from src.core.agent import Agent
from src.core.memory import InMemoryStorage
from src.core.thought import ChainOfThoughtGenerator, ThoughtManager
from src.core.reward import SimpleRewardModel
from src.core.policy import LLMPolicy
from src.core.state import InformationState
from src.llm.ollama import create_ollama_client, create_completion_function
from src.tools.document_retrieval import DocumentSearchTool, DocumentReadTool, DocumentListTool
from src.tools.search import WebSearchTool

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_agent(model_name: str = "deepseek-r1:14b", verbose: bool = False) -> Agent:
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
    
    # Store the model name directly on the agent for later use
    agent.model_name = model_name
    
    # Add tools to the agent
    agent.add_tool(DocumentSearchTool())
    agent.add_tool(DocumentReadTool())
    agent.add_tool(DocumentListTool())
    agent.add_tool(WebSearchTool())
    
    return agent

def run_interactive_session(agent: Agent):
    """
    Run an interactive session with the agent.
    
    Args:
        agent: The initialized agent
    """
    print("\n🤖 Welcome to the Research Assistant! (powered by Agentic IR)\n")
    print("I can answer questions based on the research papers in the repository.")
    print("Type 'list papers' to see available papers.")
    print("Type 'quit', 'exit', or 'bye' to end the session.\n")
    
    available_actions = [
        "document_search",
        "document_read",
        "document_list",
        "web_search",
        "synthesize_information",
        "ask_clarification"
    ]
    
    # Check if there are any papers in the repository
    docs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "research_papers"
    )
    paper_files = glob.glob(os.path.join(docs_dir, "*.md"))
    paper_files = [f for f in paper_files if not f.endswith("README.md")]
    
    if not paper_files:
        print("⚠️ No research papers found in the repository!")
        print(f"Please add Markdown files to: {docs_dir}\n")
    else:
        print(f"📚 Found {len(paper_files)} research papers in the repository.\n")
    
    # Interactive loop
    while True:
        # Get user input
        user_input = input("👤 You: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\n🤖 Thank you for using the Research Assistant. Goodbye!")
            break
            
        # Check for list papers command
        if user_input.lower() in ["list papers", "list documents", "show papers"]:
            tool_result = agent.call_tool("document_list")
            
            if tool_result.success:
                documents = tool_result.result.get("documents", [])
                print("\n📄 Available Papers:")
                for i, doc in enumerate(documents, 1):
                    print(f"{i}. {doc['title']} ({doc['filename']})")
                print()
                continue
        
        # Special case for PARTNR framework from Meta
        if "partnr" in user_input.lower() and "meta" in user_input.lower():
            # Find the PARTNR paper
            partnr_paper = None
            tool_result = agent.call_tool("document_list")
            
            if tool_result.success:
                documents = tool_result.result.get("documents", [])
                for doc in documents:
                    if "PARTNR" in doc["filename"]:
                        partnr_paper = doc
                        break
            
            if partnr_paper:
                print(f"\n📖 Reading PARTNR paper...")
                
                # Read the full paper
                read_result = agent.call_tool("document_read", filename=partnr_paper['filename'])
                
                if read_result.success:
                    paper_content = read_result.result.get("content", "")
                    
                    # Generate a focused explanation using the LLM
                    prompt = f"Please explain how the PARTNR framework from Meta works based on this research paper:\n\n"
                    prompt += f"Title: {partnr_paper['title']}\n\n"
                    prompt += f"Content:\n{paper_content[:8000]}..."  # Limit content size
                    
                    # Use the LLM to generate a summary
                    try:
                        ollama_client = create_ollama_client(model_name=agent.model_name)
                        completion_fn = ollama_client.create_completion_function()
                        explanation = completion_fn(prompt)
                        
                        print(f"\n🤖 Explanation of the PARTNR framework:\n")
                        print(explanation)
                        print()
                    except Exception as e:
                        print(f"\n❌ Error generating explanation: {e}")
                else:
                    print(f"\n❌ Error reading PARTNR paper: {read_result.error}")
                
                continue
        
        # Check for paper explanation requests
        paper_number_match = None
        for pattern in [
            r"explain paper ([0-9]+)",
            r"explain the ([0-9]+)[a-z]* paper", 
            r"explain ([0-9]+)",
            r"explain paper #([0-9]+)",
            r"explain the #([0-9]+) paper",
            r"summarize paper ([0-9]+)",
            r"summarize the ([0-9]+)[a-z]* paper",
            r"explain the second paper",
            r"explain the third paper",
            r"explain the first paper"
        ]:
            import re
            match = re.search(pattern, user_input.lower())
            if match:
                if pattern == r"explain the second paper":
                    paper_number_match = 2
                elif pattern == r"explain the third paper":
                    paper_number_match = 3
                elif pattern == r"explain the first paper":
                    paper_number_match = 1
                else:
                    paper_number_match = int(match.group(1))
                break
        
        if paper_number_match is not None:
            # Get the list of papers
            tool_result = agent.call_tool("document_list")
            
            if tool_result.success:
                documents = tool_result.result.get("documents", [])
                
                if 1 <= paper_number_match <= len(documents):
                    selected_paper = documents[paper_number_match - 1]
                    print(f"\n📖 Reading paper: {selected_paper['title']}...")
                    
                    # Read the full paper
                    read_result = agent.call_tool("document_read", filename=selected_paper['filename'])
                    
                    if read_result.success:
                        paper_content = read_result.result.get("content", "")
                        
                        # Generate a summary using the LLM
                        prompt = f"Please provide a comprehensive summary of the following research paper:\n\n"
                        prompt += f"Title: {selected_paper['title']}\n\n"
                        prompt += f"Content:\n{paper_content[:8000]}..."  # Limit content size
                        
                        # Use the LLM to generate a summary
                        try:
                            ollama_client = create_ollama_client(model_name=agent.model_name)
                            completion_fn = ollama_client.create_completion_function()
                            summary = completion_fn(prompt)
                            
                            print(f"\n🤖 Summary of {selected_paper['title']}:\n")
                            print(summary)
                            print()
                        except Exception as e:
                            print(f"\n❌ Error generating summary: {e}")
                            print("Here's the beginning of the paper instead:\n")
                            print(paper_content[:1000])
                    else:
                        print(f"\n❌ Error reading paper: {read_result.error}")
                else:
                    print(f"\n❌ Invalid paper number. Please choose a number between 1 and {len(documents)}.")
                
                continue
        
        # Create a new state with the user input
        current_state = InformationState(
            id=str(user_input)[:8],
            text=user_input,
            available_actions=available_actions,
            timestamp=0,
            data={}
        )
        
        # Set the agent's current state
        agent.set_current_state(current_state)
        
        # Set target state based on user question
        target_state = InformationState(
            id="target",
            text=f"A comprehensive, accurate answer to the user's question: {user_input}",
            timestamp=0,
            data={}
        )
        agent.set_target_state(target_state)
        
        # Process the user input
        print("\n🧠 Thinking...")
        
        # Search the documents for relevant information
        search_result = agent.call_tool("document_search", query=user_input, max_results=3)
        
        response = ""
        document_content = []
        
        # Check if search was successful
        if search_result.success and search_result.result.get("num_results", 0) > 0:
            results = search_result.result.get("results", [])
            
            # Collect document content
            for result in results:
                document_content.append(f"From {result['source']}, section '{result['section']}':\n{result['content']}")
            
            # Combine document content
            combined_content = "\n\n".join(document_content)
            
            # Generate response
            prompt = f"Based on the following research paper excerpts, answer this question: {user_input}\n\n"
            prompt += f"Research paper excerpts:\n{combined_content}"
            
            try:
                ollama_client = create_ollama_client(model_name=agent.model_name)
                completion_fn = ollama_client.create_completion_function()
                response = completion_fn(prompt)
            except Exception as e:
                response = f"I encountered an error while generating a response: {e}"
        else:
            response = "I couldn't find specific information about that in the research papers. Could you try another question or check if the papers contain information on this topic?"
        
        # Display the response
        print(f"\n🤖 Response: {response}\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Research Assistant using Agentic IR")
    parser.add_argument("--model", type=str, default="deepseek-r1:14b",
                        help="Ollama model name to use")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    args = parser.parse_args()
    
    try:
        # Set up the agent
        print("📝 Setting up the Research Assistant...")
        agent = setup_agent(model_name=args.model, verbose=args.verbose)
        
        # Run interactive session
        run_interactive_session(agent)
        
    except KeyboardInterrupt:
        print("\n\n🤖 Session interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main() 