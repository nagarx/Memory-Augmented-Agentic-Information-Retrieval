# Agentic IR Project Structure

This document provides an overview of the Agentic IR project structure and organization.

## Directory Structure

```
agentic-ir/
├── config/                 # Configuration files
├── data/                   # Data files
│   └── research_papers/    # Markdown-formatted research papers for the research assistant
├── LICENSE                 # MIT License
├── README.md               # Project README with overview and usage instructions
├── requirements.txt        # Python dependencies
├── setup.py                # Python package setup
├── src/                    # Source code
│   ├── applications/       # Application-specific implementations
│   ├── core/               # Core agent components
│   │   ├── agent.py        # Main agent implementation
│   │   ├── memory.py       # Memory/state storage
│   │   ├── policy.py       # Action selection policies
│   │   ├── reward.py       # Reward modeling
│   │   ├── state.py        # Information state representation
│   │   └── thought.py      # Thought generation and management
│   ├── environments/       # Environment implementations
│   │   └── life_assistant.py  # Life assistant environment
│   ├── examples/           # Example applications
│   │   ├── life_assistant_example.py  # Life assistant demo
│   │   └── research_assistant_example.py  # Research paper assistant
│   ├── llm/                # LLM integration
│   │   └── ollama.py       # Ollama client for local LLM inference
│   ├── methods/            # Implementation of key methods (RAG, reflection, etc.)
│   └── tools/              # Tools the agent can use
│       ├── base.py         # Base tool interface
│       ├── document_retrieval.py  # Tools for research paper retrieval
│       └── search.py       # Web search tools
└── tests/                  # Unit and integration tests
```

## Core Components

### Agent (`src/core/agent.py`)

The main implementation of the Agentic IR agent. This integrates memory, thought generation, policy, and tools to enable the agent to interact with the environment.

### Memory (`src/core/memory.py`)

Storage for states and transitions, enabling the agent to remember past experiences and retrieve relevant information.

### Policy (`src/core/policy.py`)

Responsible for selecting actions based on the current state. Includes random, LLM-based, and hybrid policies.

### Reward (`src/core/reward.py`)

Implements reward modeling for evaluating states and transitions, which can be used for reinforcement learning or evaluation.

### State (`src/core/state.py`)

Defines the information state representation, which is the core data structure of the Agentic IR framework.

### Thought (`src/core/thought.py`)

Handles thought generation and management, enabling the agent to reason about the current state.

## Tools

### Base Tool (`src/tools/base.py`)

Defines the base tool interface that all tools must implement.

### Document Retrieval (`src/tools/document_retrieval.py`)

Tools for searching and retrieving information from research papers.

### Search (`src/tools/search.py`)

Web search and content retrieval tools.

## Environments

### Life Assistant (`src/environments/life_assistant.py`)

An environment for a life assistant application that can help with weather, calendar, restaurants, etc.

## Examples

### Life Assistant Example (`src/examples/life_assistant_example.py`)

Demonstrates using the Agentic IR framework to build a life assistant.

### Research Assistant Example (`src/examples/research_assistant_example.py`)

Demonstrates using the Agentic IR framework to build a research paper assistant that can answer questions based on document content.

## Data

### Research Papers (`data/research_papers/`)

Contains Markdown-formatted research papers for the research assistant example. 