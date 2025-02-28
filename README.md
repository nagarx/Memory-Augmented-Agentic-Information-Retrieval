# Agentic Information Retrieval

An implementation of the Agentic Information Retrieval (Agentic IR) framework based on the paper by Weinan Zhang, Junwei Liao, Ning Li, and Kounianhua Du.

## What is Agentic IR?

Agentic Information Retrieval (Agentic IR) is a novel IR paradigm shaped by the capabilities of LLM agents. Unlike traditional IR systems that filter predefined sets of candidate items, Agentic IR expands the scope of accessible tasks and leverages a suite of new techniques to redefine information retrieval.

The key components of Agentic IR include:

- An agent with memory (MEM), thought (THT), and tools (TOOL)
- A recurrent cycle of observation, reasoning, and action
- Ability to transition between information states to reach the user's target state

## Features

- **Modular Architecture**: Core components are designed to be easily extended and customized
- **State Management**: Representation and tracking of information states
- **Memory System**: Storage and retrieval of states and transitions
- **Thought Generation**: Chain-of-thought reasoning for decision making
- **Policy Learning**: Action selection based on current state
- **Reward Modeling**: Evaluation of actions and states
- **Tool Integration**: Framework for adding custom tools
- **Environment Support**: Interface for creating custom environments
- **Local LLM Integration**: Support for Ollama-based local language models

## Applications

This implementation includes examples of Agentic IR for three types of applications:

1. **Life Assistant**: Proactive, context-aware personal assistance
2. **Business Assistant**: Document retrieval and information integration
3. **Coding Assistant**: Programming help, debugging, and documentation

## Getting Started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) for local LLM inference (with deepseek-r1:14b model installed)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/agentic-ir.git
cd agentic-ir

# Install dependencies
pip install -e .
```

### Running Examples

```bash
# Run the life assistant example
python -m src.examples.life_assistant_example --model deepseek-r1:14b

# Run the research assistant example (for question answering with research papers)
python -m src.examples.research_assistant_example --model deepseek-r1:14b

# Run the business assistant example (coming soon)
# python -m src.examples.business_assistant_example

# Run the coding assistant example (coming soon)
# python -m src.examples.coding_assistant_example
```

### Research Papers Repository

The research assistant example uses markdown files in the `data/research_papers` directory to answer questions. To use your own research papers:

1. Convert papers to markdown format
2. Place them in the `data/research_papers` directory
3. Run the research assistant example
4. Ask questions about the papers or use commands like "explain paper 1"

## Usage

### Basic Example

```python
from core.agent import Agent
from core.memory import InMemoryStorage
from core.thought import ChainOfThoughtGenerator, ThoughtManager
from core.reward import SimpleRewardModel
from core.policy import LLMPolicy
from llm.ollama import create_ollama_client, create_completion_function
from environments.life_assistant import LifeAssistantEnvironment
from tools.search import WebSearchTool, WebContentTool

# Set up LLM client
ollama_client = create_ollama_client(model_name="deepseek-r1:14b")
completion_fn = ollama_client.create_completion_function()

# Set up agent components
memory = InMemoryStorage()
thought_manager = ThoughtManager()
thought_generator = ChainOfThoughtGenerator(llm_fn=completion_fn)
reward_model = SimpleRewardModel()
policy = LLMPolicy(llm_fn=completion_fn, thought_generator=thought_generator, memory=memory)

# Create agent
agent = Agent(
    memory=memory,
    thought_generator=thought_generator,
    thought_manager=thought_manager,
    policy=policy,
    reward_model=reward_model
)

# Add tools
agent.add_tool(WebSearchTool())
agent.add_tool(WebContentTool())

# Create environment
env = LifeAssistantEnvironment()

# Run the agent
initial_state = env.reset()
agent.set_current_state(initial_state)
result = agent.run(env, max_steps=10)
print(result.final_state.text)
```

## Project Structure

- `src/core/`: Core agent implementation components
- `src/tools/`: Tools the agent can use
- `src/methods/`: Implementation of key methods (RAG, reflection, etc.)
- `src/environments/`: Environment interfaces
- `src/llm/`: LLM integration
- `src/applications/`: Application-specific implementations
- `src/examples/`: Example applications
- `config/`: Configuration files
- `tests/`: Unit tests

## Extending the Framework

### Creating Custom Tools

```python
from tools.base import BaseTool, ToolResult

class MyCustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_custom_tool",
            description="Description of my custom tool",
            parameters=[
                {
                    "name": "param1",
                    "description": "Description of param1",
                    "type": "string",
                    "required": True
                }
            ]
        )
    
    def _execute(self, param1: str) -> ToolResult:
        # Implement your tool logic here
        return ToolResult(
            success=True,
            result={"output": f"Processed {param1}"}
        )
```

### Creating Custom Environments

```python
from core.agent import Environment
from core.policy import Action
from core.state import InformationState

class MyCustomEnvironment(Environment):
    def reset(self) -> InformationState:
        # Initialize and return the starting state
        return InformationState(...)
    
    def step(self, action: Action) -> Tuple[InformationState, float, bool, Dict[str, Any]]:
        # Process the action and return the next state, reward, done flag, and info
        return next_state, reward, done, info
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original paper: "Agentic Information Retrieval" by Weinan Zhang, Junwei Liao, Ning Li, and Kounianhua Du
- Ollama project for providing local LLM inference capabilities 