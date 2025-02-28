"""
Thought module for Agentic IR.

This module implements the thought (THT) component from the Agentic IR paper.
Thoughts represent the agent's reasoning and planning processes.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from pydantic import BaseModel, Field
import uuid
import time
import logging

from .state import InformationState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Thought(BaseModel):
    """
    Represents a thought in the agent's reasoning process.
    
    A thought captures a step in the agent's internal reasoning or planning.
    """
    
    # Unique identifier for the thought
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this thought")
    
    # The text content of the thought
    text: str = Field(..., description="The textual content of the thought")
    
    # The type of thought (e.g., analyze, plan, decide)
    type: str = Field(..., description="The type of thought")
    
    # The state ID that this thought is about
    state_id: str = Field(..., description="The ID of the state this thought is about")
    
    # Parent thought ID (for thought chains)
    parent_id: Optional[str] = Field(None, description="The ID of the parent thought, if any")
    
    # Timestamp when the thought was created
    timestamp: float = Field(default_factory=time.time, description="Timestamp when this thought was created")
    
    # Any additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about this thought")

class ThoughtManager:
    """
    Manages the agent's thoughts.
    
    The ThoughtManager stores, organizes, and retrieves thoughts.
    """
    
    def __init__(self):
        """Initialize the ThoughtManager."""
        self.thoughts: Dict[str, Thought] = {}
        self.thought_chains: Dict[str, List[str]] = {}  # state_id -> list of thought IDs
    
    def add_thought(self, thought: Thought) -> None:
        """
        Add a thought to the manager.
        
        Args:
            thought: The thought to add
        """
        self.thoughts[thought.id] = thought
        
        # Add to the appropriate thought chain
        if thought.state_id not in self.thought_chains:
            self.thought_chains[thought.state_id] = []
        self.thought_chains[thought.state_id].append(thought.id)
    
    def get_thought(self, thought_id: str) -> Optional[Thought]:
        """
        Retrieve a thought by ID.
        
        Args:
            thought_id: The ID of the thought to retrieve
            
        Returns:
            The thought if found, None otherwise
        """
        return self.thoughts.get(thought_id)
    
    def get_thoughts_for_state(self, state_id: str) -> List[Thought]:
        """
        Retrieve all thoughts for a specific state.
        
        Args:
            state_id: The ID of the state
            
        Returns:
            A list of thoughts for the state
        """
        thought_ids = self.thought_chains.get(state_id, [])
        return [self.thoughts[tid] for tid in thought_ids if tid in self.thoughts]
    
    def get_thought_chain(self, starting_thought_id: str) -> List[Thought]:
        """
        Retrieve a chain of thoughts starting from a given thought.
        
        Args:
            starting_thought_id: The ID of the starting thought
            
        Returns:
            A list of thoughts forming a chain
        """
        result = []
        current_id = starting_thought_id
        
        while current_id is not None and current_id in self.thoughts:
            thought = self.thoughts[current_id]
            result.append(thought)
            current_id = thought.parent_id
            
        return result[::-1]  # Reverse to get chronological order
    
    def clear(self) -> None:
        """Clear all thoughts."""
        self.thoughts.clear()
        self.thought_chains.clear()

class ThoughtGenerator:
    """
    Base class for thought generation.
    
    A ThoughtGenerator is responsible for generating thoughts based on the current state.
    """
    
    def generate_thoughts(self, state: InformationState, **kwargs) -> List[Thought]:
        """
        Generate thoughts based on the current state.
        
        Args:
            state: The current state
            **kwargs: Additional arguments for thought generation
            
        Returns:
            A list of generated thoughts
        """
        raise NotImplementedError("Subclasses must implement generate_thoughts")

class ChainOfThoughtGenerator(ThoughtGenerator):
    """
    Generates a chain of thoughts using an LLM.
    
    This generator produces a series of connected thoughts representing
    the agent's reasoning process.
    """
    
    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None, thought_steps: Optional[List[str]] = None, 
                max_thoughts: int = 3, verbose: bool = False):
        """
        Initialize the ChainOfThoughtGenerator.
        
        Args:
            llm_fn: Function to call the LLM (can be set later)
            thought_steps: Types of thoughts to generate (e.g., ["analyze", "plan", "decide"])
            max_thoughts: Maximum number of thoughts to generate
            verbose: Whether to print verbose output
        """
        self.llm_fn = llm_fn
        self.thought_steps = thought_steps or ["analyze", "plan", "decide"]
        self.max_thoughts = max_thoughts
        self.verbose = verbose
    
    def generate_thoughts(self, state: InformationState, llm_fn: Optional[Callable[[str], str]] = None, **kwargs) -> List[Thought]:
        """
        Generate a chain of thoughts based on the current state.
        
        Args:
            state: The current state
            llm_fn: Function to call the LLM (overrides the one set in __init__)
            **kwargs: Additional arguments for thought generation
            
        Returns:
            A list of generated thoughts forming a chain
        """
        # Use the provided llm_fn or the one from initialization
        fn = llm_fn or self.llm_fn
        if fn is None:
            raise ValueError("No LLM function provided for thought generation")
        
        thoughts = []
        parent_id = None
        
        # Generate thoughts for each step
        for i, step in enumerate(self.thought_steps[:self.max_thoughts]):
            # Generate prompt for this step
            prompt = self._generate_step_prompt(step, state, thoughts)
            
            if self.verbose:
                logger.info(f"Generating thought of type '{step}'")
            
            # Get response from LLM
            response = fn(prompt)
            
            # Create thought
            thought = Thought(
                text=response,
                type=step,
                state_id=state.id,
                parent_id=parent_id
            )
            
            thoughts.append(thought)
            parent_id = thought.id
            
            if self.verbose:
                logger.info(f"Generated thought: {response[:50]}...")
        
        return thoughts
    
    def _generate_step_prompt(self, step: str, state: InformationState, previous_thoughts: List[Thought]) -> str:
        """
        Generate a prompt for a specific thought step.
        
        Args:
            step: The type of thought step
            state: The current state
            previous_thoughts: Previous thoughts in the chain
            
        Returns:
            A prompt for the LLM
        """
        # Base prompt with state information
        prompt = f"Current Information State: {state.text}\n\n"
        
        # Add available actions if any
        if state.available_actions:
            prompt += f"Available Actions: {', '.join(state.available_actions)}\n\n"
        
        # Add previous thoughts
        if previous_thoughts:
            prompt += "Previous thoughts:\n"
            for i, thought in enumerate(previous_thoughts):
                prompt += f"[{thought.type}] {thought.text}\n\n"
        
        # Add specific instructions based on the step
        if step == "analyze":
            prompt += "Analyze the current situation. What is the information provided and what is the goal?"
        elif step == "plan":
            prompt += "Based on the analysis, what plan can be formulated to reach the goal? What steps need to be taken?"
        elif step == "decide":
            prompt += "Based on the analysis and plan, what action should be taken next? Provide a clear decision."
        elif step == "reflect":
            prompt += "Reflect on the current state and the process so far. What has been learned and what could be improved?"
        else:
            prompt += f"Generate a thought of type '{step}' based on the current state and previous thoughts."
            
        return prompt 