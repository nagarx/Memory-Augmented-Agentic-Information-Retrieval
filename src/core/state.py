"""
Information State representation for Agentic IR.

This module defines the information state, which is a key concept in Agentic IR.
An information state represents the current state of information for the agent and user.
"""

from typing import Dict, Any, Optional, List, Callable
from pydantic import BaseModel, Field

class InformationState(BaseModel):
    """
    Represents an information state in the Agentic IR framework.
    
    An information state contains all the relevant information at a specific
    point in time during an agent's interaction with the environment.
    """
    
    # Unique identifier for the state
    id: str = Field(..., description="Unique identifier for this state")
    
    # Textual representation of the state
    text: str = Field(..., description="Textual representation of the state")
    
    # Structured data associated with the state
    data: Dict[str, Any] = Field(default_factory=dict, description="Structured data associated with this state")
    
    # Metadata about the state
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about this state")
    
    # Parent state (for tracking history)
    parent_id: Optional[str] = Field(None, description="ID of the parent state, if any")
    
    # Actions available in this state
    available_actions: List[str] = Field(default_factory=list, description="Actions available in this state")
    
    # Time when the state was created
    timestamp: float = Field(..., description="Unix timestamp when this state was created")
    
    def to_prompt(self) -> str:
        """
        Convert the state to a prompt that can be sent to the LLM.
        
        Returns:
            str: A string representation of the state for the LLM.
        """
        return f"Current Information State: {self.text}\n\nAvailable Actions: {', '.join(self.available_actions)}"
    
    def compare(self, target_state: "InformationState", similarity_fn: Optional[Callable] = None) -> float:
        """
        Compare this state with a target state to compute similarity.
        
        Args:
            target_state: The target state to compare with
            similarity_fn: Optional custom similarity function
            
        Returns:
            float: A value between 0 and 1 indicating similarity (1 = identical)
        """
        if similarity_fn is not None:
            return similarity_fn(self, target_state)
        
        # Default implementation: simple text similarity
        # In a real implementation, you would use more sophisticated methods
        common_words_self = set(self.text.lower().split())
        common_words_target = set(target_state.text.lower().split())
        
        if not common_words_self or not common_words_target:
            return 0.0
            
        overlap = len(common_words_self.intersection(common_words_target))
        union = len(common_words_self.union(common_words_target))
        
        return overlap / union

class StateTransition(BaseModel):
    """
    Represents a transition between two information states.
    
    A state transition occurs when an agent takes an action in
    an environment, changing the current information state.
    """
    
    # Source and target states
    source_state_id: str = Field(..., description="ID of the source state")
    target_state_id: str = Field(..., description="ID of the target state")
    
    # Action that caused the transition
    action: str = Field(..., description="Action that caused the transition")
    
    # Parameters for the action
    action_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the action")
    
    # Success/failure of the transition
    success: bool = Field(..., description="Whether the transition was successful")
    
    # Reward associated with the transition
    reward: float = Field(0.0, description="Reward associated with this transition")
    
    # Time when the transition occurred
    timestamp: float = Field(..., description="Unix timestamp when this transition occurred")
    
    # Any additional information about the transition
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about this transition") 