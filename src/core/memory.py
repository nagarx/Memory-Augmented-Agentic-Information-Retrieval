"""
Memory module for Agentic IR.

This module implements the memory component of the Agentic IR framework,
which is responsible for storing and retrieving information states and transitions.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from abc import ABC, abstractmethod

from .state import InformationState, StateTransition

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Memory(ABC):
    """
    Abstract base class for memory implementations.
    
    Memory is responsible for storing and retrieving information states
    and transitions between states.
    """
    
    @abstractmethod
    def add_state(self, state: InformationState) -> None:
        """
        Add a state to memory.
        
        Args:
            state: The state to add
        """
        pass
    
    @abstractmethod
    def add_transition(self, transition: StateTransition) -> None:
        """
        Add a transition to memory.
        
        Args:
            transition: The transition to add
        """
        pass
    
    @abstractmethod
    def get_state(self, state_id: str) -> Optional[InformationState]:
        """
        Retrieve a state by ID.
        
        Args:
            state_id: The ID of the state to retrieve
            
        Returns:
            The state if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_transition(self, source_id: str, target_id: str) -> Optional[StateTransition]:
        """
        Retrieve a transition by source and target state IDs.
        
        Args:
            source_id: The ID of the source state
            target_id: The ID of the target state
            
        Returns:
            The transition if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_transitions_from(self, state_id: str) -> List[StateTransition]:
        """
        Retrieve all transitions from a given state.
        
        Args:
            state_id: The ID of the source state
            
        Returns:
            A list of transitions
        """
        pass
    
    @abstractmethod
    def get_transitions_to(self, state_id: str) -> List[StateTransition]:
        """
        Retrieve all transitions to a given state.
        
        Args:
            state_id: The ID of the target state
            
        Returns:
            A list of transitions
        """
        pass
    
    @abstractmethod
    def get_state_history(self, state_id: str) -> List[InformationState]:
        """
        Retrieve the history of states leading to the given state.
        
        Args:
            state_id: The ID of the state
            
        Returns:
            A list of states in chronological order
        """
        pass
    
    @abstractmethod
    def search_states(self, query: str, limit: int = 5) -> List[InformationState]:
        """
        Search for states matching the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            A list of matching states
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all states and transitions from memory.
        """
        pass

class InMemoryStorage(Memory):
    """
    In-memory implementation of the Memory interface.
    
    This implementation stores all states and transitions in memory.
    """
    
    def __init__(self):
        """Initialize the in-memory storage."""
        self.states: Dict[str, InformationState] = {}
        self.transitions: Dict[Tuple[str, str], StateTransition] = {}
        self.transitions_from: Dict[str, Set[str]] = {}
        self.transitions_to: Dict[str, Set[str]] = {}
        self.state_transitions: Dict[str, List[str]] = {}  # State ID -> List of transition IDs
        logger.info("Initialized InMemoryStorage")
    
    def add_state(self, state: InformationState) -> None:
        """
        Add a state to memory.
        
        Args:
            state: The state to add
        """
        self.states[state.id] = state
        logger.debug(f"Added state {state.id} to memory")
    
    def add_transition(self, transition: StateTransition) -> None:
        """
        Add a transition to memory.
        
        Args:
            transition: The transition to add
        """
        key = (transition.source_state_id, transition.target_state_id)
        self.transitions[key] = transition
        
        # Update the from/to mappings
        if transition.source_state_id not in self.transitions_from:
            self.transitions_from[transition.source_state_id] = set()
        self.transitions_from[transition.source_state_id].add(transition.target_state_id)
        
        if transition.target_state_id not in self.transitions_to:
            self.transitions_to[transition.target_state_id] = set()
        self.transitions_to[transition.target_state_id].add(transition.source_state_id)
        
        # Generate a unique ID for the transition
        transition_id = f"{transition.source_state_id}_{transition.target_state_id}_{transition.timestamp}"
        
        # Store the transition
        self.state_transitions[transition.source_state_id] = self.state_transitions.get(transition.source_state_id, [])
        self.state_transitions[transition.source_state_id].append(transition_id)
        
        logger.debug(f"Added transition from {transition.source_state_id} to {transition.target_state_id}")
    
    def get_state(self, state_id: str) -> Optional[InformationState]:
        """
        Retrieve a state by ID.
        
        Args:
            state_id: The ID of the state to retrieve
            
        Returns:
            The state if found, None otherwise
        """
        return self.states.get(state_id)
    
    def get_transition(self, source_id: str, target_id: str) -> Optional[StateTransition]:
        """
        Retrieve a transition by source and target state IDs.
        
        Args:
            source_id: The ID of the source state
            target_id: The ID of the target state
            
        Returns:
            The transition if found, None otherwise
        """
        key = (source_id, target_id)
        return self.transitions.get(key)
    
    def get_transitions_from(self, state_id: str) -> List[StateTransition]:
        """
        Retrieve all transitions from a given state.
        
        Args:
            state_id: The ID of the source state
            
        Returns:
            A list of transitions
        """
        if state_id not in self.transitions_from:
            return []
        
        result = []
        for target_id in self.transitions_from[state_id]:
            transition = self.get_transition(state_id, target_id)
            if transition:
                result.append(transition)
        
        return result
    
    def get_transitions_to(self, state_id: str) -> List[StateTransition]:
        """
        Retrieve all transitions to a given state.
        
        Args:
            state_id: The ID of the target state
            
        Returns:
            A list of transitions
        """
        if state_id not in self.transitions_to:
            return []
        
        result = []
        for source_id in self.transitions_to[state_id]:
            transition = self.get_transition(source_id, state_id)
            if transition:
                result.append(transition)
        
        return result
    
    def get_state_history(self, state_id: str) -> List[InformationState]:
        """
        Retrieve the history of states leading to the given state.
        
        Args:
            state_id: The ID of the state
            
        Returns:
            A list of states in chronological order
        """
        result = []
        current_id = state_id
        
        while current_id:
            state = self.get_state(current_id)
            if not state:
                break
                
            result.insert(0, state)  # Insert at the beginning to maintain chronological order
            current_id = state.parent_id
        
        return result
    
    def search_states(self, query: str, limit: int = 5) -> List[InformationState]:
        """
        Search for states matching the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            A list of matching states
        """
        # Simple implementation: search for the query in the state text
        query_lower = query.lower()
        matches = []
        
        for state in self.states.values():
            if query_lower in state.text.lower():
                matches.append(state)
                if len(matches) >= limit:
                    break
        
        return matches
    
    def clear(self) -> None:
        """
        Clear all states and transitions from memory.
        """
        self.states.clear()
        self.transitions.clear()
        self.transitions_from.clear()
        self.transitions_to.clear()
        self.state_transitions.clear()
        logger.info("Cleared memory")
    
    def get_relevant_experiences(self, state: InformationState, max_results: int = 3) -> List[Tuple[InformationState, StateTransition]]:
        """
        Get relevant past experiences for a state.
        
        Args:
            state: The current state
            max_results: Maximum number of results to return
            
        Returns:
            A list of (state, transition) tuples
        """
        # Simple implementation: just return the most recent transitions
        results = []
        
        # Sort states by timestamp (most recent first)
        sorted_states = sorted(
            [s for s in self.states.values() if s.id != state.id],
            key=lambda s: s.timestamp if s.timestamp else 0,
            reverse=True
        )
        
        # Get the most recent states and their transitions
        for past_state in sorted_states[:max_results]:
            transitions = self.get_transitions_from(past_state.id)
            if transitions:
                # Use the first transition for simplicity
                results.append((past_state, transitions[0]))
        
        return results

class PersistentMemory(Memory):
    """
    Persistent implementation of the Memory interface.
    
    This implementation stores states and transitions on disk.
    """
    
    def __init__(self, storage_path: str):
        """
        Initialize the persistent storage.
        
        Args:
            storage_path: Path to the storage directory
        """
        self.storage_path = storage_path
        self.in_memory = InMemoryStorage()  # Use in-memory storage as a cache
        logger.info(f"Initialized PersistentMemory at {storage_path}")
    
    def add_state(self, state: InformationState) -> None:
        """
        Add a state to memory.
        
        Args:
            state: The state to add
        """
        self.in_memory.add_state(state)
        self._save_state(state)
    
    def add_transition(self, transition: StateTransition) -> None:
        """
        Add a transition to memory.
        
        Args:
            transition: The transition to add
        """
        self.in_memory.add_transition(transition)
        self._save_transition(transition)
    
    def get_state(self, state_id: str) -> Optional[InformationState]:
        """
        Retrieve a state by ID.
        
        Args:
            state_id: The ID of the state to retrieve
            
        Returns:
            The state if found, None otherwise
        """
        state = self.in_memory.get_state(state_id)
        if state:
            return state
            
        # Try to load from disk
        state = self._load_state(state_id)
        if state:
            self.in_memory.add_state(state)
        
        return state
    
    def get_transition(self, source_id: str, target_id: str) -> Optional[StateTransition]:
        """
        Retrieve a transition by source and target state IDs.
        
        Args:
            source_id: The ID of the source state
            target_id: The ID of the target state
            
        Returns:
            The transition if found, None otherwise
        """
        transition = self.in_memory.get_transition(source_id, target_id)
        if transition:
            return transition
            
        # Try to load from disk
        transition = self._load_transition(source_id, target_id)
        if transition:
            self.in_memory.add_transition(transition)
        
        return transition
    
    def get_transitions_from(self, state_id: str) -> List[StateTransition]:
        """
        Retrieve all transitions from a given state.
        
        Args:
            state_id: The ID of the source state
            
        Returns:
            A list of transitions
        """
        # For simplicity, we'll just use the in-memory implementation
        # In a real implementation, you would need to ensure all transitions are loaded
        return self.in_memory.get_transitions_from(state_id)
    
    def get_transitions_to(self, state_id: str) -> List[StateTransition]:
        """
        Retrieve all transitions to a given state.
        
        Args:
            state_id: The ID of the target state
            
        Returns:
            A list of transitions
        """
        # For simplicity, we'll just use the in-memory implementation
        return self.in_memory.get_transitions_to(state_id)
    
    def get_state_history(self, state_id: str) -> List[InformationState]:
        """
        Retrieve the history of states leading to the given state.
        
        Args:
            state_id: The ID of the state
            
        Returns:
            A list of states in chronological order
        """
        # For simplicity, we'll just use the in-memory implementation
        return self.in_memory.get_state_history(state_id)
    
    def search_states(self, query: str, limit: int = 5) -> List[InformationState]:
        """
        Search for states matching the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            A list of matching states
        """
        # For simplicity, we'll just use the in-memory implementation
        return self.in_memory.search_states(query, limit)
    
    def clear(self) -> None:
        """
        Clear all states and transitions from memory.
        """
        self.in_memory.clear()
        # In a real implementation, you would also clear the on-disk storage
    
    def _save_state(self, state: InformationState) -> None:
        """
        Save a state to disk.
        
        Args:
            state: The state to save
        """
        # In a real implementation, you would save the state to disk
        # For example, using JSON serialization
        pass
    
    def _save_transition(self, transition: StateTransition) -> None:
        """
        Save a transition to disk.
        
        Args:
            transition: The transition to save
        """
        # In a real implementation, you would save the transition to disk
        pass
    
    def _load_state(self, state_id: str) -> Optional[InformationState]:
        """
        Load a state from disk.
        
        Args:
            state_id: The ID of the state to load
            
        Returns:
            The state if found, None otherwise
        """
        # In a real implementation, you would load the state from disk
        return None
    
    def _load_transition(self, source_id: str, target_id: str) -> Optional[StateTransition]:
        """
        Load a transition from disk.
        
        Args:
            source_id: The ID of the source state
            target_id: The ID of the target state
            
        Returns:
            The transition if found, None otherwise
        """
        # In a real implementation, you would load the transition from disk
        return None 