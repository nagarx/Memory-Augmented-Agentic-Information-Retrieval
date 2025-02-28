"""
Policy module for Agentic IR.

This module implements the agent policy π(at|x(st)) from the Agentic IR paper.
The policy is responsible for deciding the next action to take given the current state.
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
from pydantic import BaseModel, Field
import random
import time
import json
import logging

from .state import InformationState
from .memory import Memory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Action(BaseModel):
    """
    Represents an action that the agent can take.
    
    An action is a specific operation that the agent can perform
    to change the state of the environment.
    """
    
    # Name of the action
    name: str = Field(..., description="Name of the action")
    
    # Parameters for the action
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the action")
    
    # Description of the action
    description: Optional[str] = Field(None, description="Description of what the action does")
    
    # Whether the action is valid in the current state
    is_valid: bool = Field(True, description="Whether the action is valid in the current state")
    
    # Any additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about this action")

class Policy:
    """
    Base class for agent policies.
    
    A policy is responsible for deciding the next action to take
    given the current state of the environment.
    """
    
    def select_action(self, state: InformationState, **kwargs) -> Action:
        """
        Select an action to take given the current state.
        
        Args:
            state: The current state
            **kwargs: Additional arguments for action selection
            
        Returns:
            The selected action
        """
        raise NotImplementedError("Subclasses must implement select_action")
    
    def update(self, state: InformationState, action: Action, next_state: InformationState, 
               reward: float, **kwargs) -> None:
        """
        Update the policy based on experience.
        
        Args:
            state: The state before the action
            action: The action taken
            next_state: The state after the action
            reward: The reward received
            **kwargs: Additional arguments for policy update
        """
        raise NotImplementedError("Subclasses must implement update")

class RandomPolicy(Policy):
    """
    A simple policy that selects actions randomly.
    
    This policy is useful for exploration or as a baseline.
    """
    
    def select_action(self, state: InformationState, **kwargs) -> Action:
        """
        Select a random action from the available actions.
        
        Args:
            state: The current state
            **kwargs: Additional arguments for action selection
            
        Returns:
            A randomly selected action
        """
        if not state.available_actions:
            raise ValueError("No available actions in the current state")
            
        action_name = random.choice(state.available_actions)
        return Action(name=action_name)
    
    def update(self, state: InformationState, action: Action, next_state: InformationState, 
               reward: float, **kwargs) -> None:
        """
        Update the policy based on experience (no-op for random policy).
        
        Args:
            state: The state before the action
            action: The action taken
            next_state: The state after the action
            reward: The reward received
            **kwargs: Additional arguments for policy update
        """
        # No-op for random policy
        pass

class LLMPolicy(Policy):
    """
    A policy that uses an LLM to select actions.
    
    This policy uses a language model to generate actions based on the current state,
    memory of past interactions, and potentially generated thoughts.
    """
    
    def __init__(self, llm_fn: Callable[[str], str], memory: Optional[Memory] = None, 
                 thought_generator = None, thought_manager = None, verbose: bool = False):
        """
        Initialize the LLMPolicy.
        
        Args:
            llm_fn: Function to call the LLM
            memory: Memory component for storing and retrieving past experiences
            thought_generator: ThoughtGenerator for generating thoughts
            thought_manager: ThoughtManager for storing and organizing thoughts
            verbose: Whether to print verbose output
        """
        self.llm_fn = llm_fn
        self.memory = memory
        self.thought_generator = thought_generator
        self.thought_manager = thought_manager
        self.verbose = verbose
    
    def select_action(self, state: InformationState, **kwargs) -> Action:
        """
        Select an action using the LLM.
        
        Args:
            state: The current state
            **kwargs: Additional arguments for action selection
            
        Returns:
            The selected action
        """
        thoughts = []
        
        # Generate thoughts if there's a thought generator and manager
        if self.thought_generator and self.thought_manager:
            if self.verbose:
                logger.info("Generating thoughts...")
            thoughts = self.thought_generator.generate_thoughts(state, llm_fn=self.llm_fn)
            for thought in thoughts:
                self.thought_manager.add_thought(thought)
        
        # Get relevant experiences from memory if available
        relevant_experiences = []
        if self.memory:
            if self.verbose:
                logger.info("Retrieving relevant experiences...")
            relevant_experiences = self.memory.get_relevant_experiences(state)
        
        # Generate prompt for LLM
        prompt = self._generate_action_prompt(state, thoughts, relevant_experiences)
        
        # Get response from LLM
        if self.verbose:
            logger.info("Generating action with LLM...")
        response = self.llm_fn(prompt)
        
        # Parse the response to extract action
        if self.verbose:
            logger.info(f"LLM response: {response[:100]}...")
        action = self._parse_action_response(response, state.available_actions)
        
        if self.verbose:
            logger.info(f"Selected action: {action.name}")
            
        return action
    
    def update(self, state: InformationState, action: Action, next_state: InformationState, 
               reward: float, **kwargs) -> None:
        """
        Update the policy based on experience (no explicit update for LLM policy).
        
        Args:
            state: The state before the action
            action: The action taken
            next_state: The state after the action
            reward: The reward received
            **kwargs: Additional arguments for policy update
        """
        # LLM policy doesn't explicitly update parameters, but we can store the experience in memory
        if self.memory:
            from .state import StateTransition
            transition = StateTransition(
                source_state_id=state.id,
                target_state_id=next_state.id,
                action=action.name,
                action_params=action.parameters,
                success=True,  # Assuming success since we don't know otherwise
                reward=reward,
                timestamp=time.time()
            )
            
            if self.verbose:
                logger.info(f"Storing transition with reward {reward}")
            
            self.memory.add_transition(transition)
    
    def _generate_action_prompt(self, state: InformationState, thoughts: List[Any], 
                               relevant_experiences: List[Tuple[Any, Any]]) -> str:
        """
        Generate a prompt for action selection.
        
        Args:
            state: The current state
            thoughts: List of generated thoughts
            relevant_experiences: List of relevant past experiences
            
        Returns:
            A prompt for the LLM
        """
        prompt = "You are an AI agent tasked with selecting the best action to take in the current situation.\n\n"
        
        prompt += f"Current state: {state.text}\n\n"
        
        if state.available_actions:
            prompt += "Available actions:\n"
            for action in state.available_actions:
                prompt += f"- {action}\n"
            prompt += "\n"
        
        if thoughts:
            prompt += "Your thoughts on the current situation:\n"
            for thought in thoughts:
                prompt += f"[{thought.type}] {thought.text}\n\n"
        
        if relevant_experiences:
            prompt += "Relevant past experiences:\n"
            for i, (exp_state, exp_transition) in enumerate(relevant_experiences[:3]):  # Limit to top 3 for brevity
                prompt += f"Experience {i+1}:\n"
                prompt += f"State: {exp_state.text}\n"
                prompt += f"Action taken: {exp_transition.action}\n"
                if exp_transition.action_params:
                    params_str = ", ".join([f"{k}: {v}" for k, v in exp_transition.action_params.items()])
                    prompt += f"Action parameters: {params_str}\n"
                prompt += f"Success: {exp_transition.success}\n"
                prompt += f"Reward: {exp_transition.reward}\n\n"
        
        prompt += "Based on the current state, your thoughts, and relevant past experiences, select the best action to take.\n"
        prompt += "Provide your response in the following JSON format:\n"
        prompt += "{\n"
        prompt += '  "action": "action_name",\n'
        prompt += '  "parameters": {"param1": "value1", "param2": "value2"},\n'
        prompt += '  "reasoning": "explanation of why this action was chosen"\n'
        prompt += "}\n"
        
        return prompt
    
    def _parse_action_response(self, response: str, available_actions: List[str]) -> Action:
        """
        Parse the LLM response to extract the selected action.
        
        Args:
            response: The LLM response
            available_actions: List of available action names
            
        Returns:
            The parsed action
        """
        # Try to extract JSON from the response
        try:
            # Find JSON part in the response
            response = response.strip()
            start_idx = response.find("{")
            end_idx = response.rfind("}")
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx+1]
                action_data = json.loads(json_str)
                
                action_name = action_data.get("action", "")
                parameters = action_data.get("parameters", {})
                reasoning = action_data.get("reasoning", "")
                
                # Validate action name
                if action_name in available_actions:
                    if self.verbose:
                        logger.info(f"Successfully parsed action '{action_name}' from LLM response")
                    return Action(
                        name=action_name,
                        parameters=parameters,
                        description=reasoning,
                        is_valid=True
                    )
                else:
                    # If the action is not available, try to find a close match
                    if available_actions:
                        action_name = available_actions[0]  # Default to first available action
                        logger.warning(f"Action '{action_data.get('action', '')}' not available, defaulting to '{action_name}'")
                        return Action(
                            name=action_name,
                            parameters=parameters,
                            description=f"Original action not available. {reasoning}",
                            is_valid=True
                        )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Error parsing JSON action: {e}")
            # If JSON parsing fails, try to extract action name directly
            for action_name in available_actions:
                if action_name in response:
                    logger.info(f"Extracted action '{action_name}' from non-JSON response")
                    return Action(
                        name=action_name,
                        parameters={},
                        description="Extracted from non-JSON response",
                        is_valid=True
                    )
        
        # If all else fails, return a default action
        if available_actions:
            default_action = available_actions[0]
            logger.warning(f"Parsing failed, defaulting to action '{default_action}'")
            return Action(
                name=default_action,
                parameters={},
                description="Default action selected due to parsing failure",
                is_valid=True
            )
        else:
            logger.error("No available actions and parsing failed")
            return Action(
                name="no_op",
                parameters={},
                description="No available actions",
                is_valid=False
            )

class HybridPolicy(Policy):
    """
    A hybrid policy that combines multiple policies.
    
    This policy can use different strategies for exploration vs. exploitation,
    or can combine the outputs of multiple policies.
    """
    
    def __init__(self, policies: Dict[str, Policy], default_policy: str):
        """
        Initialize the HybridPolicy.
        
        Args:
            policies: Dictionary mapping policy names to policy instances
            default_policy: The name of the default policy to use
        """
        self.policies = policies
        self.default_policy = default_policy
        self.weights = {name: 1.0 for name in policies}
    
    def select_action(self, state: InformationState, policy_name: Optional[str] = None, **kwargs) -> Action:
        """
        Select an action using one of the policies.
        
        Args:
            state: The current state
            policy_name: The name of the policy to use (default: self.default_policy)
            **kwargs: Additional arguments for action selection
            
        Returns:
            The selected action
        """
        if policy_name is None:
            policy_name = self.default_policy
            
        if policy_name not in self.policies:
            raise ValueError(f"Unknown policy: {policy_name}")
            
        return self.policies[policy_name].select_action(state, **kwargs)
    
    def update(self, state: InformationState, action: Action, next_state: InformationState, 
               reward: float, **kwargs) -> None:
        """
        Update all policies based on experience.
        
        Args:
            state: The state before the action
            action: The action taken
            next_state: The state after the action
            reward: The reward received
            **kwargs: Additional arguments for policy update
        """
        for policy in self.policies.values():
            policy.update(state, action, next_state, reward, **kwargs)
            
        # Optionally update weights based on performance
        if "update_weights" in kwargs and kwargs["update_weights"]:
            self._update_weights(state, action, next_state, reward)
    
    def _update_weights(self, state: InformationState, action: Action, next_state: InformationState, reward: float) -> None:
        """
        Update the weights of the policies based on performance.
        
        Args:
            state: The state before the action
            action: The action taken
            next_state: The state after the action
            reward: The reward received
        """
        # Simple implementation: increase weight for policies that would have selected the same action
        for name, policy in self.policies.items():
            hypothetical_action = policy.select_action(state)
            if hypothetical_action.name == action.name:
                self.weights[name] *= (1.0 + 0.1 * reward)  # Increase weight if reward is positive
            else:
                self.weights[name] *= (1.0 - 0.05 * reward)  # Decrease weight if reward is positive
                
            # Ensure weights stay positive
            self.weights[name] = max(0.1, self.weights[name]) 