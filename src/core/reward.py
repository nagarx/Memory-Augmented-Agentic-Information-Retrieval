"""
Reward module for Agentic IR.

This module implements reward modeling and evaluation for the Agentic IR framework.
Rewards are used to evaluate the success of the agent's actions and for reinforcement learning.
"""

from typing import Dict, Any, Callable, Optional, List, Tuple
from pydantic import BaseModel, Field
import time

from .state import InformationState, StateTransition

# Type definitions
RewardFunction = Callable[[InformationState, InformationState, Optional[StateTransition]], float]

class RewardConfig(BaseModel):
    """
    Configuration for reward calculation.
    
    This defines the parameters and weights for reward computation.
    """
    
    # Success/target state weight
    target_state_weight: float = Field(1.0, description="Weight for target state similarity reward")
    
    # Step cost weight (to encourage efficiency)
    step_cost_weight: float = Field(0.1, description="Weight for step cost penalty")
    
    # Time cost weight (to encourage speed)
    time_cost_weight: float = Field(0.01, description="Weight for time cost penalty")
    
    # Custom reward weights
    custom_rewards: Dict[str, float] = Field(default_factory=dict, description="Weights for custom reward components")

class RewardEvent(BaseModel):
    """
    Represents a reward event in the agent's history.
    
    A reward event records the reward given for a state or transition,
    along with metadata about how it was calculated.
    """
    
    # State or transition ID
    state_id: Optional[str] = Field(None, description="ID of the state being rewarded, if any")
    transition_id: Optional[str] = Field(None, description="ID of the transition being rewarded, if any")
    
    # Reward value
    value: float = Field(..., description="Reward value")
    
    # Components of the reward
    components: Dict[str, float] = Field(default_factory=dict, description="Components of the reward value")
    
    # Timestamp
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp when this reward was calculated")
    
    # Any additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about this reward event")

class RewardModel:
    """
    Base class for reward models.
    
    A reward model computes rewards for states and transitions,
    which can be used for reinforcement learning or evaluation.
    """
    
    def calculate_reward(self, current_state: InformationState, target_state: InformationState, 
                         transition: Optional[StateTransition] = None) -> RewardEvent:
        """
        Calculate reward for a state or transition.
        
        Args:
            current_state: The current state
            target_state: The target state to compare with
            transition: The transition to calculate reward for, if any
            
        Returns:
            A RewardEvent with the calculated reward
        """
        raise NotImplementedError("Subclasses must implement calculate_reward")
    
    def calculate_cumulative_reward(self, states: List[InformationState], 
                                   transitions: List[StateTransition], 
                                   target_state: InformationState) -> float:
        """
        Calculate cumulative reward for a sequence of states and transitions.
        
        Args:
            states: The sequence of states
            transitions: The sequence of transitions
            target_state: The target state
            
        Returns:
            The cumulative reward
        """
        raise NotImplementedError("Subclasses must implement calculate_cumulative_reward")

class SimpleRewardModel(RewardModel):
    """
    A simple reward model for Agentic IR.
    
    This model calculates rewards based on similarity to the target state,
    step cost, and time cost.
    """
    
    def __init__(self, weights: Dict[str, float] = None, custom_reward_fns: Dict[str, RewardFunction] = None):
        """
        Initialize the SimpleRewardModel.
        
        Args:
            weights: Weights for different reward components
            custom_reward_fns: Custom reward functions to include
        """
        self.weights = weights or {"similarity": 0.7, "step_cost": 0.1, "time_cost": 0.01}
        self.custom_reward_fns = custom_reward_fns or {}
    
    def calculate_reward(self, current_state: InformationState, target_state: InformationState, 
                         transition: Optional[StateTransition] = None) -> RewardEvent:
        """
        Calculate reward for a state or transition.
        
        Args:
            current_state: The current state
            target_state: The target state to compare with
            transition: The transition to calculate reward for, if any
            
        Returns:
            A RewardEvent with the calculated reward
        """
        reward_components = {}
        
        # Calculate similarity reward
        similarity = current_state.compare(target_state)
        target_reward = similarity * self.weights.get("similarity", 0.7)
        reward_components["target_similarity"] = target_reward
        
        # Apply step cost if there's a transition
        step_cost = 0.0
        if transition is not None:
            step_cost = -self.weights.get("step_cost", 0.1)
            reward_components["step_cost"] = step_cost
            
            # Apply time cost
            if transition.timestamp > 0:
                time_diff = time.time() - transition.timestamp
                time_cost = -time_diff * self.weights.get("time_cost", 0.01)
                reward_components["time_cost"] = time_cost
            else:
                time_cost = 0.0
        else:
            time_cost = 0.0
        
        # Apply custom rewards
        custom_reward_total = 0.0
        for name, fn in self.custom_reward_fns.items():
            if name in self.weights:
                weight = self.weights[name]
                value = fn(current_state, target_state, transition)
                weighted_value = value * weight
                reward_components[name] = weighted_value
                custom_reward_total += weighted_value
        
        # Calculate total reward
        total_reward = target_reward + step_cost + time_cost + custom_reward_total
        
        # Create reward event
        event = RewardEvent(
            state_id=current_state.id if transition is None else None,
            transition_id=transition.source_state_id + "_" + transition.target_state_id if transition is not None else None,
            value=total_reward,
            components=reward_components
        )
        
        return event
    
    def calculate_cumulative_reward(self, states: List[InformationState], 
                                   transitions: List[StateTransition], 
                                   target_state: InformationState) -> float:
        """
        Calculate cumulative reward for a sequence of states and transitions.
        
        Args:
            states: The sequence of states
            transitions: The sequence of transitions
            target_state: The target state
            
        Returns:
            The cumulative reward
        """
        # Ensure states and transitions have consistent length
        if len(states) != len(transitions) + 1:
            raise ValueError(f"Expected {len(transitions) + 1} states, got {len(states)}")
        
        total_reward = 0.0
        
        # Calculate rewards for each step
        for i in range(len(transitions)):
            current_state = states[i]
            next_state = states[i + 1]
            transition = transitions[i]
            
            # Calculate reward for transition
            event = self.calculate_reward(next_state, target_state, transition)
            total_reward += event.value
        
        # Add final state reward
        final_event = self.calculate_reward(states[-1], target_state)
        total_reward += final_event.value
        
        return total_reward

class LLMRewardModel(RewardModel):
    """
    A reward model that uses an LLM to calculate rewards.
    
    This model uses an LLM to evaluate states and transitions,
    which can be more flexible for complex tasks.
    """
    
    def __init__(self, llm_fn: Callable[[str], str], config: Optional[RewardConfig] = None):
        """
        Initialize the LLMRewardModel.
        
        Args:
            llm_fn: Function to call the LLM
            config: Configuration for reward calculation
        """
        self.llm_fn = llm_fn
        self.config = config or RewardConfig()
    
    def calculate_reward(self, current_state: InformationState, target_state: InformationState, 
                         transition: Optional[StateTransition] = None) -> RewardEvent:
        """
        Calculate reward for a state or transition using an LLM.
        
        Args:
            current_state: The current state
            target_state: The target state to compare with
            transition: The transition to calculate reward for, if any
            
        Returns:
            A RewardEvent with the calculated reward
        """
        # Generate prompt
        prompt = self._generate_reward_prompt(current_state, target_state, transition)
        
        # Get response from LLM
        response = self.llm_fn(prompt)
        
        # Parse the response to extract reward and components
        reward, components = self._parse_llm_response(response)
        
        # Create reward event
        event = RewardEvent(
            state_id=current_state.id if transition is None else None,
            transition_id=transition.source_state_id + "_" + transition.target_state_id if transition is not None else None,
            value=reward,
            components=components
        )
        
        return event
    
    def calculate_cumulative_reward(self, states: List[InformationState], 
                                   transitions: List[StateTransition], 
                                   target_state: InformationState) -> float:
        """
        Calculate cumulative reward for a sequence of states and transitions using an LLM.
        
        Args:
            states: The sequence of states
            transitions: The sequence of transitions
            target_state: The target state
            
        Returns:
            The cumulative reward
        """
        # Generate prompt for cumulative reward
        prompt = self._generate_cumulative_reward_prompt(states, transitions, target_state)
        
        # Get response from LLM
        response = self.llm_fn(prompt)
        
        # Parse the response to extract cumulative reward
        cumulative_reward = self._parse_cumulative_reward_response(response)
        
        return cumulative_reward
    
    def _generate_reward_prompt(self, current_state: InformationState, target_state: InformationState, 
                               transition: Optional[StateTransition] = None) -> str:
        """
        Generate a prompt for calculating reward.
        
        Args:
            current_state: The current state
            target_state: The target state
            transition: The transition, if any
            
        Returns:
            The prompt for the LLM
        """
        prompt = "You are an evaluation system for an AI agent. Please evaluate the current state compared to the target state and assign a reward.\n\n"
        
        prompt += f"Target state: {target_state.text}\n\n"
        prompt += f"Current state: {current_state.text}\n\n"
        
        if transition is not None:
            prompt += f"Action taken: {transition.action}\n"
            if transition.action_params:
                params_str = ", ".join([f"{k}: {v}" for k, v in transition.action_params.items()])
                prompt += f"Action parameters: {params_str}\n"
            prompt += "\n"
        
        prompt += "Evaluate how close the current state is to the target state. Provide a reward between -1 and 1, where:\n"
        prompt += "1.0 means the current state fully satisfies the target state.\n"
        prompt += "0.0 means the current state is neutral or unrelated to the target state.\n"
        prompt += "-1.0 means the current state is completely opposed to or harmful for reaching the target state.\n\n"
        
        prompt += "Please provide your evaluation in the following format:\n"
        prompt += "Reward: [numerical value between -1 and 1]\n"
        prompt += "Reasoning: [your reasoning]\n"
        prompt += "Components:\n"
        prompt += "- Similarity: [numerical value between 0 and 1]\n"
        prompt += "- Progress: [numerical value between -1 and 1]\n"
        prompt += "- Efficiency: [numerical value between -1 and 1]\n"
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Tuple[float, Dict[str, float]]:
        """
        Parse the LLM response to extract reward and components.
        
        Args:
            response: The LLM response
            
        Returns:
            A tuple of (reward, components)
        """
        lines = response.strip().split("\n")
        reward = 0.0
        components = {}
        
        for line in lines:
            line = line.strip()
            
            # Extract the main reward
            if line.startswith("Reward:"):
                try:
                    reward = float(line.split("Reward:")[1].strip())
                except (ValueError, IndexError):
                    pass
            
            # Extract components
            elif line.startswith("-") and ":" in line:
                try:
                    component_name = line.split("-")[1].split(":")[0].strip()
                    component_value = float(line.split(":")[1].strip())
                    components[component_name.lower()] = component_value
                except (ValueError, IndexError):
                    pass
        
        return reward, components
    
    def _generate_cumulative_reward_prompt(self, states: List[InformationState], 
                                          transitions: List[StateTransition], 
                                          target_state: InformationState) -> str:
        """
        Generate a prompt for calculating cumulative reward.
        
        Args:
            states: The sequence of states
            transitions: The sequence of transitions
            target_state: The target state
            
        Returns:
            The prompt for the LLM
        """
        prompt = "You are an evaluation system for an AI agent. Please evaluate the sequence of states and transitions and assign a cumulative reward.\n\n"
        
        prompt += f"Target state: {target_state.text}\n\n"
        prompt += f"Initial state: {states[0].text}\n\n"
        prompt += f"Final state: {states[-1].text}\n\n"
        
        prompt += "Sequence of actions:\n"
        for i, transition in enumerate(transitions):
            prompt += f"{i+1}. Action: {transition.action}\n"
            if transition.action_params:
                params_str = ", ".join([f"{k}: {v}" for k, v in transition.action_params.items()])
                prompt += f"   Parameters: {params_str}\n"
            prompt += f"   Result: {states[i+1].text}\n\n"
        
        prompt += "Evaluate how well the agent achieved the target state. Provide a cumulative reward between -1 and 1, where:\n"
        prompt += "1.0 means the agent fully achieved the target state efficiently.\n"
        prompt += "0.0 means the agent made no progress towards the target state.\n"
        prompt += "-1.0 means the agent moved away from or damaged the possibility of reaching the target state.\n\n"
        
        prompt += "Please provide your evaluation in the following format:\n"
        prompt += "Cumulative Reward: [numerical value between -1 and 1]\n"
        prompt += "Reasoning: [your reasoning]\n"
        
        return prompt
    
    def _parse_cumulative_reward_response(self, response: str) -> float:
        """
        Parse the LLM response to extract cumulative reward.
        
        Args:
            response: The LLM response
            
        Returns:
            The cumulative reward
        """
        lines = response.strip().split("\n")
        cumulative_reward = 0.0
        
        for line in lines:
            line = line.strip()
            
            # Extract the cumulative reward
            if line.startswith("Cumulative Reward:"):
                try:
                    cumulative_reward = float(line.split("Cumulative Reward:")[1].strip())
                except (ValueError, IndexError):
                    pass
                break
        
        return cumulative_reward 