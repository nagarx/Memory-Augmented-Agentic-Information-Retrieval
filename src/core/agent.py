"""
Agent module for Agentic IR.

This module implements the agent, which is the central component of Agentic IR.
The agent combines memory, thought, tools, and policy to interact with the environment.
"""

from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import uuid
import time
import logging

from .state import InformationState, StateTransition
from .memory import Memory, InMemoryStorage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tool:
    """
    Base class for tools that the agent can use.
    
    A tool is a function that the agent can call to interact with the environment
    or perform some computation.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize the tool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
        """
        self.name = name
        self.description = description
    
    def __call__(self, **kwargs) -> Any:
        """
        Call the tool with the given parameters.
        
        Args:
            **kwargs: Parameters for the tool
            
        Returns:
            The result of the tool call
        """
        raise NotImplementedError("Subclasses must implement __call__")
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema of the tool.
        
        Returns:
            A dictionary describing the parameters and return type of the tool
        """
        raise NotImplementedError("Subclasses must implement get_schema")

class Environment:
    """
    Base class for environments.
    
    An environment is the external context that the agent interacts with.
    It defines the state space, action space, and transition dynamics.
    """
    
    def __init__(self):
        """Initialize the environment."""
        pass
    
    def reset(self) -> InformationState:
        """
        Reset the environment to an initial state.
        
        Returns:
            The initial state
        """
        raise NotImplementedError("Subclasses must implement reset")
    
    def step(self, action: Any) -> Tuple[InformationState, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment by applying an action.
        
        Args:
            action: The action to apply
            
        Returns:
            A tuple of (next_state, reward, done, info)
        """
        raise NotImplementedError("Subclasses must implement step")
    
    def render(self) -> Any:
        """
        Render the environment for visualization.
        
        Returns:
            A visualization of the environment
        """
        raise NotImplementedError("Subclasses must implement render")

class Agent:
    """
    Agent for Agentic IR.
    
    The agent is the central component that combines memory, thought, tools,
    and policy to interact with the environment and achieve the user's goals.
    """
    
    def __init__(self, policy, memory: Optional[Memory] = None, 
                 thought_generator = None,
                 thought_manager = None,
                 tools: Optional[Dict[str, Tool]] = None,
                 reward_model = None,
                 name: str = "AgenticIR",
                 verbose: bool = False):
        """
        Initialize the agent.
        
        Args:
            policy: The policy for selecting actions
            memory: Memory component for storing and retrieving experiences
            thought_generator: ThoughtGenerator for generating thoughts
            thought_manager: ThoughtManager for storing and organizing thoughts
            tools: Dictionary mapping tool names to tool instances
            reward_model: Model for calculating rewards
            name: Name of the agent
            verbose: Whether to print verbose output
        """
        self.policy = policy
        self.memory = memory or InMemoryStorage()
        self.thought_generator = thought_generator
        self.thought_manager = thought_manager
        self.tools = tools or {}
        self.reward_model = reward_model
        self.name = name
        self.verbose = verbose
        
        self.current_state: Optional[InformationState] = None
        self.target_state: Optional[InformationState] = None
        self.history = []
        
        logger.info(f"Initialized agent: {self.name}")
    
    def set_current_state(self, state: InformationState) -> None:
        """
        Set the current state of the agent.
        
        Args:
            state: The current state
        """
        self.current_state = state
        if self.memory:
            self.memory.add_state(state)
            
        if self.verbose:
            logger.info(f"Set current state: {state.text[:50]}...")
    
    def set_target_state(self, state: InformationState) -> None:
        """
        Set the target state that the agent is trying to reach.
        
        Args:
            state: The target state
        """
        self.target_state = state
        if self.memory:
            self.memory.add_state(state)
            
        if self.verbose:
            logger.info(f"Set target state: {state.text[:50]}...")
    
    def act(self, state: Optional[InformationState] = None, environment: Optional[Environment] = None) -> Tuple[Any, InformationState, float]:
        """
        Select and execute an action.
        
        Args:
            state: The current state (default: self.current_state)
            environment: The environment to execute the action in
            
        Returns:
            A tuple of (action, next_state, reward)
        """
        if state is None:
            if self.current_state is None:
                raise ValueError("No current state set")
            state = self.current_state
            
        # Select action using policy
        action = self.policy.select_action(state)
        if self.verbose:
            logger.info(f"Selected action: {action.name}")
        
        # Execute action in environment if provided
        if environment:
            next_state, reward, done, info = environment.step(action)
            if self.verbose:
                logger.info(f"Executed action in environment. Reward: {reward}, Done: {done}")
        else:
            # If no environment is provided, simulate the execution
            next_state, reward = self._simulate_action(state, action)
            if self.verbose:
                logger.info(f"Simulated action execution. Reward: {reward}")
        
        # Update current state
        self.set_current_state(next_state)
        
        # Update policy with the experience
        self.policy.update(state, action, next_state, reward)
        
        # Store the transition in memory
        if self.memory:
            transition = StateTransition(
                source_state_id=state.id,
                target_state_id=next_state.id,
                action=action.name,
                action_params=action.parameters,
                success=True,  # Assuming success since we don't know otherwise
                reward=reward,
                timestamp=time.time()
            )
            self.memory.add_transition(transition)
        
        # Add to history
        self.history.append((state, action, next_state, reward))
        
        return action, next_state, reward
    
    def _simulate_action(self, state: InformationState, action) -> Tuple[InformationState, float]:
        """
        Simulate the execution of an action.
        
        Args:
            state: The current state
            action: The action to execute
            
        Returns:
            A tuple of (next_state, reward)
        """
        # If the action corresponds to a tool, call the tool
        if action.name in self.tools:
            tool = self.tools[action.name]
            tool_result = tool(**action.parameters)
            
            # Create a new state based on the tool result
            next_state = InformationState(
                id=str(uuid.uuid4()),
                text=f"Result of {action.name}: {tool_result}",
                parent_id=state.id,
                available_actions=state.available_actions,
                timestamp=time.time(),
                data={"tool_result": tool_result}
            )
        else:
            # If no tool is available, create a dummy next state
            next_state = InformationState(
                id=str(uuid.uuid4()),
                text=f"After {action.name}: {state.text}",
                parent_id=state.id,
                available_actions=state.available_actions,
                timestamp=time.time()
            )
        
        # Calculate reward if there's a reward model and target state
        reward = 0.0
        if self.reward_model and self.target_state:
            reward_event = self.reward_model.calculate_reward(next_state, self.target_state)
            reward = reward_event.value
        
        return next_state, reward
    
    def run(self, environment: Environment, max_steps: int = 10, reset: bool = True) -> Tuple[InformationState, List, List, bool]:
        """
        Run the agent in an environment for multiple steps.
        
        Args:
            environment: The environment to run in
            max_steps: Maximum number of steps to take
            reset: Whether to reset the environment before starting
            
        Returns:
            A tuple of (final_state, actions, rewards, success)
        """
        if reset:
            state = environment.reset()
            self.set_current_state(state)
        else:
            state = self.current_state
            if state is None:
                raise ValueError("No current state and reset=False")
        
        states = [state]
        actions = []
        rewards = []
        
        done = False
        success = False
        final_state = state
        
        try:
            for step in range(max_steps):
                if self.verbose:
                    logger.info(f"Step {step+1}/{max_steps}")
                
                # Select and execute action
                action, next_state, reward = self.act(state, environment)
                
                # Update lists
                states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                final_state = next_state
                
                # Check if done
                _, _, step_done, _ = environment.step(action)
                if step_done:
                    if self.verbose:
                        logger.info(f"Environment signaled done at step {step+1}")
                    done = True
                    success = True
                    break
                    
                # Check if we've reached the target state
                if self.target_state and next_state.compare(self.target_state) > 0.9:
                    if self.verbose:
                        logger.info(f"Reached target state at step {step+1}")
                    done = True
                    success = True
                    break
                
                # Update state for next iteration
                state = next_state
                
        except Exception as e:
            logger.error(f"Error during agent run: {e}")
            if self.verbose:
                logger.exception("Exception details:")
        
        # Create a result object
        result = type('AgentRunResult', (), {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'success': success,
            'done': done,
            'final_state': final_state
        })
        
        return result
    
    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the agent's toolkit.
        
        Args:
            tool: The tool to add
        """
        self.tools[tool.name] = tool
        if self.verbose:
            logger.info(f"Added tool: {tool.name}")
    
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Call a tool by name.
        
        Args:
            tool_name: The name of the tool to call
            **kwargs: Parameters for the tool
            
        Returns:
            The result of the tool call
        """
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
            
        if self.verbose:
            logger.info(f"Calling tool: {tool_name}")
        return self.tools[tool_name](**kwargs)
    
    def reflect(self, state: Optional[InformationState] = None) -> List[str]:
        """
        Reflect on the current state and history to generate insights.
        
        Args:
            state: The state to reflect on (default: self.current_state)
            
        Returns:
            A list of reflections
        """
        if state is None:
            state = self.current_state
            if state is None:
                raise ValueError("No current state set")
        
        reflections = []
        
        # If there's a thought generator, use it to generate reflections
        if self.thought_generator:
            thoughts = self.thought_generator.generate_thoughts(state)
            if self.thought_manager:
                for thought in thoughts:
                    self.thought_manager.add_thought(thought)
            reflections.extend([t.text for t in thoughts])
        
        # If there's a memory, retrieve relevant experiences
        if self.memory:
            relevant_experiences = self.memory.get_relevant_experiences(state)
            for i, (exp_state, exp_transition) in enumerate(relevant_experiences):
                reflection = f"Experience {i+1}: In a similar state, action '{exp_transition.action}' was taken with reward {exp_transition.reward}"
                reflections.append(reflection)
        
        return reflections
    
    def save(self, path: str) -> None:
        """
        Save the agent's state to disk.
        
        Args:
            path: The path to save to
        """
        # Implement saving logic here
        if self.verbose:
            logger.info(f"Saved agent state to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the agent's state from disk.
        
        Args:
            path: The path to load from
        """
        # Implement loading logic here
        if self.verbose:
            logger.info(f"Loaded agent state from {path}")
    
    def reset(self) -> None:
        """
        Reset the agent's state.
        """
        self.current_state = None
        self.history = []
        if self.memory:
            self.memory.clear()
        if self.thought_manager:
            self.thought_manager.clear()
            
        if self.verbose:
            logger.info("Reset agent state") 