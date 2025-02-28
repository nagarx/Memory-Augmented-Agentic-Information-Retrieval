"""
Ollama integration for the Agentic IR framework.

This module provides integration with Ollama for local LLM inference.
"""

import json
import logging
import requests
from typing import Dict, Any, List, Optional, Callable, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Client for interacting with Ollama API.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "deepseek-r1:14b"):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: The base URL of the Ollama API
            model_name: The name of the model to use
        """
        self.base_url = base_url
        self.model_name = model_name
        self.generate_endpoint = f"{base_url}/api/generate"
        self.embeddings_endpoint = f"{base_url}/api/embeddings"
        logger.info(f"Initialized OllamaClient with model: {model_name}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                temperature: float = 0.7, max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Generate text using Ollama.
        
        Args:
            prompt: The prompt to generate from
            system_prompt: An optional system prompt
            temperature: The temperature for generation
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            A dictionary with the generated text
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "num_predict": max_tokens
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(self.generate_endpoint, json=payload)
            
            # Check for successful response
            if response.status_code != 200:
                logger.error(f"Error from Ollama API: {response.status_code}, {response.text}")
                return {"error": f"API error: {response.status_code}", "response": ""}
            
            # Try to parse JSON response
            try:
                result = response.json()
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from Ollama: {e}")
                # Try to extract the response manually if JSON parsing fails
                text = response.text
                if "response" in text:
                    # Simple extraction of the response field
                    try:
                        response_text = text.split('"response":"')[1].split('","done')[0]
                        # Unescape JSON string
                        response_text = response_text.encode().decode('unicode_escape')
                        return {"response": response_text}
                    except:
                        pass
                
                # Return a fallback response
                return {"error": "JSON parsing error", "response": text[:1000]}
                
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return {"error": str(e), "response": ""}
    
    def get_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings for a text.
        
        Args:
            text: The text to get embeddings for
            
        Returns:
            A list of embeddings
        """
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        try:
            response = requests.post(self.embeddings_endpoint, json=payload)
            result = response.json()
            return result.get("embedding", [])
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return []
    
    def create_completion_function(self, system_prompt: Optional[str] = None,
                                 temperature: float = 0.7, max_tokens: int = 2000) -> Callable[[str], str]:
        """
        Create a callable function for generating text.
        
        Args:
            system_prompt: An optional system prompt
            temperature: The temperature for generation
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            A callable function that takes a prompt and returns generated text
        """
        def completion_function(prompt: str) -> str:
            response = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if "error" in response and response["error"]:
                logger.error(f"Error in completion function: {response['error']}")
                return f"Error: {response['error']}"
                
            return response.get("response", "")
        
        return completion_function
    
    def create_tool_calling_function(self, system_prompt: Optional[str] = None,
                                  temperature: float = 0.1) -> Callable[[str], Dict[str, Any]]:
        """
        Create a callable function for tool calling.
        
        This function will attempt to parse the LLM output as JSON.
        
        Args:
            system_prompt: An optional system prompt
            temperature: The temperature for generation
            
        Returns:
            A callable function that takes a prompt and returns a dictionary
        """
        def tool_calling_function(prompt: str) -> Dict[str, Any]:
            # Add instructions to output JSON
            enhanced_prompt = prompt + "\n\nYou must respond with a valid JSON object."
            
            response_text = self.create_completion_function(
                system_prompt=system_prompt,
                temperature=temperature
            )(enhanced_prompt)
            
            # Try to extract JSON from the response
            try:
                # Find JSON part in the response
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}")
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx+1]
                    return json.loads(json_str)
                
                # Try alternative format with ```json
                json_pattern = "```json"
                if json_pattern in response_text:
                    parts = response_text.split(json_pattern)
                    for part in parts[1:]:  # Skip the part before the first ```json
                        closing_marker = "```"
                        if closing_marker in part:
                            json_str = part.split(closing_marker)[0].strip()
                            return json.loads(json_str)
                
                # If no JSON found, try to parse the whole response
                return json.loads(response_text)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing JSON from response: {e}")
                logger.error(f"Raw response: {response_text}")
                return {"error": "Failed to parse JSON", "text": response_text}
        
        return tool_calling_function

# Default system prompts
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant powered by the Agentic Information Retrieval framework.
Your role is to help users find and process information effectively and accurately."""

THOUGHT_PROCESS_SYSTEM_PROMPT = """You are generating thoughts for an AI agent.
Your role is to explore ideas, analyze information, and suggest possible approaches to solve the user's query.
Be thorough, analytical, and creative in your thinking process."""

POLICY_SYSTEM_PROMPT = """You are determining the next action for an AI agent.
Your role is to select the most appropriate action based on the current state and thoughts.
Consider the available tools, the user's query, and the information collected so far."""

def create_ollama_client(base_url: str = "http://localhost:11434", model_name: str = "deepseek-r1:14b") -> OllamaClient:
    """
    Create an Ollama client.
    
    Args:
        base_url: The base URL of the Ollama API
        model_name: The name of the model to use
        
    Returns:
        An initialized OllamaClient
    """
    return OllamaClient(base_url=base_url, model_name=model_name)

def create_completion_function(model_name: str = "deepseek-r1:14b", 
                             system_prompt: Optional[str] = None,
                             temperature: float = 0.7) -> Callable[[str], str]:
    """
    Create a completion function using Ollama.
    
    Args:
        model_name: The name of the model to use
        system_prompt: An optional system prompt
        temperature: The temperature for generation
        
    Returns:
        A callable function that takes a prompt and returns generated text
    """
    client = create_ollama_client(model_name=model_name)
    return client.create_completion_function(system_prompt=system_prompt, temperature=temperature)

def create_thought_process_function(
    client: Optional[OllamaClient] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    base_url: str = "http://localhost:11434",
    model_name: str = "deepseek-r1:14b"
) -> Callable[[str], str]:
    """
    Create a function for generating thought processes.
    
    Args:
        client: Optional OllamaClient (will be created if not provided)
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        base_url: The base URL for the Ollama API (used if client is not provided)
        model_name: The model to use (used if client is not provided)
        
    Returns:
        A function that takes a prompt and returns a thought process
    """
    return create_completion_function(
        model_name=model_name,
        system_prompt=THOUGHT_PROCESS_SYSTEM_PROMPT,
        temperature=temperature
    )

def create_policy_function(
    client: Optional[OllamaClient] = None,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    base_url: str = "http://localhost:11434",
    model_name: str = "deepseek-r1:14b"
) -> Callable[[str], str]:
    """
    Create a function for policy decisions.
    
    Args:
        client: Optional OllamaClient (will be created if not provided)
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        base_url: The base URL for the Ollama API (used if client is not provided)
        model_name: The model to use (used if client is not provided)
        
    Returns:
        A function that takes a prompt and returns a policy decision
    """
    return create_completion_function(
        model_name=model_name,
        system_prompt=POLICY_SYSTEM_PROMPT,
        temperature=temperature
    ) 