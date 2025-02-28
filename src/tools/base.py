"""
Base tool interfaces for the Agentic IR framework.

This module defines the base Tool interface and common functionality.
"""

import inspect
import logging
import json
from typing import Dict, Any, List, Optional, Callable, Type, get_type_hints
from abc import ABC, abstractmethod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolParameter:
    """
    Represents a parameter for a tool.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        type: str,
        required: bool = True,
        default: Any = None,
        enum: Optional[List[Any]] = None
    ):
        """
        Initialize a tool parameter.
        
        Args:
            name: The name of the parameter
            description: A description of the parameter
            type: The type of the parameter (string, integer, boolean, etc.)
            required: Whether the parameter is required
            default: The default value for the parameter
            enum: A list of allowed values for the parameter
        """
        self.name = name
        self.description = description
        self.type = type
        self.required = required
        self.default = default
        self.enum = enum
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the parameter to a dictionary.
        
        Returns:
            A dictionary representation of the parameter
        """
        result = {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "required": self.required
        }
        
        if self.default is not None:
            result["default"] = self.default
            
        if self.enum is not None:
            result["enum"] = self.enum
            
        return result

class ToolResult:
    """
    Represents the result of a tool execution.
    """
    
    def __init__(
        self,
        success: bool,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a tool result.
        
        Args:
            success: Whether the tool execution was successful
            result: The result of the tool execution
            error: An error message if the execution failed
            metadata: Additional metadata about the execution
        """
        self.success = success
        self.result = result or {}
        self.error = error
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary.
        
        Returns:
            A dictionary representation of the result
        """
        result = {
            "success": self.success,
            "result": self.result
        }
        
        if self.error:
            result["error"] = self.error
            
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result

class BaseTool(ABC):
    """
    Base class for tools in the Agentic IR framework.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize a tool.
        
        Args:
            name: The name of the tool
            description: A description of the tool
            parameters: A list of parameter definitions
        """
        self.name = name
        self.description = description
        self.parameters = [ToolParameter(**p) for p in (parameters or [])]
        logger.info(f"Initialized tool: {name}")
    
    def __call__(self, **kwargs) -> ToolResult:
        """
        Call the tool with the given parameters.
        
        Args:
            **kwargs: The parameters to pass to the tool
            
        Returns:
            The result of the tool execution
        """
        # Validate parameters
        validation_result = self._validate_parameters(kwargs)
        if not validation_result["valid"]:
            return ToolResult(
                success=False,
                error=validation_result["error"]
            )
        
        # Execute the tool
        return self._execute(**kwargs)
    
    @abstractmethod
    def _execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given parameters.
        
        Args:
            **kwargs: The parameters to pass to the tool
            
        Returns:
            The result of the tool execution
        """
        pass
    
    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the parameters for the tool.
        
        Args:
            params: The parameters to validate
            
        Returns:
            A dictionary with "valid" and optionally "error" keys
        """
        # Check for required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                return {
                    "valid": False,
                    "error": f"Missing required parameter: {param.name}"
                }
        
        # Check parameter types (basic validation)
        for name, value in params.items():
            param = next((p for p in self.parameters if p.name == name), None)
            if param is None:
                return {
                    "valid": False,
                    "error": f"Unknown parameter: {name}"
                }
            
            # Type checking (simplified)
            if param.type == "string" and not isinstance(value, str):
                return {
                    "valid": False,
                    "error": f"Parameter {name} should be a string"
                }
            elif param.type == "integer" and not isinstance(value, int):
                return {
                    "valid": False,
                    "error": f"Parameter {name} should be an integer"
                }
            elif param.type == "boolean" and not isinstance(value, bool):
                return {
                    "valid": False,
                    "error": f"Parameter {name} should be a boolean"
                }
            
            # Check enum values
            if param.enum is not None and value not in param.enum:
                return {
                    "valid": False,
                    "error": f"Parameter {name} should be one of: {', '.join(map(str, param.enum))}"
                }
        
        return {"valid": True}
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema for the tool.
        
        Returns:
            A dictionary representing the tool schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters]
        }
    
    def to_json_schema(self) -> str:
        """
        Convert the tool schema to JSON.
        
        Returns:
            A JSON string representing the tool schema
        """
        return json.dumps(self.get_schema(), indent=2)

class FunctionBasedTool(BaseTool):
    """
    A tool that wraps a Python function.
    """
    
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameter_descriptions: Optional[Dict[str, str]] = None
    ):
        """
        Initialize a function-based tool.
        
        Args:
            func: The function to wrap
            name: The name of the tool (defaults to the function name)
            description: A description of the tool (defaults to the function docstring)
            parameter_descriptions: Descriptions for the function parameters
        """
        self.func = func
        name = name or func.__name__
        description = description or (func.__doc__ or "").strip() or f"Execute the {name} function"
        
        # Get function signature and type hints
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        # Build parameters from function signature
        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name == "self" or param.kind == param.VAR_KEYWORD:
                continue
                
            param_type = type_hints.get(param_name, Any)
            param_type_str = self._get_type_str(param_type)
            
            parameters.append({
                "name": param_name,
                "description": (parameter_descriptions or {}).get(param_name, f"The {param_name} parameter"),
                "type": param_type_str,
                "required": param.default == param.empty
            })
            
            # Add default value if present
            if param.default != param.empty:
                parameters[-1]["default"] = param.default
        
        super().__init__(name, description, parameters)
    
    def _execute(self, **kwargs) -> ToolResult:
        """
        Execute the wrapped function.
        
        Args:
            **kwargs: The parameters to pass to the function
            
        Returns:
            The result of the function execution
        """
        try:
            result = self.func(**kwargs)
            
            # If the result is already a ToolResult, return it
            if isinstance(result, ToolResult):
                return result
                
            # Otherwise, wrap it in a ToolResult
            return ToolResult(
                success=True,
                result={"result": result}
            )
        except Exception as e:
            logger.error(f"Error executing function {self.name}: {e}")
            return ToolResult(
                success=False,
                error=f"Error executing function: {str(e)}"
            )
    
    def _get_type_str(self, type_hint: Type) -> str:
        """
        Convert a Python type hint to a string type.
        
        Args:
            type_hint: The type hint to convert
            
        Returns:
            A string representation of the type
        """
        if type_hint == str:
            return "string"
        elif type_hint == int:
            return "integer"
        elif type_hint == float:
            return "number"
        elif type_hint == bool:
            return "boolean"
        elif type_hint == list or getattr(type_hint, "__origin__", None) == list:
            return "array"
        elif type_hint == dict or getattr(type_hint, "__origin__", None) == dict:
            return "object"
        else:
            return "string"  # Default to string for unknown types 