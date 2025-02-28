"""
Tools for the Agentic IR framework.

This package contains tools that the agent can use to interact with the environment.
"""

from .base import BaseTool, ToolResult
from .search import WebSearchTool, WebContentTool
from .document_retrieval import DocumentSearchTool, DocumentReadTool, DocumentListTool

__all__ = [
    'BaseTool',
    'ToolResult',
    'WebSearchTool',
    'WebContentTool',
    'DocumentSearchTool',
    'DocumentReadTool',
    'DocumentListTool'
] 