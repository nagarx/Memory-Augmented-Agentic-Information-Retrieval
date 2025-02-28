"""
Document retrieval tools for the Agentic IR framework.

This module provides tools for searching and retrieving information from 
local markdown files containing research papers.
"""

import os
import re
import logging
import glob
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .base import BaseTool, ToolResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """
    Represents a chunk of text from a document.
    """
    content: str
    source: str
    section: str
    start_line: int
    end_line: int
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the document chunk to a dictionary.
        
        Returns:
            A dictionary representation
        """
        return {
            "content": self.content,
            "source": self.source,
            "section": self.section,
            "start_line": self.start_line,
            "end_line": self.end_line
        }

class DocumentRetriever:
    """
    Class for retrieving information from documents.
    """
    
    def __init__(self, docs_dir: str = None):
        """
        Initialize the document retriever.
        
        Args:
            docs_dir: Directory containing the documents
        """
        # Use default path if none provided
        self.docs_dir = docs_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "research_papers"
        )
        logger.info(f"Document retriever initialized with directory: {self.docs_dir}")
    
    def search_documents(self, query: str, max_results: int = 5) -> List[DocumentChunk]:
        """
        Search through documents for relevant chunks based on query.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant document chunks
        """
        results = []
        
        # Get list of markdown files
        md_files = glob.glob(os.path.join(self.docs_dir, "*.md"))
        logger.info(f"Found {len(md_files)} markdown files to search")
        
        # Simple search - in a real implementation you would use embeddings and vector search
        query_terms = query.lower().split()
        
        for file_path in md_files:
            if file_path.endswith("README.md"):  # Skip the README
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split the document into sections based on markdown headers
                sections = self._split_into_sections(content)
                
                for section_title, section_content, start_line, end_line in sections:
                    # Calculate a simple relevance score
                    score = self._calculate_relevance(section_content, query_terms)
                    
                    if score > 0:
                        results.append((
                            DocumentChunk(
                                content=section_content,
                                source=os.path.basename(file_path),
                                section=section_title,
                                start_line=start_line,
                                end_line=end_line
                            ),
                            score
                        ))
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        # Sort results by relevance score
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top results
        return [chunk for chunk, _ in results[:max_results]]
    
    def _split_into_sections(self, content: str) -> List[Tuple[str, str, int, int]]:
        """
        Split a document into sections based on markdown headers.
        
        Args:
            content: The document content
            
        Returns:
            List of tuples (section_title, section_content, start_line, end_line)
        """
        lines = content.split('\n')
        sections = []
        current_section = "Document"
        current_content = []
        start_line = 0
        
        for i, line in enumerate(lines):
            # Check if this is a header line
            if line.startswith('#'):
                # If we have accumulated content, add it to sections
                if current_content:
                    section_content = '\n'.join(current_content)
                    sections.append((current_section, section_content, start_line, i-1))
                
                # Start a new section
                current_section = line.lstrip('#').strip()
                current_content = []
                start_line = i
            else:
                current_content.append(line)
        
        # Add the last section
        if current_content:
            section_content = '\n'.join(current_content)
            sections.append((current_section, section_content, start_line, len(lines)-1))
        
        return sections
    
    def _calculate_relevance(self, text: str, query_terms: List[str]) -> float:
        """
        Calculate a simple relevance score for text matching query terms.
        
        Args:
            text: The text to evaluate
            query_terms: List of query terms
            
        Returns:
            A relevance score
        """
        text_lower = text.lower()
        score = 0.0
        
        for term in query_terms:
            if term in text_lower:
                # Count occurrences
                count = text_lower.count(term)
                score += count * 0.1
                
                # Bonus for terms in the first paragraph (likely more important)
                first_para = text_lower.split('\n\n')[0] if '\n\n' in text_lower else text_lower
                if term in first_para:
                    score += 0.5
        
        return score
    
    def get_document_metadata(self) -> List[Dict[str, Any]]:
        """
        Get metadata for all available documents.
        
        Returns:
            List of document metadata
        """
        metadata = []
        
        # Get list of markdown files
        md_files = glob.glob(os.path.join(self.docs_dir, "*.md"))
        
        for file_path in md_files:
            if file_path.endswith("README.md"):  # Skip the README
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.readlines()
                
                # Extract basic metadata from content
                filename = os.path.basename(file_path)
                title = None
                authors = None
                
                for line in content[:20]:  # Check just the first 20 lines
                    if line.startswith('# '):
                        title = line[2:].strip()
                    elif line.startswith('## Authors') and len(content) > content.index(line) + 1:
                        authors = content[content.index(line) + 1].strip()
                
                metadata.append({
                    "filename": filename,
                    "title": title or filename,
                    "authors": authors or "Unknown",
                    "path": file_path
                })
            except Exception as e:
                logger.error(f"Error extracting metadata from {file_path}: {e}")
        
        return metadata

class DocumentSearchTool(BaseTool):
    """
    Tool for searching through research papers.
    """
    
    def __init__(self, docs_dir: str = None):
        """
        Initialize the document search tool.
        
        Args:
            docs_dir: Directory containing the documents
        """
        super().__init__(
            name="document_search",
            description="Search through research papers for information",
            parameters=[
                {
                    "name": "query",
                    "description": "The search query",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "max_results",
                    "description": "Maximum number of results to return",
                    "type": "integer",
                    "required": False,
                    "default": 5
                }
            ]
        )
        self.retriever = DocumentRetriever(docs_dir)
        logger.info("Initialized DocumentSearchTool")
    
    def _execute(self, query: str, max_results: int = 5) -> ToolResult:
        """
        Execute the document search.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            A ToolResult containing the search results
        """
        logger.info(f"Searching documents for: {query}")
        
        try:
            results = self.retriever.search_documents(query, max_results)
            
            # Format results
            formatted_results = []
            for chunk in results:
                formatted_results.append({
                    "source": chunk.source,
                    "section": chunk.section,
                    "content": chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line
                })
            
            return ToolResult(
                success=True,
                result={
                    "query": query,
                    "num_results": len(results),
                    "results": formatted_results
                }
            )
        except Exception as e:
            logger.error(f"Error during document search: {e}")
            return ToolResult(
                success=False,
                error=f"Error during document search: {str(e)}"
            )

class DocumentReadTool(BaseTool):
    """
    Tool for reading a specific document or section.
    """
    
    def __init__(self, docs_dir: str = None):
        """
        Initialize the document read tool.
        
        Args:
            docs_dir: Directory containing the documents
        """
        super().__init__(
            name="document_read",
            description="Read a specific document or section",
            parameters=[
                {
                    "name": "filename",
                    "description": "The filename of the document to read",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "section",
                    "description": "The section title to read (if omitted, reads the entire document)",
                    "type": "string",
                    "required": False
                }
            ]
        )
        self.retriever = DocumentRetriever(docs_dir)
        logger.info("Initialized DocumentReadTool")
    
    def _execute(self, filename: str, section: Optional[str] = None) -> ToolResult:
        """
        Execute the document read.
        
        Args:
            filename: The filename of the document to read
            section: The section title to read (if omitted, reads the entire document)
            
        Returns:
            A ToolResult containing the document content
        """
        logger.info(f"Reading document: {filename}, section: {section}")
        
        try:
            file_path = os.path.join(self.retriever.docs_dir, filename)
            
            if not os.path.exists(file_path):
                return ToolResult(
                    success=False,
                    error=f"Document not found: {filename}"
                )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # If a section is specified, extract just that section
            if section:
                sections = self.retriever._split_into_sections(content)
                matching_sections = [s for s in sections if section.lower() in s[0].lower()]
                
                if matching_sections:
                    section_title, section_content, start_line, end_line = matching_sections[0]
                    return ToolResult(
                        success=True,
                        result={
                            "filename": filename,
                            "section": section_title,
                            "content": section_content,
                            "start_line": start_line,
                            "end_line": end_line
                        }
                    )
                else:
                    return ToolResult(
                        success=False,
                        error=f"Section not found: {section}"
                    )
            else:
                # Return the entire document
                return ToolResult(
                    success=True,
                    result={
                        "filename": filename,
                        "section": "Full Document",
                        "content": content
                    }
                )
        except Exception as e:
            logger.error(f"Error reading document: {e}")
            return ToolResult(
                success=False,
                error=f"Error reading document: {str(e)}"
            )

class DocumentListTool(BaseTool):
    """
    Tool for listing available documents.
    """
    
    def __init__(self, docs_dir: str = None):
        """
        Initialize the document list tool.
        
        Args:
            docs_dir: Directory containing the documents
        """
        super().__init__(
            name="document_list",
            description="List all available research papers",
            parameters=[]
        )
        self.retriever = DocumentRetriever(docs_dir)
        logger.info("Initialized DocumentListTool")
    
    def _execute(self) -> ToolResult:
        """
        Execute the document list.
        
        Returns:
            A ToolResult containing the list of available documents
        """
        logger.info("Listing available documents")
        
        try:
            metadata = self.retriever.get_document_metadata()
            
            return ToolResult(
                success=True,
                result={
                    "count": len(metadata),
                    "documents": metadata
                }
            )
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return ToolResult(
                success=False,
                error=f"Error listing documents: {str(e)}"
            ) 