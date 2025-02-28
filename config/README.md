# Configuration Directory

This directory is intended for configuration files for the Agentic IR system.

## Usage

You can place YAML, JSON, or INI configuration files here to customize the behavior of the Agentic IR components and examples.

## Example Configuration

```yaml
# config/default.yaml
llm:
  model: deepseek-r1:14b
  base_url: http://localhost:11434
  
memory:
  type: in_memory
  
logging:
  level: INFO
``` 