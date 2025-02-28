from setuptools import setup, find_packages

setup(
    name="agentic-ir",
    version="0.1.0",
    description="Implementation of Agentic Information Retrieval",
    author="AgenticIR Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"": ["py.typed"]},
    install_requires=[
        "pydantic>=2.0.0",
        "requests>=2.28.0",
        "pyyaml>=6.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "python-dotenv>=1.0.0",
        "ollama>=0.1.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "agentic-ir-assistant=src.examples.life_assistant_example:main",
            "agentic-ir-research=src.examples.research_assistant_example:main",
        ],
    },
) 