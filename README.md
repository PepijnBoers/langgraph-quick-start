# Multi-Agent System with LangGraph

This project implements a multi-agent system using the LangGraph framework. The system routes user queries to specialized agents, each with specific capabilities, and returns structured responses.

## Features

- **Router Agent**: Analyzes user queries and routes them to the appropriate specialist agent
- **Research Agent**: Searches for information using web search tools
- **Calculator Agent**: Performs mathematical calculations
- **Formatter Agent**: Creates structured responses with answers and mnemonics
- **Streaming Execution**: Real-time feedback on query processing

## Architecture

The system uses a directed graph to manage the flow of information between agents:

1. User query enters the system
2. Router agent determines which specialist agent should handle the query
3. Specialist agent processes the query, using tools when necessary
4. Formatter agent creates a structured response
5. System returns the final answer

## Multi-Agent System

The multi-agent system is the main focus of this project. It demonstrates how to:

- Create specialized agents with different capabilities
- Route queries to the appropriate agent
- Use tools within agents to extend their capabilities
- Format responses in a consistent, structured way
- Stream the execution process for real-time feedback

## Streaming Execution

The system supports streaming execution, which provides real-time feedback as the query is processed. This allows you to see:

- Which agent is handling the query
- What tools are being used
- Intermediate results from each agent
- The final structured response

## Tools

The project includes a tools folder with utility functions:

- **vergelijker.py**: Contains utility functions for comparison operations

These tools are separate from the main multi-agent system but can be used to extend its functionality.

## Usage

```python
from multi_agent_system import stream_graph_execution

# Execute the multi-agent system
result = stream_graph_execution("What is the current weather?")
```

## Example Queries

- "What is the current weather?"
- "Calculate 25 * 16 + 42"
- "Who is the current US President?"

## Requirements

- Python 3.8+
- LangGraph
- LangChain
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)

## Installation

```bash
pip install -r requirements.txt
```

## Running the System

```bash
python multi_agent_system.py
```

Here's the "How It Works" section that I removed from the README:

### How It Works

```
START
  │
  ▼
router_agent
  │
  ▼
specialized_agent (research_agent, calculator_agent, knowledge_agent)
  │
  ├─────► tools (if tool calls are made)
  │       │
  │       ▼
  │     specialized_agent (return to the same agent)
  │
  ▼
formatter_agent
  │
  ▼
END
```

The system uses LangGraph's StateGraph to create a directed graph of agents:


Each agent has a specific role:

- **Router Agent**: Determines which specialist should handle the query
- **Research Agent**: Handles factual queries using web search
- **Calculator Agent**: Handles mathematical calculations
- **Formatter Agent**: Creates a structured final response

The system also includes a custom tools node that processes tool calls from the agents.