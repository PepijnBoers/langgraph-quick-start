# Multi-Agent LLM System with LangGraph

This project demonstrates how to create a graph of different LLM agents that communicate with each other using the LangGraph framework. The system includes:

1. A router agent that directs user queries to specialized agents
2. Specialized agents that can use tools to answer specific types of questions
3. A formatter agent that creates structured JSON responses with answers and mnemonics

## Project Structure

- `multi_agent_system.py`: Basic implementation of a multi-agent system
- `advanced_multi_agent_system.py`: Advanced implementation with more agents and better JSON formatting
- `tools/`: Directory containing tool implementations

## Features

- **Agent Routing**: Initial router agent directs queries to the appropriate specialized agent
- **Tool Usage**: Specialized agents can use tools like web search, calculator, and knowledge base
- **Structured Responses**: Final formatter agent creates JSON responses with answers and mnemonics
- **State Management**: System maintains state across the entire conversation flow
- **Memory**: Conversation history is preserved for context

## Requirements

```
pip install langgraph langchain-openai langchain-core
```

You'll also need to set your OpenAI API key:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

## Usage

### Basic Example

```python
from multi_agent_system import graph, stream_graph_execution

# Stream the execution of the graph with intermediate results
stream_graph_execution("What is the capital of France?")
```

### Advanced Example

```python
from advanced_multi_agent_system import graph, stream_graph_execution
import json

# Example queries to test
queries = [
    "What is the capital of France?",
    "Calculate 25 * 16 + 42",
    "Tell me about Python programming language"
]

for query in queries:
    stream_graph_execution(query)
```

### Streaming Intermediate Results

The system provides real-time visibility into the execution flow by streaming intermediate results:

```
--- Processing query: What is the capital of France? ---

[ROUTER]
Router selected: research_agent
----------------------------------------
[RESEARCH]
I'll search for information about the capital of France.
----------------------------------------
[TOOLS]
web_search: "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Brazil is Brasília."
----------------------------------------
[RESEARCH]
Based on the search results, the capital of France is Paris.
----------------------------------------
[FORMATTER]
{
  "answer": "The capital of France is Paris.",
  "mnemonic": "PAF - Paris As France's capital",
  "source": "web search",
  "confidence": 0.98
}
----------------------------------------

Final Answer:
{
  "answer": "The capital of France is Paris.",
  "mnemonic": "PAF - Paris As France's capital",
  "source": "web search",
  "confidence": 0.98
}
```

This streaming functionality allows you to:
- See which agent is handling the query in real-time
- Monitor tool usage and results
- Track the flow of information through the system
- Debug and understand the decision-making process

## How It Works

1. **User Input**: The system receives a user query
2. **Router Agent**: Analyzes the query and routes it to the appropriate specialized agent
3. **Specialized Agent**: Processes the query using relevant tools
4. **Tool Usage**: If needed, the agent uses tools to gather information
5. **Formatter Agent**: Creates a structured JSON response with an answer and mnemonic
6. **Final Output**: The system returns the formatted response to the user

## Implementation Details

The implementation uses a verbose and customizable approach with LangGraph:

1. **Custom Tool Node**: Instead of using the prebuilt ToolNode, we implement a custom `CustomToolNode` class that provides more control over tool execution and state management.

2. **Custom Tools Condition**: We implement a custom `tools_condition_custom` function that determines routing based on whether tools were called and the current agent.

3. **Explicit State Management**: The state includes fields for tracking the current agent, tools used, and the final answer.

4. **Conditional Routing**: We use conditional edges to route between nodes based on the state, providing fine-grained control over the flow.

5. **Memory Checkpointing**: We use a `MemorySaver` to maintain state across multiple invocations of the graph.

6. **Structured Outputs**: We use LangChain's structured output capabilities to ensure that agent responses conform to specific schemas, making the system more reliable.

### Structured Output Schemas

We define Pydantic models for structured outputs:

```python
class RouterResponse(BaseModel):
    """Response from the router agent."""
    agent: str = Field(..., description="The name of the agent that should handle the query")

class FormattedAnswer(BaseModel):
    """Structured response with answer and mnemonic."""
    answer: str = Field(..., description="A concise answer to the user's query")
    mnemonic: str = Field(..., description="A memorable phrase or acronym to help remember the answer")
    source: str = Field(..., description="The source of the information")
    confidence: float = Field(..., description="A number between 0 and 1 indicating confidence")
    tools_used: List[str] = Field(default_factory=list, description="List of tools used")
```

These schemas ensure that:
- The router agent always returns a valid agent name
- The formatter agent always returns a properly structured answer
- The system is more robust against unexpected outputs

## Graph Structure

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

## Extending the System

You can extend this system by:

1. Adding new specialized agents for different domains
2. Creating custom tools for specific tasks
3. Enhancing the formatter agent to include additional information
4. Implementing more complex routing logic

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction) 