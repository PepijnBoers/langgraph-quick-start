import os
import json
from typing import Annotated, Dict, List, Literal, TypedDict, Union, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

#------------------------------------------------------------------------------
# STATE DEFINITION
#------------------------------------------------------------------------------

# Define the state for our multi-agent system
class AgentState(TypedDict):
    """State for the multi-agent system."""
    messages: Annotated[List, add_messages]  # The conversation history
    current_agent: str  # Which agent is currently active
    final_answer: Dict[str, Any]  # The final answer in JSON format

#------------------------------------------------------------------------------
# TOOLS
#------------------------------------------------------------------------------

# Create a web search tool
@tool
def web_search(query: str) -> str:
    """Search the web for information on a given query."""
    # In a real implementation, this would use a search API
    # For this example, we'll simulate a response
    if "weather" in query.lower():
        return "The current weather is sunny with a high of 75°F."
    elif "population" in query.lower():
        return "The current world population is approximately 8 billion people."
    elif "president" in query.lower():
        return "The current US President is Joe Biden."
    else:
        return f"Here are some search results for: {query}"

# Create a calculator tool
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Define a custom class for handling tools
class CustomToolNode:
    """A custom node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: AgentState):
        # Get the last message
        if messages := state.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        
        # Process tool calls
        outputs = []
        for tool_call in message.tool_calls:
            # Parse the arguments
            try:
                args = json.loads(tool_call["args"])
            except:
                args = tool_call["args"]
            
            # Call the tool
            tool_result = self.tools_by_name[tool_call["name"]].invoke(args)
            
            # Create a tool message with the result
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        
        # Return the updated state with the tool messages
        return {"messages": outputs}

# Define the tools list
tools = [web_search, calculator]

#------------------------------------------------------------------------------
# ROUTER AGENT
#------------------------------------------------------------------------------

# Define structured output schema for router
class RouterResponse(BaseModel):
    """Response from the router agent."""
    agent: str = Field(..., description="The name of the agent that should handle the query")

# Define LLM for router and bind structured output
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
router_llm_structured = router_llm.with_structured_output(RouterResponse)

# Create the router agent
def router_agent(state: AgentState):
    """
    Router agent that decides which specialized agent should handle the query.
    """
    messages = state["messages"]
    
    # Create a system message for the router
    router_system_message = SystemMessage(
        content="""You are a router agent that determines which specialized agent should handle a user query.
        
        Available agents:
        - research_agent: For questions requiring factual information or research
        - calculator_agent: For mathematical calculations or numerical analysis
        
        Analyze the user's query and respond with the name of the agent that should handle it.
        """
    )
    
    # Combine the system message with the conversation history
    router_messages = [router_system_message] + messages
    
    # Get the router's decision using structured output
    response = router_llm_structured.invoke(router_messages)
    
    # Get the agent name from the structured response
    next_agent = response.agent.strip().lower()
    
    # Validate the agent name
    if next_agent not in ["research_agent", "calculator_agent"]:
        # Default to research agent if invalid
        next_agent = "research_agent"
    
    # Return the updated state
    return {"current_agent": next_agent}

# Define the condition for routing to specialized agents
def route_to_agent(state: AgentState):
    """
    Determine which agent to route to based on the current_agent field.
    """
    return state["current_agent"]

#------------------------------------------------------------------------------
# RESEARCH AGENT
#------------------------------------------------------------------------------

# Define LLM for specialist agents
specialist_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Pre-bind tools to specialist LLMs for better performance
research_llm = specialist_llm.bind_tools(tools=[web_search])
calculator_llm = specialist_llm.bind_tools(tools=[calculator])

# Create the research agent
def research_agent(state: AgentState):
    """
    Research agent that can search the web for information.
    """
    messages = state["messages"]
    
    # Create a system message for the research agent
    research_system_message = SystemMessage(
        content="""You are a research agent that can search the web for information.
        Use the web_search tool to find information when needed.
        Provide detailed and accurate responses based on the search results.
        """
    )
    
    # Combine the system message with the conversation history
    research_messages = [research_system_message] + messages
    
    # Get the research agent's response using the pre-bound LLM
    response = research_llm.invoke(research_messages)
    
    # Return the updated state with the agent's response
    return {"messages": [response]}

#------------------------------------------------------------------------------
# CALCULATOR AGENT
#------------------------------------------------------------------------------

# Create the calculator agent
def calculator_agent(state: AgentState):
    """
    Calculator agent that can perform mathematical calculations.
    """
    messages = state["messages"]
    
    # Create a system message for the calculator agent
    calculator_system_message = SystemMessage(
        content="""You are a calculator agent that can perform mathematical calculations.
        Use the calculator tool to evaluate mathematical expressions.
        Provide clear explanations of the calculations and results.
        """
    )
    
    # Combine the system message with the conversation history
    calculator_messages = [calculator_system_message] + messages
    
    # Get the calculator agent's response using the pre-bound LLM
    response = calculator_llm.invoke(calculator_messages)
    
    # Return the updated state with the agent's response
    return {"messages": [response]}

#------------------------------------------------------------------------------
# FORMATTER AGENT
#------------------------------------------------------------------------------

# Define structured output schema for formatter
class FormattedAnswer(BaseModel):
    """Structured response with answer and mnemonic."""
    answer: str = Field(..., description="A concise answer to the user's query")
    mnemonic: str = Field(..., description="A memorable phrase or acronym to help remember the answer")
    source: str = Field(..., description="The source of the information (e.g., 'web search', 'calculation')")
    confidence: float = Field(..., description="A number between 0 and 1 indicating confidence in the answer")

# Define LLM for formatter
formatter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
formatter_llm_structured = formatter_llm.with_structured_output(FormattedAnswer)

# Create the formatter agent
def formatter_agent(state: AgentState):
    """
    Formatter agent that creates a JSON response with an answer and mnemonic.
    """
    messages = state["messages"]
    
    # Create a system message for the formatter agent
    formatter_system_message = SystemMessage(
        content="""You are a formatter agent that creates a structured response.
        
        Based on the conversation history, create a response with:
        - A concise answer to the user's query
        - A memorable phrase or acronym to help remember the answer
        - The source of the information (e.g., "web search", "calculation")
        - Your confidence in the answer (a number between 0 and 1)
        """
    )
    
    # Combine the system message with the conversation history
    formatter_messages = [formatter_system_message] + messages
    
    # Get the formatter agent's response using structured output
    response = formatter_llm_structured.invoke(formatter_messages)
    
    # Return the final answer as a dictionary using model_dump() instead of dict()
    return {"final_answer": response.model_dump()}

#------------------------------------------------------------------------------
# ROUTING CONDITIONS
#------------------------------------------------------------------------------

# Define a custom tools condition function
def tools_condition_custom(state: AgentState):
    """
    Custom condition to check if tools were called in the last message.
    Returns the name of the next node to call.
    """
    # Get the last message
    last_message = state["messages"][-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # If tool calls exist, route to the tools node
        return "tools"
    else:
        # If no tool calls, route based on the current agent
        if state["current_agent"] == "research_agent" or state["current_agent"] == "calculator_agent":
            # If we're in a specialized agent and no tools were called, go to formatter
            return "formatter_agent"
        else:
            # Otherwise, end the graph
            return END

# Define a function to route tools back to the appropriate agent
def route_tools_to_agent(state: AgentState):
    """
    Route tools back to the appropriate agent.
    """
    return state["current_agent"]

#------------------------------------------------------------------------------
# GRAPH CONSTRUCTION
#------------------------------------------------------------------------------

# Build the graph
def build_graph():
    """
    Build the multi-agent system graph.
    """
    # Create the graph
    graph = StateGraph(AgentState)
    
    # Add the nodes
    graph.add_node("router_agent", router_agent)
    graph.add_node("research_agent", research_agent)
    graph.add_node("calculator_agent", calculator_agent)
    graph.add_node("formatter_agent", formatter_agent)
    
    # Add custom tool node
    custom_tool_node = CustomToolNode(tools=tools)
    graph.add_node("tools", custom_tool_node)
    
    # Add the edges
    graph.add_edge(START, "router_agent")
    graph.add_conditional_edges("router_agent", route_to_agent)
    
    # The key insight is that add_conditional_edges works differently from add_edge. 
    # When you use add_conditional_edges, you're not specifying the destination nodes directly in this line. 
    # Instead, you're providing a function (route_to_agent) that will determine the destination at runtime.
    # This is done by returning the name of the next node. In this case: "research_agent" or "calculator_agent".

    # Connect specialized agents to the tools condition
    graph.add_conditional_edges("research_agent", tools_condition_custom)
    graph.add_conditional_edges("calculator_agent", tools_condition_custom)
    
    # Connect tools back to the appropriate agent using conditional edges
    # Routes to the "formatter_agent" eventually.
    graph.add_conditional_edges("tools", route_tools_to_agent)

    # Connect the formatter to the end
    graph.add_edge("formatter_agent", END)
    
    # Compile the graph
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

# Create the graph
graph = build_graph()

#------------------------------------------------------------------------------
# EXECUTION AND DISPLAY
#------------------------------------------------------------------------------

# Function to extract content from node updates
def extract_content(data):
    """Extract content from node updates for display."""
    # Extract content from router agent
    if 'router_agent' in data:
        return {
            'type': 'router',
            'content': f"Router selected: {data['router_agent'].get('current_agent', 'unknown')}"
        }
    
    # Extract content from research agent
    elif 'research_agent' in data and 'messages' in data['research_agent']:
        for message in data['research_agent']['messages']:
            if hasattr(message, 'content'):
                return {
                    'type': 'research',
                    'content': message.content
                }
    
    # Extract content from calculator agent
    elif 'calculator_agent' in data and 'messages' in data['calculator_agent']:
        for message in data['calculator_agent']['messages']:
            if hasattr(message, 'content'):
                return {
                    'type': 'calculator',
                    'content': message.content
                }
    
    # Extract content from tools
    elif 'tools' in data and 'messages' in data['tools']:
        tool_outputs = []
        for message in data['tools']['messages']:
            if hasattr(message, 'content'):
                tool_outputs.append(f"{message.name}: {message.content}")
        if tool_outputs:
            return {
                'type': 'tools',
                'content': '\n'.join(tool_outputs)
            }
    
    # Extract content from formatter agent
    elif 'formatter_agent' in data and 'final_answer' in data['formatter_agent']:
        return {
            'type': 'formatter',
            'content': json.dumps(data['formatter_agent']['final_answer'], indent=2)
        }
    
    return None

# Function to stream graph execution and display intermediate results
def stream_graph_execution(query):
    """Stream the execution of the graph and display intermediate results."""
    # Create the initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "current_agent": "",
        "final_answer": {}
    }
    
    # Create a configuration with a thread_id for the checkpointer
    config = {"configurable": {"thread_id": f"example_thread_{query}"}}
    
    # Stream the graph execution
    print(f"\n--- Processing query: {query} ---\n")
    events = graph.stream(initial_state, config, stream_mode="updates")
    
    final_answer = None
    
    for event in events:
        # Skip events that don't contain node updates
        if not any(node in event for node in ["router_agent", "research_agent", "calculator_agent", "tools", "formatter_agent"]):
            continue
        
        # Extract and display content
        content = extract_content(event)
        if content:
            print(f"[{content['type'].upper()}]")
            print(content['content'])
            print("-" * 40)
            
            # Save the final answer if this is from the formatter agent
            if content['type'] == 'formatter':
                try:
                    final_answer = json.loads(content['content'])
                except:
                    final_answer = content['content']
    
    # Print the final answer
    print("\nFinal Answer:")
    if final_answer:
        print(json.dumps(final_answer, indent=2))
    else:
        print("No final answer available.")
    
    return final_answer

#------------------------------------------------------------------------------
# MAIN EXECUTION
#------------------------------------------------------------------------------

# Example usage
if __name__ == "__main__":
    # Example queries to test
    queries = [
        "What is the current weather?",
        "Calculate 25 * 16 + 42",
        "Who is the current US President?"
    ]
    
    for query in queries:
        stream_graph_execution(query)
        print("\n" + "=" * 60 + "\n") 