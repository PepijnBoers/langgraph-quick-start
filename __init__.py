from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from .tools.sql_tool import sql_tool
# from tools.sql_tool import sql_tool


# Define llm and bind tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [sql_tool]
llm_with_tools = llm.bind_tools(tools)


# class State(TypedDict):
#     messages: Annotated[list, add_messages]
#     session: str = None
#     tiles: list[dict]


# Build graph using the State defined above
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# Define LangGraph State object
# Define the schema for the input
class InputState(TypedDict):
    messages: Annotated[list, add_messages]
    session: str = None


# Define the schema for the output
class OutputState(TypedDict):
    messages: Annotated[list, add_messages]
    tiles: list[dict]


# Define the overall schema, combining both input and output
class OverallState(InputState, OutputState):
    pass


# Build the graph with input and output schemas specified
graph_builder = StateGraph(OverallState, input=InputState, output=OutputState)
# graph_builder = StateGraph(State)

def chatbot(state: OverallState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Add chatbot node and make it the START
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

# Add tools node and connect chatbot to tools
# Any time a tool is called, we return to the chatbot 
# to decide the next step.
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

# Compile graph with memory (remembers conversation)
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


with open(f"{Path(__file__).parent}/template.txt", "r") as f:
    template = f.read()


def extract_content(data):
    # First try to get AIMessage content from chatbot messages
    if 'chatbot' in data and 'messages' in data['chatbot']:
        for message in data['chatbot']['messages']:
            if hasattr(message, 'content'):
                # return message.content
                return {'type': 'text', 'content': message.content}
    
    # If no AIMessage content, try to get tiles
    if 'tools' in data and 'tiles' in data['tools']:
        # return data['tools']['tiles']
        return {'type': 'json', 'content': data['tools']['tiles']}
    
    return None


message_store = {}

if __name__ == "__main__":
    session_id = str(1)
    
    while True:
        user_input = input("Geef input: ")

        config = {"configurable": {"thread_id": session_id}}
        
        message_store[session_id] = message_store.get(session_id, [SystemMessage(content=template)])
        message_store[session_id].append(HumanMessage(content=user_input))

        events = graph.stream({"messages": message_store[session_id], "session": session_id}, config, stream_mode="updates")
        for event in events:
            if "messages" in event.keys():
                # not a node update
                continue
            print(extract_content(event), end="\n\n", flush=True)

    # response = graph.invoke({"messages": message_store[session_id], "session": f"session_id"}, config)
    # print(response["messages"][-1].content)