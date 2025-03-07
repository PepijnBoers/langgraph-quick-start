import sqlite3
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import Any, Annotated

@tool
def sql_tool(tool_call_id: Annotated[str, InjectedToolCallId], query:str):
    """Function to query a database table."""
    conn = sqlite3.connect("polisvergelijker.db")
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    conn.close()

    # Create the Markdown table
    markdown_table = "| " + " | ".join(column_names) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(column_names)) + " |\n"

    for row in results:
        markdown_table += "| " + " | ".join(map(str, row)) + " |\n"

    # return markdown_table

    return Command(
        update={
            # update the state keys
            "tiles": {"name": "test-name", "price": 123.45, "url": "https://nu.nl"},
            # update the message history
            "messages": [
                ToolMessage(
                    markdown_table, tool_call_id=tool_call_id
                )
            ],
        }
    )

