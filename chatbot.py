from os import name
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

import json
from langchain_core.messages import ToolMessage

tool = TavilySearch(max_results=2)
# tools = [tool]
# print(tool.invoke("What's a 'node' in a LangGraph?"))
llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)

# Create a StateGraph
class State(TypedDict):
  messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Add a node

def chatbot(state: State):
  return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

# Add an entry point
graph_builder.add_edge(START, "chatbot")

class BasicToolNode:

  def __init__(self, tools: list) -> None:
    self.tools_by_name = {tool.name: tool for tool in tools}

  def __call__(self, inputs: dict):
    if messages := inputs.get("messages", []):
      message = messages[-1]
    else:
      raise ValueError("No message found in input")
    outputs = []
    for tool_call in message.tool_calls:
      tool_result = self.tools_by_name[tool_call["name"]].invoke(
        tool_call["args"]
      )
      outputs.append(
        ToolMessage(
          content=json.dumps(tool_result),
          name=tool_call["name"],
          id=tool_call["id"],
        )
      )
      return {"messages": outputs}

tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# Compile
graph = graph_builder.compile()

# Run the chatbot
def stream_graph_updates(user_input: str):
  for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
    for value in event.values():
      print("Assistant:", value["messages"][-1].content)

# while True:
#   try:
#     user_input = input("User: ")
#     if user_input.lower() in ["quit", "exit", "q"]:
#       print("Goodbye!")
#       break
#     stream_graph_updates(user_input)
#   except:
#     user_input = "What do you know about LangGraph?"
#     print("User: " + user_input)
#     break

