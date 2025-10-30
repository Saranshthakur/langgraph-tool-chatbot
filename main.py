from typing import Annotated
from typing_extensions import TypedDict

# Import Arxiv and Wikipedia API wrappers and their corresponding tool classes
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun

# Initialize tools
arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
)
wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
)

# List of tools the chatbot can call
tools = [arxiv_tool, wiki_tool]

# Define conversation state using LangGraph

from langgraph.graph.message import add_messages

# State holds a list of messages between user and AI
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create a state graph
from langgraph.graph import StateGraph, START, END
graph_builder = StateGraph(State)

# Initialize LLM
from langchain_groq import ChatGroq
from google.colab import userdata

# Get API key
llm = ChatGroq(
    groq_api_key=userdata.get("groq_api_key"),
    model="gemma2-9b-it"  # Instruction-tuned Gemma 2 model
)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools=tools)

# Define chatbot node

def chatbot(state: State):
    """
    Main chatbot logic.
    Takes message history as input,
    passes it to the LLM (with tools bound),
    and returns the LLM’s latest message.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build the LangGraph workflow
from langgraph.prebuilt import ToolNode, tools_condition

# Add chatbot and tool nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

# Conditional routing: if LLM requests a tool, go to 'tools' node

graph_builder.add_conditional_edges("chatbot", tools_condition)

# After tools execute, pass results back to chatbot
graph_builder.add_edge("tools", "chatbot")

# Define start node
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()

# Simulate a single user message
result = graph.invoke({"messages": [("user", "Hi there!, What is LangGraph")]})

# final message
final_msg = result["messages"][-1]

# Print chatbot’s final response
print(final_msg.content)
