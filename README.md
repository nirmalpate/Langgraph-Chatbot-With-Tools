# Langgraph-Chatbot-With-Tools
Langgraph AI ChatBot With Tools (Wiki ..)

# LangGraph Chatbot With Tools

A conversational chatbot built with **LangGraph**, powered by an LLM hosted on **Groq**, that can search **Wikipedia** and **Arxiv** when it needs external information. Tracing is handled via **LangSmith**.

## Overview

This project extends a basic LangGraph chatbot by giving it tools. Instead of only answering from what the LLM already knows, the bot can decide mid-conversation to look something up on Wikipedia or Arxiv, then use those results to form its answer.

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — graph-based orchestration framework for LLM apps
- [LangChain](https://github.com/langchain-ai/langchain) — core LLM app framework
- [Groq](https://groq.com/) — fast LLM inference (via `langchain_groq`)
- [LangSmith](https://smith.langchain.com/) — tracing and observability
- [Wikipedia](https://pypi.org/project/wikipedia/) & [arxiv](https://pypi.org/project/arxiv/) — tool data sources

## Setup

### 1. Install dependencies

```bash
pip install langgraph langsmith langchain langchain_groq langchain_community
pip install arxiv wikipedia
```

### 2. Set up API keys

You'll need:
- A **Groq API key** — get one at [console.groq.com](https://console.groq.com)
- A **LangSmith API key** — get one at [smith.langchain.com](https://smith.langchain.com) (Settings → API Keys)

In Google Colab, store these as secrets (key icon in the sidebar).

```python
from google.colab import userdata

groq_api_key = userdata.get("groq1api1key")
langsmith = userdata.get("LANGSMITH_API_KEY")
```

### 3. Enable tracing

```python
import os

os.environ["LANGCHAIN_API_KEY"] = langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "CourseLanggraph"
```

## Setting Up the Tools

```python
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun

# Arxiv tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Wikipedia tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

tools = [wiki_tool, arxiv_tool]
```

- `top_k_results=1` — only return the single most relevant result per search
- `doc_content_chars_max=2000` — cap each result to 2000 characters, so the LLM's context doesn't get flooded

## Building the Graph

### Define the LLM and bind the tools to it

```python
from langchain_groq import ChatGroq

llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")
llm_with_tools = llm.bind_tools(tools=tools)
```

> ⚠️ Groq's available models change over time. Verify the current list for your account before hardcoding a model name:
> ```python
> import requests
>
> url = "https://api.groq.com/openai/v1/models"
> headers = {"Authorization": f"Bearer {groq_api_key}"}
> response = requests.get(url, headers=headers)
> for m in response.json()["data"]:
>     print(m["id"])
> ```
> Make sure to pick a chat/instruction model (e.g. `openai/gpt-oss-20b`, `openai/gpt-oss-120b`) — not a classifier model like `meta-llama/llama-prompt-guard-2-*`, which only outputs a score, not a reply.

### Define the state

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

### Define the chatbot node

```python
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
```

### Wire up the graph with tool routing

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()
```

### How the flow works

```
START → chatbot → (tools_condition decides)
                       ├── needs a tool → tools → back to chatbot → ...
                       └── no tool needed → END
```

- `chatbot` sends the conversation to the LLM.
- `tools_condition` checks the LLM's response: did it ask to call a tool?
  - If yes → routes to the `tools` node, which actually runs Wikipedia/Arxiv and returns the result to `chatbot`.
  - If no → the conversation ends and the LLM's reply is the final answer.

## Running the Chatbot

```python
user_input = "Hi there!, My name is John"

events = graph.stream(
    {"messages": [("user", user_input)]}, stream_mode="values"
)

for event in events:
    event["messages"][-1].pretty_print()
```

- `stream_mode="values"` returns the **full state** after each step (rather than just a diff), so `event["messages"][-1]` reliably gives the latest message.
- `.pretty_print()` neatly formats each message, including tool calls/results when a search happens along the way.

## Notes

- All conversation history is kept in memory only for the current session (not persisted). For persistence across restarts, use LangGraph's checkpointer feature with a real database (e.g. SQLite/Postgres).
- Every run is automatically traced in LangSmith under the `CourseLanggraph` project.
- The bot can loop through `chatbot → tools → chatbot` more than once in a single turn if it needs multiple searches before answering.

## License

MIT
