from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.prebuilt import InjectedState
from langchain.agents import create_agent, AgentState
from langchain_core.runnables.config import RunnableConfig
from langchain.messages import HumanMessage, AIMessageChunk, ToolMessage, RemoveMessage
from langchain.tools import tool, InjectedToolCallId
from langchain.agents.middleware import before_agent
import chainlit as cl

from dotenv import load_dotenv
from groq import APIStatusError
from datetime import datetime, UTC
from langchain_groq import ChatGroq
from typing import Dict, Any, Annotated
from tavily import TavilyClient

from state import DeepAgentState, TODO
from prompts import WRITE_TODOS_DESCRIPTION, TODO_USAGE_INSTRUCTIONS, SIMPLE_RESEARCH_INSTRUCTIONS
from langgraph.types import Command

load_dotenv()

tavily_client = TavilyClient()

@tool(description=WRITE_TODOS_DESCRIPTION, parse_docstring=True)
def write_todos(
        todos: list[TODO], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Create or update the agent's TODO list for task planning and tracking.

    Args:
        todos: List of Todo items with content and status
        tool_call_id: Tool call identifier for message response

    Returns:
        Command to update agent state with new TODO list
    """
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )

def read_todos(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Read the current TODO list from the agent state.

    This tool allows the agent to retrieve and review the current TODO list
    to stay focused on remaining tasks and track progress through complex workflows.

    Args:
        state: Injected agent state containing the current TODO list
        tool_call_id: Injected tool call identifier for message tracking

    Returns:
        Formatted string representation of the current TODO list
    """
    todos = state.get("todos", [])
    if not todos:
        return "No todos currently in the list."
    
    result = "Current TODO List:\n"
    for i, todo in enumerate(todos, 1):
        status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
        emoji = status_emoji.get(todo["status"], "❓")
        result += f"{i}. {emoji} {todo['content']} ({todo['status']})\n"

    return result.strip()

@tool(parse_docstring=True)
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information on a specific topic.
    
    This tool performs web searches and returns relevant results
    for the given query. Use this when you need to gather information from
    the internet about any topic.

    Args:
        query: The search query string. Be specific and clear about what
                information you're looking for.

    Returns:
        Search results from search engine.
        
    Example:
        web_search("machine learning applications in healthcare")
    """

    return tavily_client.search(query, max_results=3)

tools = [web_search, write_todos, read_todos]

memory = InMemorySaver()

@before_agent
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Remove all the tool messages from the state"""
    messages = state["messages"]

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    
    return {"messages": [RemoveMessage(id=m.id) for m in tool_messages]}


@cl.on_chat_start
async def start():

    model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    app = create_agent(
        model=model,
        tools=tools,
        checkpointer=memory,
        middleware=[
            trim_messages
        ],
        state_schema=DeepAgentState,
        system_prompt=TODO_USAGE_INSTRUCTIONS.format(system_time=datetime.now(tz=UTC).isoformat())
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + SIMPLE_RESEARCH_INSTRUCTIONS
    )

    cl.user_session.set("agent", app)

    await cl.Message(
        content="👋 Hello! I'm an AI agent with access to Web Search. I can help you with:\n\n"
                "🌤️ **Weather information** - Ask about weather in any location\n"
                "🔢 **Latest News** - latest National or International News\n"
                "🔍 **Web searches** - Search for information\n\n"
                "How can I assist you today?",
    ).send()

@cl.on_message
async def main(message: cl.Message):

    app = cl.user_session.get("agent")

    try:
        answer = cl.Message(content="")
        await answer.send()

        config: RunnableConfig = {
            "configurable": {"thread_id": cl.context.session.thread_id}
        }
    
        # Stream the agent's response
        for event in app.stream(
            {"messages": [HumanMessage(content=message.content)]},
            config,
            stream_mode="messages",
        ):
            msg = event[0]
            if isinstance(msg, AIMessageChunk) and msg.content:
                answer.content += msg.content
                await answer.update()

            if isinstance(msg, AIMessageChunk) and msg.tool_calls:
                tool_name = msg.tool_calls[0]["name"]
                answer.content += f"\n\n{tool_name}\n"
    except APIStatusError as e:
        print(e)
        if e.status_code == 429:
            await cl.Message(
                content="⚠️ Too many requests"
            ).send()
    except Exception as e:
        await cl.Message(
            content="Something went wrong"
        ).send()


@cl.on_chat_end
async def end():
    """Handle chat end."""
    await cl.Message(content="👋 Goodbye! Feel free to start a new chat anytime.").send()