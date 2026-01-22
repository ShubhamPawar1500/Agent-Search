from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain.agents import create_agent, AgentState
from langchain_core.runnables.config import RunnableConfig
from langchain.messages import HumanMessage, AIMessageChunk, ToolMessage, RemoveMessage
from langchain.tools import tool
from langchain.agents.middleware import before_agent
import chainlit as cl

from dotenv import load_dotenv
from groq import APIStatusError
from datetime import datetime, UTC
from langchain_groq import ChatGroq
from typing import Dict, Any
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """
    Search the web for information about a query.
    You MUST always provide a non-empty `query` string.
    """

    return tavily_client.search(query, max_results=3)

tools = [web_search]

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
        system_prompt="""

        You have access to a tool that retrieves information from a web. Use the tool to help answer user queries if needed.
        If you decide to use a tool, you MUST supply all required parameters. Never call a tool with missing or empty arguments.
        Be concise and helpful.

        System time: {system_time}
        """.format(
            system_time=datetime.now(tz=UTC).isoformat()
        )
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