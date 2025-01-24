import chainlit as cl
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.llms import ChatMessage

from app.agents.tool_router import ToolRouter as AgentToUse

# from app.agents.flowchart import Flowchart as AgentToUse


async def get_message_history() -> list[ChatMessage]:
    message_history: list[ChatMessage] | None = cl.user_session.get("message_history")
    if not message_history:
        message_history = await AgentToUse.get_initial_prompt()
        cl.user_session.set("message_history", message_history)
    return message_history


def set_message_history(message_history: list[ChatMessage]) -> None:
    cl.user_session.set("message_history", message_history)


@cl.on_chat_start
async def start_chat() -> None:
    await get_message_history()
    welcome_message = await AgentToUse.get_welcome_message()
    if welcome_message:
        await cl.Message(content=welcome_message).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    message_history = await get_message_history()
    message_history.append(ChatMessage(role=MessageRole.USER, content=message.content))

    agent = AgentToUse(message_history=message_history)

    msg = cl.Message(content="")
    await msg.send()

    async for token in agent.stream():
        await msg.stream_token(token)

    message_history = agent.get_message_history()
    set_message_history(message_history)

    await msg.update()
