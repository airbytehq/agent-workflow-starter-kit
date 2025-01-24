from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.tools import ToolOutput, ToolSelection
from llama_index.core.workflow import (
    Event,
    StopEvent,
)
from pydantic import BaseModel

from app.agents.chat_history import ChatHistory


class StreamResponseEvent(Event):
    response: ChatResponse


class StopEventResult(BaseModel):
    chat_history: list[ChatMessage]


class LlmInputEvent(Event):
    pass  # implicit history


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput


class InitialChatEvent(Event):
    pass  # implicit history


def make_stop_event(history: ChatHistory) -> StopEvent:
    return StopEvent(result=StopEventResult(chat_history=history.get()))
