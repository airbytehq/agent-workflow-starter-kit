from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from io import StringIO

from llama_index.core.base.llms.types import MessageRole
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import StopEvent
from pydantic import BaseModel, PrivateAttr

from app.agents.types import StreamResponseEvent
from app.agents.workflow_base import WorkflowBase


class AgentSettings(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0


class StreamResponse(BaseModel):
    message_history: list[ChatMessage]

    # This allows arbitrary types like the stream response. maybe?
    model_config = {"arbitrary_types_allowed": True}


class Agent(BaseModel, ABC):
    message_history: list[ChatMessage]
    settings: AgentSettings = AgentSettings()

    _last_workflow: WorkflowBase | None = PrivateAttr(default=None)

    # This allows arbitrary types like the current_stream
    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    async def get_initial_prompt(cls) -> list[ChatMessage]:
        return [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant.")
        ]

    @classmethod
    async def get_welcome_message(cls) -> str | None:
        return "How can I help you?"

    def get_message_history(self) -> list[ChatMessage]:
        return self.message_history

    @abstractmethod
    def _workflow(self) -> WorkflowBase:
        raise NotImplementedError("Implement Agent._workflow")

    async def stream(self) -> AsyncGenerator[str, None]:
        agent = self._workflow()
        self._last_workflow = agent
        handler = agent.run(chat_history=self.message_history)

        async for ev in handler.stream_events():
            if isinstance(ev, StreamResponseEvent):
                if ev.response.delta:
                    yield ev.response.delta
            if isinstance(ev, StopEvent):
                self.message_history = ev.result.chat_history

    async def answer(self) -> str:
        buffer = StringIO()
        async for token in self.stream():
            buffer.write(token)
        return buffer.getvalue()

    @classmethod
    async def answer_from_message_history(
        cls, message_history: list[ChatMessage]
    ) -> str:
        seed = await cls.get_initial_prompt()
        all_messages = seed + message_history
        agent = cls(message_history=all_messages)
        return await agent.answer()

    @classmethod
    async def answer_from_query(cls, query: str) -> str:
        history = [ChatMessage(role=MessageRole.USER, content=query)]
        return await cls.answer_from_message_history(history)
