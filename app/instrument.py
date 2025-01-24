import asyncio
import inspect
import json
import re
import uuid
from datetime import datetime
from typing import Any

import pytz  # type: ignore[import-untyped]
from chainlit import Step
from chainlit.context import get_context
from literalai.observability.step import TrueStepType
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.instrumentation.span_handlers.null import NullSpanHandler
from pydantic import BaseModel, Field, PrivateAttr

LOGGING_ENABLED = False


def has_chainlit() -> bool:
    try:
        get_context()
        return True
    except Exception:
        return False


def utc_now() -> str:
    dt = datetime.now(pytz.UTC)
    return dt.isoformat() + "Z"


class ChatStep(BaseModel):
    name: str
    type: TrueStepType
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    language: str | None = "text"
    input: str | None = None
    output: str | None = None

    _cl_step: Step | None = PrivateAttr(default=None)

    async def send(self) -> None:
        if not has_chainlit():
            return await self.log()

        step = self._cl_step
        if not step:
            step = Step(
                name=self.name,
                type=self.type,
                id=self.id,
                language=self.language,
            )
            self._cl_step = step

        if self.input:
            step.input = self.input
        if self.output:
            step.output = self.output

        await step.send()  # type: ignore[no-untyped-call]

    async def update(self) -> None:
        if not has_chainlit():
            return await self.log()

        step = self._cl_step
        if step:
            await step.update()  # type: ignore[no-untyped-call]

    async def log(self) -> None:
        # not using chainlit. log it.
        if not LOGGING_ENABLED:
            return
        print(f"ChatStep: {self.name} {self.type} {self.id}  {self.language}")
        if self.input:
            print(f"    ->   Input: {self.input}")
        if self.output:
            print(f"    ->   Output: {self.output}")


class LlamaCallback(BaseCallbackHandler):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[],
        )
        self.steps: dict[str, ChatStep] = {}

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.
            parent_id (str): parent event id.
        """

        llm_input: str | None = None
        name: str = event_type
        if event_type is CBEventType.LLM and payload:
            if EventPayload.PROMPT in payload:
                prompt = str(payload[EventPayload.PROMPT])
                llm_input = json.dumps({"prompt": prompt})
            else:
                pydantic_messages = payload.get(EventPayload.MESSAGES, [])
                messages = [m.model_dump() for m in pydantic_messages]

                additional_kwargs = payload.get(EventPayload.ADDITIONAL_KWARGS, {})

                llm_input = json.dumps(
                    {"messages": messages, "additional_kwargs": additional_kwargs},
                    indent=2,
                )

            serialized = payload.get(EventPayload.SERIALIZED, None)
            if serialized:
                model_name = serialized["model"] if serialized["model"] else None
                if model_name:
                    name = model_name

        if not llm_input:
            return ""

        key = f"{event_type}/{event_id}"

        step = self.steps.get(key)
        if not step:
            step = ChatStep(
                name=name,
                type="llm",
                id=event_id,
                language="json",
            )
            self.steps[key] = step
        step.input = llm_input

        asyncio.create_task(step.send())
        return ""

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        key = f"{event_type}/{event_id}"

        step = self.steps.get(key)
        if not step:
            return

        llm_output: str | None = None
        if event_type is CBEventType.LLM and payload:
            if EventPayload.PROMPT in payload:
                completion = str(payload[EventPayload.COMPLETION])
                llm_output = json.dumps({"completion": completion})
            else:
                response_model = payload[EventPayload.RESPONSE]
                message_model = response_model.message
                message = message_model.model_dump()
                additional_kwargs = response_model.additional_kwargs

                llm_output = json.dumps(
                    {"messages": message, "additional_kwargs": additional_kwargs},
                    indent=2,
                )

        if llm_output:
            step.output = llm_output

        asyncio.create_task(step.update())

        del self.steps[key]

    def start_trace(self, trace_id: str | None = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        pass


def get_callback_manager() -> CallbackManager:
    return CallbackManager([LlamaCallback()])


class ChainlitWorkflowSpanHandler(NullSpanHandler):
    steps: dict[str, ChatStep] = Field(default_factory=dict)

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "ChainlitWorkflowSpanHandler"

    @staticmethod
    def remove_uuid(id_: str) -> str:
        # input like Used FlowchartWorkflow.query_database-0ed36881-a352-4bcb-b432-b6c81d7d93e2
        regex = r"-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        return re.sub(regex, "", id_)

    def span_enter(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        parent_id: str | None = None,
        tags: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        key = id_
        name = self.remove_uuid(id_)
        # chainlit avatars have to match [a-zA-Z0-9_ -]
        name = re.sub(r"[^a-zA-Z0-9_ -]", " ", name)

        step = self.steps.get(key)
        if not step:
            step = ChatStep(
                name=name,
                type="run",
                id=id_,
                # language="json",
            )
            self.steps[key] = step

        step.input = str(bound_args)

        asyncio.create_task(step.send())

    def span_exit(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        result: Any | None = None,
        **kwargs: Any,
    ) -> None:
        key = id_

        step = self.steps.get(key)
        if not step:
            return

        step.output = str(result)

        asyncio.create_task(step.update())
