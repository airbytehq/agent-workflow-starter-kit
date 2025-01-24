# https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/

import os

from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.workflow import StartEvent, Workflow, step
from llama_index.core.workflow.handler import WorkflowHandler

# log workflow steps in the UI
from llama_index.core.workflow.workflow import dispatcher
from llama_index.llms.openai import OpenAI  # type: ignore[import-untyped]

from app.agents.chat_history import ChatHistory
from app.agents.types import (
    InitialChatEvent,
)
from app.instrument import ChainlitWorkflowSpanHandler, get_callback_manager
from app.steps.start_to_initial import start_to_input

dispatcher.add_span_handler(ChainlitWorkflowSpanHandler())


class WorkflowBase(Workflow):
    def __init__(
        self,
        llm: LLM | None = None,
        model: str | None = None,
        timeout: int = 120,
        verbose: bool = True,
    ) -> None:
        super().__init__(timeout=timeout, verbose=verbose)
        self.llm = llm or OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=model or "gpt-4o-mini",
            callback_manager=get_callback_manager(),
        )
        self.history: ChatHistory = ChatHistory(llm=self.llm)

    def run_with_chat_history(self, chat_history: list[ChatMessage]) -> WorkflowHandler:
        return self.run(chat_history=chat_history)  # type: ignore[no-any-return]

    def add_message(self, message: ChatMessage) -> None:
        self.history.add(message)

    @step
    def make_initial_event(self, ev: StartEvent) -> InitialChatEvent:
        return start_to_input(ev, self.history)
