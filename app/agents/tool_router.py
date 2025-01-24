from llama_index.core.base.llms.types import MessageRole
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Context,
    StopEvent,
    step,
)

from app.agents.agent import Agent
from app.agents.types import (
    InitialChatEvent,
    LlmInputEvent,
    ToolCallEvent,
)
from app.agents.workflow_base import WorkflowBase
from app.steps.llm_tool_input import llm_tool_input
from app.steps.tool_call import tool_call
from app.tools.query_database import QueryDatabaseTool
from app.tools.tool_base import ToolBaseType


class ToolRouterWorkflow(WorkflowBase):
    tools: list[ToolBaseType] = [QueryDatabaseTool]

    @step
    def handle_initial_event(self, ev: InitialChatEvent) -> LlmInputEvent:
        return LlmInputEvent()

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: LlmInputEvent
    ) -> ToolCallEvent | StopEvent:
        return await llm_tool_input(ctx, self.history, self.llm, self.tools)

    @step
    async def handle_tool_calls(self, ev: ToolCallEvent) -> LlmInputEvent:
        return await tool_call(ev.tool_calls, self.history, self.tools)


class ToolRouter(Agent):
    @classmethod
    async def get_initial_prompt(cls) -> list[ChatMessage]:
        return [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"""
                You are a helpful assistant working with CRM data to help the user navigate it.
                You have various tools available to you.

                All queries should get a limit of 10 or less rows to not break the system.
                -----
                {QueryDatabaseTool.prompt_description}
                -----
                """,
            )
        ]

    def _workflow(self) -> WorkflowBase:
        return ToolRouterWorkflow()
