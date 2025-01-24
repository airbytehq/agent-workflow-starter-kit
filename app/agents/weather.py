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
from app.tools.get_current_weather import GetCurrentWeatherTool
from app.tools.tool_base import ToolBaseType


class WeatherWorkflow(WorkflowBase):
    tools: list[ToolBaseType] = [GetCurrentWeatherTool]

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


class WeatherAgent(Agent):
    def _workflow(self) -> WeatherWorkflow:
        return WeatherWorkflow()
