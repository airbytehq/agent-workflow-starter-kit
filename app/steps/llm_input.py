from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import (
    Context,
    StopEvent,
)

from app.agents.chat_history import ChatHistory
from app.steps.llm_tool_input import llm_tool_input


async def handle_llm_input(
    ctx: Context,
    history: ChatHistory,
    llm: LLM,
) -> StopEvent:
    response = await llm_tool_input(ctx, history, llm, tools=[])
    # since we didn't give any tools, we know it's a stop event
    assert isinstance(response, StopEvent)
    return response
