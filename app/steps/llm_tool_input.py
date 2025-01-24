from typing import cast

from llama_index.core.llms import ChatResponse
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import (
    Context,
    StopEvent,
)

from app.agents.chat_history import ChatHistory
from app.agents.types import (
    StreamResponseEvent,
    ToolCallEvent,
    make_stop_event,
)
from app.tools.tool_base import ToolBase, ToolBaseType


async def llm_tool_input(
    ctx: Context,
    history: ChatHistory,
    llm: LLM,
    tools: list[ToolBaseType],
) -> ToolCallEvent | StopEvent:
    llm_tools = ToolBase.get_tool_definitions(tools or [])
    chat_history = history.get()

    if len(chat_history) == 0:
        raise ValueError(
            "No chat history! We probably ran out of context window with a recentlarge message."
        )

    tool_llm: FunctionCallingLLM | None = None
    if len(llm_tools) > 0:
        assert llm.metadata.is_function_calling_model
        tool_llm = cast(FunctionCallingLLM, llm)

    if tool_llm:
        chat_response_gen = await tool_llm.astream_chat_with_tools(
            llm_tools, chat_history=chat_history
        )
    else:
        chat_response_gen = await llm.astream_chat(messages=chat_history)

    last_response: ChatResponse | None = None
    async for response in chat_response_gen:
        last_response = response

        if response.delta:
            # not a tool call, return early to stream
            ctx.write_event_to_stream(StreamResponseEvent(response=response))

    if not last_response:
        raise ValueError("No response from LLM")

    history.add(last_response.message)

    if tool_llm:
        # now it's all done?
        tool_calls = tool_llm.get_tool_calls_from_response(
            last_response, error_on_no_tool_call=False
        )
        if tool_calls:
            return ToolCallEvent(tool_calls=tool_calls)

    return make_stop_event(history)
