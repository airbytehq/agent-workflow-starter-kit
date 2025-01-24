from llama_index.core.base.llms.types import MessageRole
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection
from llama_index.core.tools.types import AsyncBaseTool

from app.agents.chat_history import ChatHistory
from app.agents.types import (
    LlmInputEvent,
)
from app.tools.tool_base import ToolBase, ToolBaseType


async def tool_call(
    tool_calls: list[ToolSelection],
    history: ChatHistory,
    tools: list[ToolBaseType],
) -> LlmInputEvent:
    llm_tools = ToolBase.get_tool_definitions(tools or [])
    tools_by_name = {tool.metadata.get_name(): tool for tool in llm_tools}

    tool_msgs = []

    # call tools
    for tool_call in tool_calls:
        tool = tools_by_name.get(tool_call.tool_name)
        additional_kwargs = {
            "tool_call_id": tool_call.tool_id,
            "name": tool.metadata.get_name() if tool else tool_call.tool_name,
        }

        if not tool:
            tool_msgs.append(
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=f"Tool {tool_call.tool_name} does not exist",
                    additional_kwargs=additional_kwargs,
                )
            )
            continue

        try:
            if isinstance(tool, AsyncBaseTool):
                tool_output = await tool.acall(**tool_call.tool_kwargs)
            else:
                tool_output = tool(**tool_call.tool_kwargs)

            tool_msgs.append(
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=tool_output.content,
                    additional_kwargs=additional_kwargs,
                )
            )
        except Exception as e:
            tool_msgs.append(
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=f"Encountered error in tool call. You can try again with different inputs. But give up after 3 times.\nError: {e}",
                    additional_kwargs=additional_kwargs,
                )
            )

    for msg in tool_msgs:
        history.add(msg)

    return LlmInputEvent()
