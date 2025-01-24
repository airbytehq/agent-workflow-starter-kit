from typing import TypeVar

from llama_index.core.llms import LLM
from pydantic import BaseModel

from app.agents.chat_history import ChatHistory

Model = TypeVar("Model", bound=BaseModel)


async def llm_structured_output(
    llm: LLM,
    output_cls: type[Model],
    history: ChatHistory,
) -> Model:
    chat_history = history.get()

    sllm = llm.as_structured_llm(output_cls=output_cls)
    response = await sllm.achat(chat_history)

    # get actual object
    output_obj = response.raw
    if not isinstance(output_obj, output_cls):
        raise ValueError(f"Expected {output_cls}, got {type(output_obj)}")

    return output_obj
