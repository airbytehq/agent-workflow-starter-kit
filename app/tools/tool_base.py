import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from llama_index.core.tools.types import (
    AsyncBaseTool,
    BaseTool,
    ToolMetadata,
    ToolOutput,
)
from pydantic import BaseModel

from app.instrument import ChatStep


class ToolResponseBase(BaseModel):
    pass


# Define a type variable 'T' for the response type that extends from BaseModel
ToolResponseType = TypeVar("ToolResponseType", bound=ToolResponseBase)


class ToolBase(BaseModel, ABC, Generic[ToolResponseType]):
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    prompt_description: ClassVar[str] = (
        ""  # if there is more to say in the initial prompt
    )

    @abstractmethod
    async def _perform_action(self) -> ToolResponseType:
        """This method must be implemented by subclasses to perform the main action."""
        pass

    async def run(self) -> ToolResponseType:
        name = f"Tool.{self.name}"
        # chainlit avatars have to match [a-zA-Z0-9_ -]
        name = re.sub(r"[^a-zA-Z0-9_ -]", " ", name)

        step = ChatStep(type="tool", name=name, language="json")
        step.input = self.model_dump_json()
        await step.send()

        result = await self._perform_action()

        step.output = result.model_dump_json()
        await step.update()
        return result

    @classmethod
    def get_metadata(cls) -> tuple[str, str]:
        name = cls.name
        description = cls.description

        if not name or not description:
            raise ValueError(
                """
                title and description are required.

                class GetCurrentWeatherTool(ToolBase[WeatherResponse]):
                    name = "get_current_weather"
                    description = "Get the current weather in a given location"
                """
            )
        return name, description

    @classmethod
    def base_tool(cls) -> BaseTool:
        return MyAsyncBaseTool(cls)

    @staticmethod
    def get_tool_definitions(tools: list["ToolBaseType"]) -> list[BaseTool]:
        out: list[BaseTool] = []
        for Model in tools:
            out.append(Model.base_tool())
        return out


class MyAsyncBaseTool(AsyncBaseTool, Generic[ToolResponseType]):
    def __init__(self, Model: type[ToolBase[ToolResponseType]]):
        self.Model = Model

    @property
    def metadata(self) -> ToolMetadata:
        name, description = self.Model.get_metadata()
        return ToolMetadata(
            name=name,
            description=description,
            fn_schema=self.Model,
            return_direct=False,
        )

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        raise NotImplementedError(
            "This method should not be called directly. Call Async method instead!"
        )

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        instance = self.Model(**kwargs)
        tool_output = await instance.run()
        name, _ = self.Model.get_metadata()
        return ToolOutput(
            content=tool_output.model_dump_json(),
            tool_name=name,
            raw_input=kwargs,
            raw_output=tool_output.model_dump(),
        )


ToolBaseType = type[ToolBase[Any]]
