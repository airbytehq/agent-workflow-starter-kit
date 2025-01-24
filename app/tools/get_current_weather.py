from typing import Literal

from pydantic import Field

from app.tools.tool_base import ToolBase, ToolResponseBase


class WeatherResponse(ToolResponseBase):
    location: str
    unit: Literal["fahrenheit", "celsius"]
    temperature: str
    forecast: list[str]


class GetCurrentWeatherTool(ToolBase[WeatherResponse]):
    name = "get_current_weather"
    description = "Get the current weather in a given location"

    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: Literal["fahrenheit", "celsius"] = Field(
        default="fahrenheit", description="The unit of temperature"
    )

    async def _perform_action(self) -> WeatherResponse:
        return WeatherResponse(
            location=self.location,
            temperature="60",
            unit=self.unit,
            forecast=["windy"],
        )
