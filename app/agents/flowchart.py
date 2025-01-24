from typing import Literal

from llama_index.core.base.llms.types import MessageRole
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Context,
    Event,
    StopEvent,
    step,
)
from pydantic import BaseModel

from app.agents.agent import Agent
from app.agents.types import (
    InitialChatEvent,
)
from app.agents.workflow_base import WorkflowBase
from app.steps.llm_input import handle_llm_input
from app.steps.llm_structured_output import llm_structured_output
from app.tools.query_database import QueryDatabaseTool
from app.tools.tool_base import ToolBase

MAX_QUERY_ATTEMPTS = 5


class PickApproachEvent(Event):
    pass  # implicit history


class ChatApproachEvent(Event):
    pass  # implicit history


class SemanticApproachEvent(Event):
    pass  # implicit history


class QueryApproachEvent(Event):
    # implicit history
    attempt: int = 0


class SelectedApproach(BaseModel):
    approach: Literal[
        "simple_asnwer_from_llm",
        "semantic_search",
        "query_database",
    ]


class DecideOnQuery(BaseModel):
    query: str


class FlowchartWorkflow(WorkflowBase):
    tools = ToolBase.get_tool_definitions([QueryDatabaseTool])

    @step
    def handle_initial_event(self, ev: InitialChatEvent) -> PickApproachEvent:
        return PickApproachEvent()

    @step
    async def pick_approach(
        self, ev: PickApproachEvent
    ) -> ChatApproachEvent | SemanticApproachEvent | QueryApproachEvent:
        message = ChatMessage.from_str(
            "What approach should we take to answer the users question and given the information we already have?",
            role=MessageRole.ASSISTANT,
        )
        self.history.add(message)
        approach = await llm_structured_output(self.llm, SelectedApproach, self.history)

        if approach.approach == "simple_asnwer_from_llm":
            return ChatApproachEvent()
        elif approach.approach == "semantic_search":
            return SemanticApproachEvent()
        elif approach.approach == "query_database":
            return QueryApproachEvent()
        else:
            raise ValueError(f"Unknown approach: {approach.approach}")

    @step
    async def chat_response(self, ctx: Context, ev: ChatApproachEvent) -> StopEvent:
        return await handle_llm_input(ctx, self.history, self.llm)

    @step
    async def semantic_search(
        self, ev: SemanticApproachEvent
    ) -> ChatApproachEvent | SemanticApproachEvent:
        message = ChatMessage.from_str(
            "Currently can't do semantic searchs. Tell user you can't help them.",
            role=MessageRole.ASSISTANT,
        )
        self.history.add(message)
        return ChatApproachEvent()  # let it respond nicely

    @step
    async def query_database(
        self, ev: QueryApproachEvent
    ) -> ChatApproachEvent | QueryApproachEvent:
        message = ChatMessage.from_str(
            f"""What query should we run to answer the user's question? Only use follwoing tables: contact, opportunity, account.
            Do not retry again with the same queries you have already tried if they did not work.
            -----
                {QueryDatabaseTool.prompt_description}
            -----
            """,
            role=MessageRole.ASSISTANT,
        )
        self.history.add(message)
        response = await llm_structured_output(self.llm, DecideOnQuery, self.history)
        query = response.query

        try:
            results = await QueryDatabaseTool(query=query).run()

            if len(results.query_result_rows) > 0:
                message = ChatMessage.from_str(
                    f"We made this query (do not share with user):\n```{query}```\n\nWe found the following results: \n```{results.model_dump_json()}```",
                    role=MessageRole.ASSISTANT,
                )
                self.history.add(message)
                return ChatApproachEvent()  # let it respond based on what we found
            elif ev.attempt < MAX_QUERY_ATTEMPTS:
                message = ChatMessage.from_str(
                    "We found no results. You should try again with a new query. This is the query (do not share with user) we tried:\n```{query}```",
                    role=MessageRole.ASSISTANT,
                )
                self.history.add(message)
                return QueryApproachEvent(attempt=ev.attempt + 1)  # query again
            else:
                message = ChatMessage.from_str(
                    "We found no results. Let the user know.",
                    role=MessageRole.ASSISTANT,
                )
                self.history.add(message)
                return ChatApproachEvent()  # let it respond nicely
        except Exception as e:
            if ev.attempt < MAX_QUERY_ATTEMPTS:
                message = ChatMessage.from_str(
                    f"Error querying the database: {e}.\n\nThis is the query (do not share with user) we tried:\n```{query}```\n You should alter it to get it to work.",
                    role=MessageRole.ASSISTANT,
                )
                self.history.add(message)
                return QueryApproachEvent(attempt=ev.attempt + 1)  # query again
            else:
                message = ChatMessage.from_str(
                    "We can not seem to get a good query. Let the user know.\n\nThis is the query (do not share with user) we tried:\n```{query}```",
                    role=MessageRole.ASSISTANT,
                )
                self.history.add(message)
                return ChatApproachEvent()  # let it respond nicely


class Flowchart(Agent):
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
        return FlowchartWorkflow()
