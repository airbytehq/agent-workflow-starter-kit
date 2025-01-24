import os
import re

import pytest
from deepeval import assert_test  # type: ignore[import-untyped]
from deepeval.metrics import GEval  # type: ignore[import-untyped]
from deepeval.test_case import (  # type: ignore[import-untyped]
    LLMTestCase,
    LLMTestCaseParams,
)
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)

from app.agents.agent import Agent
from app.agents.flowchart import Flowchart
from app.agents.tool_router import ToolRouter

correctness_metric = GEval(
    threshold=0.5,
    model="gpt-4o",
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_steps=[
        """
Check whether the facts and semantic intent in 'actual output' contradicts any facts or semantic intent in 'expected output'.
It is OK for the 'actual output' to be more detailed than the 'expected output'. The `expected output` will likely be much more simplistic.

Scoring:
Heavily penalize omission of factual details.
Do NOT penalize for including additional or excessive detail or not having the simplicity of the 'expected output'.
"""
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
)

output_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "output")
)


async def run_test(cls: type[Agent], input: str, expected_output: str) -> None:
    print(f"EVAL ({cls.__name__}) -----------------------")
    print(f"----INPUT: {input}")
    seed = await cls.get_initial_prompt()
    all_messages = seed + [ChatMessage(role=MessageRole.USER, content=input)]
    agent = cls(message_history=all_messages)

    actual_output = await agent.answer()
    print(f"----OUTPUT: {actual_output}")
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
    )

    await correctness_metric.a_measure(test_case)
    print(f"----SCORE: {correctness_metric.score} {correctness_metric.reason}")
    assert agent._last_workflow

    filename = f"{re.sub(r'[^a-zA-Z0-9_]', '_', input).lower()}_{cls.__name__}.html"
    draw_most_recent_execution(agent._last_workflow, os.path.join(output_dir, filename))

    assert_test(test_case, [correctness_metric])


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [Flowchart, ToolRouter])
async def test_correctness_llm(cls: type[Agent]) -> None:
    await run_test(
        cls,
        "What is the capital of Spain?",
        "The capital of Spain is Madrid.",
    )
    filename = f"full_{cls.__name__}.html"
    agent = cls(message_history=[])
    draw_all_possible_flows(agent._workflow(), os.path.join(output_dir, filename))


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [Flowchart, ToolRouter])
async def test_correctness_single(cls: type[Agent]) -> None:
    await run_test(
        cls,
        "What is our largest won opportunity?",
        "The largest won opportunity is named United Oil Refinery Generators with an amount of $915,000.",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [Flowchart, ToolRouter])
async def test_answer_relevancy_multi(cls: type[Agent]) -> None:
    await run_test(
        cls,
        "Do we have more users in USA or EU?",
        "We have more users in the USA than in the EU.",
    )
