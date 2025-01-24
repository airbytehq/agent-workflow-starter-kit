import json

import pytest

from app.tools.query_database import QueryDatabaseTool


@pytest.mark.asyncio
async def test_query_contacts_simple() -> None:
    tool = QueryDatabaseTool(query="SELECT * FROM contact LIMIT 10")
    response = await tool.run()
    assert response.query_result_rows is not None
    # print(response.query_result_rows)
    assert len(response.query_result_rows) == 10


@pytest.mark.asyncio
async def test_query_contacts_count() -> None:
    tool = QueryDatabaseTool(query="SELECT COUNT(*) AS count FROM contact")
    response = await tool.run()
    assert response.query_result_rows is not None
    # print(response.query_result_rows)
    assert len(response.query_result_rows) == 1
    assert response.query_result_rows[0]["count"] >= 10


@pytest.mark.asyncio
async def test_query_contacts_with_embedding() -> None:
    tool = QueryDatabaseTool(
        query="""
        SELECT contact.*, contact__documents.document,
            ARRAY_COSINE_SIMILARITY(embedding('Ashley'), document_embedded) AS score
        FROM contact JOIN contact__documents ON contact.id = contact__documents.record_id
        WHERE score > 0.01 ORDER BY score DESC LIMIT 10
        """
    )
    response = await tool.run()
    assert response.query_result_rows is not None
    # print(response.query_result_rows)
    assert len(response.query_result_rows) == 10
    assert "Ashley" in json.dumps(response.query_result_rows[0])
