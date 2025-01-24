import json
import os
import re
from typing import Any

import duckdb
from pydantic import Field

from app.embedding.embedding_calculator import calculate_embeddings
from app.tools.tool_base import ToolBase, ToolResponseBase

# SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'contact';
PROMPT_DESCRIPTION = """
The query_contacts function uses the following schema to query the duck db. Only use the schema below to query the duck db.

table name: contact
schema:
id	VARCHAR
remote_id	VARCHAR
remote_created_at	TIMESTAMP WITH TIME ZONE
first_name	VARCHAR
last_name	VARCHAR
account	VARCHAR
owner	VARCHAR
addresses	STRUCT(street_1 VARCHAR, city VARCHAR, state VARCHAR, postal_code VARCHAR, country VARCHAR, address_type VARCHAR)[]
email_addresses	STRUCT(email_address VARCHAR, email_address_type VARCHAR)[]
phone_numbers	STRUCT(phone_number VARCHAR, phone_number_type VARCHAR)[]
last_activity_at	TIMESTAMP WITH TIME ZONE

table name: contact__documents
schema:
record_id	VARCHAR
remote_id	VARCHAR
document	VARCHAR
document_embedded	FLOAT[768]
chunk_id	INTEGER

table name: account
schema:
id	VARCHAR
remote_id	VARCHAR
owner	VARCHAR
name	VARCHAR
description	VARCHAR
industry	VARCHAR
website	VARCHAR
number_of_employees	BIGINT
addresses	STRUCT(street_1 VARCHAR, city VARCHAR, state VARCHAR, postal_code VARCHAR, country VARCHAR, address_type VARCHAR)[]
phone_numbers	STRUCT(phone_number VARCHAR, phone_number_type VARCHAR)[]
last_activity_at	TIMESTAMP WITH TIME ZONE


table name: account__documents
schema:
record_id	VARCHAR
remote_id	VARCHAR
document	VARCHAR
document_embedded	FLOAT[768]
chunk_id	INTEGER


table name: opportunity
id	VARCHAR
remote_id	VARCHAR
remote_created_at	TIMESTAMP WITH TIME ZONE
last_activity_at	TIMESTAMP WITH TIME ZONE
name	VARCHAR
description	VARCHAR
amount	INTEGER
owner	VARCHAR
account	VARCHAR
status	VARCHAR
close_date	TIMESTAMP WITH TIME ZONE
remote_was_deleted	BOOLEAN

table name: opportunity__documents
schema:
record_id	VARCHAR
remote_id	VARCHAR
document	VARCHAR
document_embedded	FLOAT[768]
chunk_id	INTEGER

addresses is a struct with the following fields: street_1, city, state, postal_code, country, address_type
country is a two character country code like US, CA, GB, etc.

Remember to check if structs are empty before accessing them. For example:
WHERE account.addresses IS NOT NULL AND account.addresses[1].state = 'Texas'

contact.account is a foreign key to account.remote_id

opportunity.account is a foreign key to account.remote_id
opportunity.status is either OPEN, LOST, or WON. These must be matched exactly and in uppercase.
opportunity.amount is in dollars of poretnial revenue
opportunity.close_date is the date the opportunity was closed to LOST or WON

Each contact can have many documents: contact__documents.record_id is a foreign key to contact.id
Each opportunity can have many documents: opportunity__documents.record_id is a foreign key to opportunity.id
Each account can have many documents: account__documents.record_id is a foreign key to account.id

You can use the embedding() function to embed a string into a vector.
For example: SELECT contact.*, contact__documents.document, ARRAY_COSINE_SIMILARITY(embedding('Pancakes are delicious'), document_embedded) AS score FROM contact JOIN contact__documents ON contact.id = contact__documents.record_id WHERE score > 0.01 ORDER BY score DESC LIMIT 10
"""
EMBEDDING_ARRAY_SIZE = 768
SCORE_COLUMN = "__score"
DOCUMENT_COLUMN_EMBEDDED = "document_embedded"


class QueryResponse(ToolResponseBase):
    query_result_rows: list[dict[str, Any]]


class QueryDatabaseTool(ToolBase[QueryResponse]):
    name = "query_database"
    description = "Query the duck db with an SQL query. It returns a list of row values based on the query. We need to not use too much memory, so limit queries to the smallesrt number to answer the question."
    prompt_description = PROMPT_DESCRIPTION

    query: str = Field(description="The SQL query to search for duckdb database.")

    async def _perform_action(self) -> QueryResponse:
        path = os.getenv("DUCKDB_PATH")
        if not path:
            path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "salesforce.duckdb")
            )
        query = self.query

        # match the first to embded and replace it with the vectorized query. handle single and double quotes, ignore case
        while match := re.search(
            r"embedding\((['\"])?([^)]+)\1\)(?![^\[]*\])", query, re.IGNORECASE
        ):
            vectorized_query = (await calculate_embeddings(match.group(2))).vectors
            query = query.replace(
                match.group(0),
                f"[{', '.join(map(str, vectorized_query))}]::FLOAT[{EMBEDDING_ARRAY_SIZE}]",
            )

        # connect and query the duckdb
        con = duckdb.connect(path)
        df = con.sql(query).df()
        json_result = df.to_json(orient="records")
        return QueryResponse(query_result_rows=json.loads(json_result))
