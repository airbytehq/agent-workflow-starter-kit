# Agent Workflow Starter Kit

Broadly speaking, the future of getting things done with our new magical AI APIs are “Agents.” The definition varies, but I think the folks at Anthropic did a great job on the blog post [Building effective agents](https://www.anthropic.com/research/building-effective-agents).

I wanted to try out two appraoches.

* Workflow: a flowchart of decisions and tool calls to answer a prompt from the user. It commonly will have steps to determine which tools to use and evaluate the results.
* Agent: an LLM (like OpenAI) is given a prompt from the user and a set of tool definitions and decides how to use them to answer the prompt.

In this app, I made it easy to try out both to talk to Salesforce data (fake data checked into a duckdb database in this repo).

## Quickstart

To get started with this example, follow these steps:

1. Ensure you have Python and Poetry installed on your system.
2. Clone the repository and navigate to this directory.
3. `poetry env use 3.10.15`
3. Install the required dependencies by running `poetry install`.
4. Set your OpenAI API key as an environment variable: `cp .env.example .env` and add your key to `.env`.
5. Download the embedding model: `poetry run download_embedding_model`
6. Run the application: `./dev.sh`.
7. Open http://localhost:8080/chat


### Running the test suite

`poetry run pytest -s app/tests`

### More

Read more about this on the blog post [here](TODO)