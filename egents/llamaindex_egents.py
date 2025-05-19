import os
import asyncio
from firecrawl.firecrawl import FirecrawlApp
from typing import List, Dict, Any, Optional
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent, FunctionAgent
from llama_index.core.workflow import Context
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.google_genai import GoogleGenAI
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, ToolCall
import json
import pathlib

# Import from our modules
from tool_helpers import (
    LinkValidator,
    GitHubCommitChecker,
    extract_date,
)

github_commit_checker = GitHubCommitChecker()
link_validator = LinkValidator()
with open(pathlib.Path("~/.keys/firecrawl_key.txt").expanduser()) as f:
    fc_api_key = f.read().strip()
    os.environ["FIRECRAWL_API_KEY"] = fc_api_key


# Set up tracing
tracer_provider = register(
    endpoint="http://127.0.0.1:6006/v1/traces", project_name="egents"
)
LlamaIndexInstrumentor().instrument(
    skip_dep_check=True, tracer_provider=tracer_provider
)


# Convert existing tools to LlamaIndex FunctionTool format
def check_link_statuses_llamaindex(text: str) -> Dict[str, int]:
    """
    Parse links from the input text and
    return the dictionary mapping the links to their HTTP statuses
    Args:
        text: input string
    Returns:
        dict: dictionary mapping the links to their HTTP statuses
    """
    return link_validator.check_links_statuses(text)


def get_repo_latest_commit_date_llamaindex(repo_url: str) -> str:
    """
    Extract github repository last commit date
    Args:
        repo_url: string, the github repo url that starts with "https://github.com" or "github.com"
    Returns:
        str: date in format "Y-M-D"
    """
    return extract_date(github_commit_checker.get_last_commit_time(repo_url))


def firecrawl_tool_llamaindex(url: str) -> dict:
    """
    Scrape given URL with firecrawl
    Args:
       url: link string
    Returns:
       dict: dictionary with markdown and metadata keys
    """
    fc_app = FirecrawlApp(api_key=fc_api_key)
    return fc_app.scrape_url(url)


def setup_searx_mcp_tools(searx_url="http://localhost:8080"):
    mcp_client = BasicMCPClient(
        "uvx", args=["mcp-searxng"], env={"SEARXNG_URL": searx_url}
    )
    return McpToolSpec(client=mcp_client).to_tool_list()


# Create LlamaIndex tools
link_status_tool = FunctionTool.from_defaults(
    name="check_link_statuses",
    fn=check_link_statuses_llamaindex,
    description="Parse links from the input text and return the dictionary mapping the links to their HTTP statuses",
)

github_commit_tool = FunctionTool.from_defaults(
    name="get_repo_latest_commit_date",
    fn=get_repo_latest_commit_date_llamaindex,
    description="Extract github repository last commit date",
)

firecrawl_scrape_tool = FunctionTool.from_defaults(
    name="firecrawl_scrape",
    fn=firecrawl_tool_llamaindex,
    description="Scrape given URL with firecrawl",
)

search_tools = setup_searx_mcp_tools()

anthropic_model_name = "claude-3-7-sonnet-20250219"
gemini_model_name = "gemini-2.5-pro-exp-03-25"
# Initialize LLM models
anthropic_llm = Anthropic(
    model=anthropic_model_name,
    api_key=os.environ["ANTHROPIC_API_KEY"],
    temperature=0.2,
    max_tokens=2048,
)

gemini_llm = GoogleGenAI(
    model_name=gemini_model_name,
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.2,
    max_tokens=4096,
)

# Create agents
scraping_llamaindex_agent = FunctionAgent(
    tools=[firecrawl_scrape_tool],
    llm=anthropic_llm,
    verbose=True,
    name="ScrapingAgent",
    description="Webpage scraping agent. Uses the tool to scrape the webpages given list of URLs in order to answer questions",
    system_mrpomt="Use the tool to scrape the webpages given list of URLs in order to answer questions",
)

web_llamaindex_agent = FunctionAgent(
    tools=search_tools,
    llm=anthropic_llm,
    verbose=True,
    name="WebAgent",
    description="Agent capable of searching the web. Returns the information useful for answering a given queries. It tries to minimize the number of calls to search tool whenever it gets links in input. Uses `ScrapingAgent` to find information from specific webpages.",
    system_prompt="Agent capable of searching webpages. Returns the information useful for answering a given queries. It tries to minimize the number of calls to search tool whenever it gets links in input. Uses `ScrapingAgent` to find information from specific webpages. The answers should cite the relevant sources in markdown format (the citation should be a number of the source like [1], and the links should be stored at the end of the response)",
    max_iterations=3,
)

assistant_agent = ReActAgent(
    tools=[link_status_tool, github_commit_tool],
    llm=anthropic_llm,
    verbose=True,
    name="Assistant",
    description="Generalist agent that uses helper WebAgent for searching the web. WebAgent is used carefully because of the web search rate limits - if relevant links were collected, the WebAgent should receive them",
    max_iterations=5,
)

thinking_llamaindex_agent = ReActAgent(
    tools=[link_status_tool, github_commit_tool] + search_tools,
    llm=gemini_llm,
    verbose=True,
    name="ThinkingAgent",
    description="Agent that thinks deeply about problems and uses search tools when needed",
)

coding_llamaindex_agent = ReActAgent(
    tools=[],
    llm=gemini_llm,
    verbose=True,
    name="CodingAgent",
    description="Agent specialized in writing and analyzing code",
)

# Create agent workflow
assistant_workflow = AgentWorkflow(
    agents=[assistant_agent, web_llamaindex_agent, scraping_llamaindex_agent],
    root_agent="Assistant",
)


# Function to get agent output using LlamaIndex
async def run_task(prompt, mode="default", ctx=None, verbose=True):
    if mode == "thinking":
        workflow = thinking_llamaindex_agent
    else:
        workflow = assistant_workflow
    handler = workflow.run(prompt, ctx=ctx)
    async for event in handler.stream_events():
        if verbose and type(event) == ToolCall:
            print(f"Calling tool {event.tool_name}")

            print(json.dumps(event.tool_kwargs))
        elif verbose and type(event) == ToolCallResult:
            print(f"Tool {event.tool_name} returned")
            print(event.tool_output.content)

    response = await handler
    return response


def get_agent_output(prompt, mode="default", ctx=None):
    """
    Pass Context from llamaindex to maintain memory
    """
    response = asyncio.run(run_task(prompt, mode=mode, ctx=ctx))

    # Print the response in real-time (simulating streaming)

    return response.response.content
