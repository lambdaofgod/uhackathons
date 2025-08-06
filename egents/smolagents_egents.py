from smolagents import LiteLLMModel
from litellm import completion
import pathlib
import os
from firecrawl.firecrawl import FirecrawlApp
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    GoogleSearchTool,
)
from smolagents import LiteLLMModel, MCPClient
from smolagents import tool
from tool_helpers import LinkValidator, GitHubCommitChecker, extract_date
from typing import List, Dict, Any, Optional
from mcp import StdioServerParameters


additional_authorized_imports = ["requests", "bs4"]


class EgentConfig:
    do_visit_webpages = False


def setup_searx_mcp_tools(searx_url="http://localhost:8080"):
    server_parameters = StdioServerParameters(
        command="uvx", args=["mcp-searxng"], env={"SEARXNG_URL": searx_url}
    )
    mcp_client = MCPClient(server_parameters)
    return mcp_client.get_tools()


github_commit_checker = GitHubCommitChecker()


@tool
def get_repo_latest_commit_date(repo_url: str) -> str:
    """
    Extract github repository last commit date

    Args:
        repo_url: string, the github repo url that starts with "https://github.com" or "github.com"
    Returns:
        str: date in format "Y-M-D"
    """
    return extract_date(github_commit_checker.get_last_commit_time(repo_url))


link_validator = LinkValidator()


@tool
def check_link_statuses_tool(text: str) -> Dict[str, int]:
    """
    Parse links from the input text and
    return the dictionary mapping the links to their HTTP statuses
    Args:
        text: input string
    Returns:
        dict: dictionary mapping the links to their HTTP statuses
    """
    return link_validator.check_links_statuses(text)


@tool
def firecrawl_tool(url: str) -> dict:
    """
    Scrape given URL with firecrawl
    Args:
       url: link string
    Returns:
       dict: dictionary with markdown and metadata keys
    """
    fc_app = FirecrawlApp(api_key=fc_api_key)
    return fc_app.scrape_url(url)


with open(pathlib.Path("~/.keys/firecrawl_key.txt").expanduser()) as f:
    fc_api_key = f.read().strip()
    os.environ["FIRECRAWL_API_KEY"] = fc_api_key


anthropic_model_name = "claude-sonnet-4-20250514"
gemini_model_name = "gemini-2.5-pro-exp-03-25"

model = LiteLLMModel(
    anthropic_model_name, temperature=0.2, max_tokens=2048
)  # use_caching=True)

tools = [check_link_statuses_tool, get_repo_latest_commit_date]

thinking_model = LiteLLMModel(gemini_model_name, temperature=0.2, max_tokens=4096)
thinking_agent = CodeAgent(
    tools=tools + setup_searx_mcp_tools(),
    model=thinking_model,
    additional_authorized_imports=additional_authorized_imports,
)


scraping_agent = ToolCallingAgent(
    name="ScrapingAgent",
    description="Webpage scraping agent. Uses the tool to scrape the webpages given list of URLs in order to answer questions",
    tools=[firecrawl_tool],
    model=model,
)

web_agent = ToolCallingAgent(
    name="WebAgent",
    description="Agent capable of searching webpages. Returns the information useful for answering a given queries. It tries to minimize the number of calls to search tool whenever it gets links in input. Uses `ScrapingAgent` to find information from specific webpages. For searching repositories on github scrape `https://github.com/search?q=<SEARCH QUERY>&type=repositories` instead of using web search",
    tools=setup_searx_mcp_tools(),
    model=model,
    managed_agents=[scraping_agent],
)
agent = CodeAgent(
    tools=tools,
    model=model,
    description="Generalist agent that uses helper WebAgent for searching the web. WebAgent is used carefully because of the web search rate limits - if relevant links were collected, the WebAgent should receive them",
    managed_agents=[web_agent],
    additional_authorized_imports=additional_authorized_imports,
    max_steps=3,
)
coding_agent = CodeAgent(
    tools=[], model=thinking_model, additional_authorized_imports=["requests"]
)


def get_agent_output(prompt, mode="default"):
    if mode == "thinking":
        _agent = thinking_agent
    elif mode == "web":
        _agent = web_agent
    elif mode == "coding":
        _agent = coding_agent
    else:
        _agent = agent
    parts = []
    for p in _agent.run(prompt, stream=True):
        parts.append(p)
        print(p)
    return p.output
