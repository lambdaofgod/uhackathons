import os
import asyncio
from typing import List, Dict, Any, Optional
from llama_index.core.tools import FunctionTool
from llama_index.tools.brave_search import BraveSearchToolSpec
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent, FunctionAgent
from llama_index.core.workflow import Context
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.google_genai import GoogleGenAI
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

# Import from our modules
from tool_helpers import link_validator, github_commit_checker, extract_date, duckduckgo_search_fn
from llms import anthropic_model_name, gemini_model_name

# Set up tracing
tracer_provider = register(endpoint="http://127.0.0.1:6006/v1/traces", project_name="egents")
LlamaIndexInstrumentor().instrument(skip_dep_check=True, tracer_provider=tracer_provider)

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
    # Use the MCP server for scraping
    import requests
    response = requests.post(
        "http://localhost:8000/scrape",
        json={"url": url}
    )
    if response.status_code == 200:
        return response.json()
    else:
        return {"markdown": f"Error scraping URL: {response.status_code}", "metadata": {}}

# Create LlamaIndex tools
link_status_tool = FunctionTool.from_defaults(
    name="check_link_statuses",
    fn=check_link_statuses_llamaindex,
    description="Parse links from the input text and return the dictionary mapping the links to their HTTP statuses"
)

github_commit_tool = FunctionTool.from_defaults(
    name="get_repo_latest_commit_date",
    fn=get_repo_latest_commit_date_llamaindex,
    description="Extract github repository last commit date"
)

firecrawl_scrape_tool = FunctionTool.from_defaults(
    name="firecrawl_scrape",
    fn=firecrawl_tool_llamaindex,
    description="Scrape given URL with firecrawl"
)

# Create DuckDuckGo search tool for LlamaIndex
duckduckgo_tools = [
    FunctionTool.from_defaults(
        name="duckduckgo_search",
        fn=duckduckgo_search_fn,
        description="Search the web using DuckDuckGo. Input should be a search query."
    )
]

# Set up Brave search tools
brave_tool_spec = BraveSearchToolSpec(api_key=os.environ["BRAVE_API_KEY"])
search_tools = brave_tool_spec.to_tool_list()

# Initialize LLM models
anthropic_llm = Anthropic(
    model=anthropic_model_name,
    api_key=os.environ["ANTHROPIC_API_KEY"],
    temperature=0.2,
    max_tokens=2048
)

gemini_llm = GoogleGenAI(
    model_name=gemini_model_name,
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.2,
    max_tokens=4096
)

# Create agents
scraping_llamaindex_agent = FunctionAgent(
    tools=[firecrawl_scrape_tool],
    llm=anthropic_llm,
    verbose=True,
    name="ScrapingAgent",
    description="Webpage scraping agent. Uses the tool to scrape the webpages given list of URLs in order to answer questions",
    system_mrpomt="Use the tool to scrape the webpages given list of URLs in order to answer questions"
)

web_llamaindex_agent = FunctionAgent(
    tools=search_tools,
    llm=anthropic_llm,
    verbose=True,
    name="WebAgent",
    description="Agent capable of searching webpages. Returns the information useful for answering a given queries. It tries to minimize the number of calls to search tool whenever it gets links in input. Uses `ScrapingAgent` to find information from specific webpages. For searching repositories on github scrape `https://github.com/search?q=<SEARCH QUERY>&type=repositories` instead of using web search",
    system_prompt="Agent capable of searching webpages. Returns the information useful for answering a given queries. It tries to minimize the number of calls to search tool whenever it gets links in input. Uses `ScrapingAgent` to find information from specific webpages. For searching repositories on github scrape `https://github.com/search?q=<SEARCH QUERY>&type=repositories` instead of using web search",
    max_iterations=15
)

assistant_agent = ReActAgent(
    tools=[link_status_tool, github_commit_tool],
    llm=anthropic_llm,
    verbose=True,
    name="Assistant",
    description="Generalist agent that uses helper WebAgent for searching the web. WebAgent is used carefully because of the web search rate limits - if relevant links were collected, the WebAgent should receive them",
    max_iterations=15
)

thinking_llamaindex_agent = ReActAgent(
    tools=[link_status_tool, github_commit_tool] + search_tools,
    llm=gemini_llm,
    verbose=True,
    name="ThinkingAgent",
    description="Agent that thinks deeply about problems and uses search tools when needed"
)

coding_llamaindex_agent = ReActAgent(
    tools=[],
    llm=gemini_llm,
    verbose=True,
    name="CodingAgent",
    description="Agent specialized in writing and analyzing code"
)

# Create agent workflow
assistant_workflow = AgentWorkflow(agents=[assistant_agent, web_llamaindex_agent, scraping_llamaindex_agent], root_agent="Assistant")

# Function to get agent output using LlamaIndex
async def run_task(prompt, mode="default", ctx=None):
    if mode == "thinking":
        workflow = thinking_llamaindex_agent
    else:
        workflow = assistant_workflow
    return await workflow.run(user_msg=prompt, ctx=ctx)

def get_agent_output(prompt, mode="default", ctx=None):
    """
    Pass Context from llamaindex to maintain memory
    """
    response = asyncio.run(run_task(prompt, mode=mode, ctx=ctx))

    # Print the response in real-time (simulating streaming)
    print(response.response)

    return response.response
