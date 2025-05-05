import pathlib
import os

api_key_path = "~/.keys/anthropic_key.txt"

with open(pathlib.Path(api_key_path).expanduser()) as f:
    api_key = f.read().strip()
    os.environ["ANTHROPIC_API_KEY"] = api_key

with open(pathlib.Path("~/.keys/gemini_api_key.txt").expanduser()) as f:
    api_key = f.read().strip()
    os.environ["GEMINI_API_KEY"] = api_key

with open(pathlib.Path("~/.keys/brave.txt").expanduser()) as f:
    api_key = f.read().strip()
    os.environ["BRAVE_API_KEY"] = api_key



anthropic_model_name = "anthropic/claude-3-7-sonnet-20250219"
gemini_model_name = "gemini/gemini-2.0-flash-thinking-exp"

import requests
import re
from pydantic import BaseModel, Field
from typing import Dict

url_regex = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""

class LinkValidator(BaseModel):
    url_regex: str = Field(default=url_regex)

    def _get_links(self, text):
        return re.findall(self.url_regex, text)

    def _get_link_statuses(self, links):
        return {
            url: requests.get(url).status_code
            for url in links
        }

    def check_links_statuses(self, text: str) -> Dict[str, int]:
        """
        Parse links from the input text and
        return the dictionary mapping the links to their HTTP statuses
        """
        return self._get_link_statuses(self._get_links(text))


link_validator = LinkValidator()

import requests
from bs4 import BeautifulSoup
from datetime import datetime
from smolagents import tool


class GitHubCommitChecker:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def get_last_commit_time(self, repo_url):
        """
        Get the time of the latest commit on the default branch (main/master) of a GitHub repository.

        Args:
            repo_url (str): URL of the GitHub repository

        Returns:
            str: Timestamp of the latest commit
        """
        # Validate and clean the URL
        repo_url = self._clean_url(repo_url)

        # Try to fetch the repository page
        response = self._fetch_page(repo_url)

        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get the default branch name
        default_branch = self._get_default_branch(soup, repo_url)

        # Look for the latest commit time
        commit_time = self._extract_commit_time(soup)

        if commit_time:
            return commit_time

        # If we couldn't find it on the main page, try the commits page
        commits_url = f"{repo_url}/commits/{default_branch}"
        response = self._fetch_page(commits_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find any datetime attribute in the page
        for elem in soup.find_all(attrs={"datetime": True}):
            datetime_str = elem.get('datetime')
            if datetime_str:
                try:
                    commit_time = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                    return commit_time.strftime('%Y-%m-%d %H:%M:%S UTC')
                except ValueError:
                    continue

        return "Could not determine the latest commit time"

    def _clean_url(self, repo_url):
        """Clean and validate the repository URL."""
        if not repo_url:
            raise ValueError("Repository URL cannot be empty")

        if not repo_url.startswith('https://github.com/'):
            if repo_url.startswith('github.com/'):
                repo_url = 'https://' + repo_url
            else:
                raise ValueError("Invalid GitHub repository URL")

        # Remove trailing slash if present
        return repo_url.rstrip('/')

    def _fetch_page(self, url):
        """Fetch a web page and handle errors."""
        try:
            response = self.session.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch {url}: {str(e)}")

    def _get_default_branch(self, soup, repo_url):
        """Try to determine the default branch name."""
        # First check if we can find it in the page
        branch_element = soup.select_one('span.css-truncate-target[data-menu-button]')
        if branch_element:
            return branch_element.text.strip()

        # Try common branch names
        for branch in ['main', 'master']:
            try:
                response = self.session.head(f"{repo_url}/tree/{branch}")
                if response.status_code == 200:
                    return branch
            except:
                continue

        # Default to 'main' if we can't determine
        return 'main'

    def _extract_commit_time(self, soup):
        """Extract the commit time from the soup object."""
        # Try different selectors for the time element
        for selector in ['relative-time', 'time-ago', 'time']:
            time_elements = soup.find_all(selector)
            for time_element in time_elements:
                datetime_str = time_element.get('datetime')
                if datetime_str:
                    try:
                        commit_time = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                        return commit_time.strftime('%Y-%m-%d %H:%M:%S UTC')
                    except ValueError:
                        continue

        return None

def extract_date(date_string):
    # Parse the datetime string
    dt_object = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S %Z")

    # Extract just the date part
    date_only = dt_object.date()

    return date_only

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

from phoenix.otel import register

tracer_provider = register(endpoint="http://127.0.0.1:6006/v1/traces", project_name="egents")
LlamaIndexInstrumentor().instrument(skip_dep_check=True, tracer_provider=tracer_provider)

from typing import List, Dict, Any, Optional
from llama_index.core.tools import FunctionTool
import asyncio



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
    return app.scrape_url(url)

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

# Create a simple DuckDuckGo search tool for LlamaIndex
from duckduckgo_search import DDGS

def duckduckgo_search_fn(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using DuckDuckGo
    Args:
        query: The search query
        max_results: Maximum number of results to return
    Returns:
        List of search results with title, link, and snippet
    """
    try:
        ddgs = DDGS()
        results = []
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "link": r.get("href", ""),
                "snippet": r.get("body", "")
            })
        return results
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]

duckduckgo_tools = [
    FunctionTool.from_defaults(
        name="duckduckgo_search",
        fn=duckduckgo_search_fn,
        description="Search the web using DuckDuckGo. Input should be a search query."
    )
]

from llama_index.tools.exa import ExaToolSpec
import os
from llama_index.tools.brave_search import BraveSearchToolSpec

brave_tool_spec = BraveSearchToolSpec(api_key=os.environ["BRAVE_API_KEY"])

search_tools = brave_tool_spec.to_tool_list()

import os
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.types import Task
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.workflow import Context

anthropic_model_name = "claude-3-7-sonnet-20250219"
gemini_model_name = "gemini-2.5-pro-exp-03-25"


# Initialize LLM models
anthropic_llm = Anthropic(
    model=anthropic_model_name,#("/")[1],
    api_key=os.environ["ANTHROPIC_API_KEY"],
    temperature=0.2,
    max_tokens=2048
)

gemini_llm = GoogleGenAI(
    model_name=gemini_model_name,#("/")[1],
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
# agent_workflow = AgentWorkflow(
#     nodes={
#         "scraping_agent": scraping_llamaindex_agent,
#         "web_agent": web_llamaindex_agent,
#         "main_agent": main_llamaindex_agent,
#         "thinking_agent": thinking_llamaindex_agent,
#         "coding_agent": coding_llamaindex_agent
#     },
#     edges=[
#         ("main_agent", "web_agent"),
#         ("web_agent", "scraping_agent")
#     ]
# )

assistant_workflow = AgentWorkflow(agents=[assistant_agent, web_llamaindex_agent, scraping_llamaindex_agent], root_agent="Assistant")

ctx = Context(assistant_workflow)

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
    # if mode == "thinking":
    #     agent = thinking_llamaindex_agent
    # elif mode == "web":
    #     agent = web_llamaindex_agent
    # elif mode == "coding":
    #     agent = coding_llamaindex_agent
    # else:
    #     agent = main_llamaindex_agent

    # Use the simpler chat interface instead of Task
    response = asyncio.run(run_task(prompt, mode=mode, ctx=ctx))

    # Print the response in real-time (simulating streaming)
    print(response.response)

    return response.response

prompt = 'What is the policy gradient theorem? Write down the theorem formally, use LaTeX'
get_agent_output(prompt)

prompt = """
Suggest 5 introductory papers from 2025 in the field of mechanistic interpretability
""".strip()
agent_response = get_agent_output(prompt)
agent_response

prompt = """
How can I use Arize Phoenix with LlamaIndex?
""".strip()
agent_response = get_agent_output(prompt)
agent_response

prompt = """
How can I use tools with smolagents LLM agent library?
""".strip()
agent_response = get_agent_output(prompt, mode="thinking")
agent_response
