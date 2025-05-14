"""
MCP Egent Tools

This module provides a FastMCP server with tools for link validation and GitHub commit checking,
similar to those in llamaindex_egents.py.
"""

from typing import Dict
import fire

from fastmcp import FastMCP

# Import the classes from tool_helpers
from tool_helpers import LinkValidator, GitHubCommitChecker, extract_date

# Create the MCP server
mcp = FastMCP(
    "Egent Tools",
    dependencies=["requests", "pydantic"],
)

# Initialize the helper classes
link_validator = LinkValidator()
github_commit_checker = GitHubCommitChecker()


@mcp.tool()
def check_link_statuses(text: str) -> Dict[str, int]:
    """
    Parse links from the input text and return the dictionary mapping the links to their HTTP statuses.

    Args:
        text: input string

    Returns:
        dict: dictionary mapping the links to their HTTP statuses
    """
    return link_validator.check_links_statuses(text)


@mcp.tool()
def get_repo_latest_commit_date(repo_url: str) -> str:
    """
    Extract github repository last commit date.

    Args:
        repo_url: string, the github repo url that starts with "https://github.com" or "github.com"

    Returns:
        str: date in format "Y-M-D"
    """
    commit_time = github_commit_checker.get_last_commit_time(repo_url)
    return extract_date(commit_time)


def main(port: int = 8000):
    """
    Run the MCP server on the specified port.

    Args:
        port: The port number to run the server on (default: 8000)
    """
    mcp.run(transport="sse", port=port)


# Run the server if this file is executed directly
if __name__ == "__main__":
    fire.Fire(main)
