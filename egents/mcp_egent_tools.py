"""
MCP Egent Tools

This module provides a FastMCP server with tools for link validation and GitHub commit checking,
similar to those in llamaindex_egents.py.
"""

from fastmcp import FastMCP

# Import from our modules - reusing existing implementations
from tool_helpers import link_validator, github_commit_checker, extract_date

# Create the MCP server
mcp = FastMCP(
    "Egent Tools",
    dependencies=["requests"],
)


@mcp.tool()
def check_link_statuses(text: str) -> dict:
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
    return extract_date(github_commit_checker.get_last_commit_time(repo_url))


# Run the server if this file is executed directly
if __name__ == "__main__":
    mcp.run()
