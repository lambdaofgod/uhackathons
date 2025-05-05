"""
MCP Egent Tools

This module provides a FastMCP server with tools for link validation and GitHub commit checking,
similar to those in llamaindex_egents.py.
"""

import re
import requests
from datetime import datetime
from typing import Dict

from fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP(
    "Egent Tools",
    dependencies=["requests"],
)

# URL regex pattern for link extraction
url_regex = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'


@mcp.tool()
def check_link_statuses(text: str) -> Dict[str, int]:
    """
    Parse links from the input text and return the dictionary mapping the links to their HTTP statuses.
    
    Args:
        text: input string
        
    Returns:
        dict: dictionary mapping the links to their HTTP statuses
    """
    # Initialize LinkValidator here instead of importing
    links = re.findall(url_regex, text)
    result = {}
    
    for link in links:
        try:
            response = requests.head(link, timeout=5, allow_redirects=True)
            result[link] = response.status_code
        except Exception:
            result[link] = -1
            
    return result


@mcp.tool()
def get_repo_latest_commit_date(repo_url: str) -> str:
    """
    Extract github repository last commit date.
    
    Args:
        repo_url: string, the github repo url that starts with "https://github.com" or "github.com"
        
    Returns:
        str: date in format "Y-M-D"
    """
    # Initialize GitHubCommitChecker here instead of importing
    # Normalize the URL
    if repo_url.startswith("github.com"):
        repo_url = "https://" + repo_url
    
    if not repo_url.startswith("https://github.com"):
        return "Error: Not a valid GitHub URL"
    
    # Set up session with headers
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    try:
        # Fetch the page
        response = session.get(repo_url)
        if response.status_code != 200:
            return f"Error: Failed to fetch page (status code: {response.status_code})"
        
        # Look for the time tag with relative-time class
        time_match = re.search(r'datetime="([^"]+)"', response.text)
        if time_match:
            date_str = time_match.group(1)
            # Parse the ISO format date
            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return date_obj.strftime('%Y-%m-%d')
        else:
            return "Error: Could not find commit date information"
    except Exception as e:
        return f"Error: {str(e)}"


# Run the server if this file is executed directly
if __name__ == "__main__":
    mcp.run()
