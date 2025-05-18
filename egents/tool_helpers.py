import pathlib
import os
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, List

# from duckduckgo_search import DDGS

# URL regex pattern for link extraction
url_regex = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'\".,<>?«»""''])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""


class LinkValidator(BaseModel):
    url_regex: str = Field(default=url_regex)

    def _get_links(self, text):
        return re.findall(self.url_regex, text)

    def _get_link_statuses(self, links):
        return {url: requests.get(url).status_code for url in links}

    def check_links_statuses(self, text: str) -> Dict[str, int]:
        """
        Parse links from the input text and
        return the dictionary mapping the links to their HTTP statuses
        """
        return self._get_link_statuses(self._get_links(text))


link_validator = LinkValidator()


class GitHubCommitChecker:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

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
        soup = BeautifulSoup(response.text, "html.parser")

        # Get the default branch name
        default_branch = self._get_default_branch(soup, repo_url)

        # Look for the latest commit time
        commit_time = self._extract_commit_time(soup)

        if commit_time:
            return commit_time

        # If we couldn't find it on the main page, try the commits page
        commits_url = f"{repo_url}/commits/{default_branch}"
        response = self._fetch_page(commits_url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Try to find any datetime attribute in the page
        for elem in soup.find_all(attrs={"datetime": True}):
            datetime_str = elem.get("datetime")
            if datetime_str:
                try:
                    commit_time = datetime.fromisoformat(
                        datetime_str.replace("Z", "+00:00")
                    )
                    return commit_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                except ValueError:
                    continue

        return "Could not determine the latest commit time"

    def _clean_url(self, repo_url):
        """Clean and validate the repository URL."""
        if not repo_url:
            raise ValueError("Repository URL cannot be empty")

        if not repo_url.startswith("https://github.com/"):
            if repo_url.startswith("github.com/"):
                repo_url = "https://" + repo_url
            else:
                raise ValueError("Invalid GitHub repository URL")

        # Remove trailing slash if present
        return repo_url.rstrip("/")

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
        branch_element = soup.select_one("span.css-truncate-target[data-menu-button]")
        if branch_element:
            return branch_element.text.strip()

        # Try common branch names
        for branch in ["main", "master"]:
            try:
                response = self.session.head(f"{repo_url}/tree/{branch}")
                if response.status_code == 200:
                    return branch
            except:
                continue

        # Default to 'main' if we can't determine
        return "main"

    def _extract_commit_time(self, soup):
        """Extract the commit time from the soup object."""
        # Try different selectors for the time element
        for selector in ["relative-time", "time-ago", "time"]:
            time_elements = soup.find_all(selector)
            for time_element in time_elements:
                datetime_str = time_element.get("datetime")
                if datetime_str:
                    try:
                        commit_time = datetime.fromisoformat(
                            datetime_str.replace("Z", "+00:00")
                        )
                        return commit_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                    except ValueError:
                        continue

        return None


def extract_date(date_string):
    # Parse the datetime string
    dt_object = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S %Z")

    # Extract just the date part
    date_only = dt_object.date()

    return date_only


# Create a GitHub commit checker instance
github_commit_checker = GitHubCommitChecker()


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
            results.append(
                {
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
            )
        return results
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]
