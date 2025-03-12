import requests
import pandas as pd


def get_user_stars(username, github_token=None):
    """
    Retrieves information about a GitHub user's starred repositories and returns it as a Pandas DataFrame.

    Args:
        username (str): The GitHub username for whom to fetch starred repositories.
        github_token (str, optional): A personal GitHub access token. If provided, it will be used for authentication,
                                      which can increase rate limits and potentially access private starred repos
                                      (if the token has the necessary permissions). Defaults to None (unauthenticated requests).

    Returns:
        pandas.DataFrame: A DataFrame containing information about the user's starred repositories.
                          Returns an empty DataFrame if the user has no starred repositories or if there's an error.
                          The DataFrame columns include:
                            - 'name': Repository name
                            - 'full_name': Full repository name (owner/repo_name)
                            - 'description': Repository description
                            - 'html_url': Repository URL
                            - 'language': Programming language of the repository
                            - 'stargazers_count': Number of stars the repository has
                            - 'created_at': Repository creation timestamp
                            - 'updated_at': Repository update timestamp

    Raises:
        requests.exceptions.RequestException: If there's an issue with the API request (e.g., network error).
        ValueError: If the username is not provided or is not a string.

    Example:
        >>> df = get_user_stars('octocat')
        >>> print(df.head())
    """

    if not username or not isinstance(username, str):
        raise ValueError("Username must be a non-empty string.")

    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    base_url = f"https://api.github.com/users/{username}/starred"
    starred_repos = []
    page = 1

    try:
        while True:
            url = f"{base_url}?page={page}&per_page=100"  # Fetch 100 repos per page (max allowed)
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            repos_data = response.json()
            if (
                not repos_data
            ):  # No more starred repos on this page or user doesn't exist (empty response)
                break

            for repo in repos_data:
                starred_repos.append(
                    {
                        "name": repo.get("name"),
                        "full_name": repo.get("full_name"),
                        "description": repo.get("description"),
                        "html_url": repo.get("html_url"),
                        "language": repo.get("language"),
                        "stargazers_count": repo.get("stargazers_count"),
                        "created_at": repo.get("created_at"),
                        "updated_at": repo.get("updated_at"),
                    }
                )

            page += 1

        if not starred_repos:
            return pd.DataFrame()  # Return empty DataFrame if no starred repos found

        df = pd.DataFrame(starred_repos)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching starred repositories for user '{username}': {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of API error
    except ValueError as ve:
        raise ve  # Re-raise the ValueError from input validation
