import pandas as pd
import logging
from typing import List, Optional, Dict, Any
import pandera as pa
from pandera.typing import DataFrame, Series
import tqdm
from pathlib import Path
from datetime import datetime
import json
from noteboard.github_stars import get_user_stars

# Define schema for GitHub star information
github_star_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, nullable=False),
        "full_name": pa.Column(str, nullable=False),
        "description": pa.Column(str, nullable=True),
        "html_url": pa.Column(str, nullable=False),
        "language": pa.Column(str, nullable=True),
        "stargazers_count": pa.Column(int, nullable=False, checks=pa.Check.ge(0)),
        "created_at": pa.Column(str, nullable=False),
        "updated_at": pa.Column(str, nullable=False),
    },
    strict=False,
)


def get_github_stars_df(username: str, github_token: Optional[str] = None) -> DataFrame[github_star_schema]:
    """
    Fetch GitHub stars for a user and return as a DataFrame.
    
    Args:
        username (str): GitHub username
        github_token (str, optional): GitHub personal access token for API authentication
    
    Returns:
        DataFrame: DataFrame with GitHub star information, validated against github_star_schema
    """
    logging.info(f"Fetching GitHub stars for user: {username}")
    
    # Use the existing get_user_stars function from noteboard.github_stars
    df = get_user_stars(username, github_token)
    
    if df.empty:
        logging.warning(f"No stars found for user: {username}")
        return pd.DataFrame()
    
    logging.info(f"Found {len(df)} starred repositories")
    
    # Validate against schema
    try:
        return github_star_schema.validate(df)
    except pa.errors.SchemaError as e:
        logging.error(f"Schema validation error: {e}")
        return df  # Return unvalidated DataFrame if validation fails


def save_github_stars(username: str, output_dir: str, github_token: Optional[str] = None) -> str:
    """
    Fetch GitHub stars for a user and save to a CSV file.
    
    Args:
        username (str): GitHub username
        output_dir (str): Directory to save the CSV file
        github_token (str, optional): GitHub personal access token for API authentication
    
    Returns:
        str: Path to the saved CSV file
    """
    df = get_github_stars_df(username, github_token)
    if df.empty:
        logging.warning(f"No data to save for user: {username}")
        return ""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{username}_github_stars_{timestamp}.csv"
    filepath = output_path / filename
    
    df.to_csv(filepath, index=False)
    logging.info(f"Saved GitHub stars to: {filepath}")
    
    return str(filepath)


def load_github_stars(filepath: str) -> DataFrame[github_star_schema]:
    """
    Load GitHub stars from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        DataFrame: DataFrame with GitHub star information
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        logging.error(f"Error loading GitHub stars from {filepath}: {e}")
        return pd.DataFrame()


def load_raw_github_stars_df(directory: str) -> pd.DataFrame:
    """
    Load CSV files from directory and combine them into a single DataFrame.
    
    Args:
        directory (str): Path to directory containing CSV files with GitHub stars data
        
    Returns:
        pd.DataFrame: Combined DataFrame with username from filenames
    """
    directory = Path(directory)
    dfs = []
    
    for csv_file in directory.glob("*_github_stars_*.csv"):
        try:
            df = pd.read_csv(csv_file)
            # Extract username from filename (format: username_github_stars_timestamp.csv)
            username = csv_file.stem.split("_github_stars_")[0]
            df["username"] = username
            dfs.append(df)
        except Exception as e:
            logging.warning(f"Error loading {csv_file}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Loaded {len(combined_df)} GitHub stars records for {combined_df['username'].nunique()} users")
    
    return combined_df
