import pandas as pd
from pathlib import Path
from typing import List
import yt_dlp
import pandera as pa
from pandera.typing import DataFrame, Series
import tqdm
import tqdm.contrib.concurrent

def load_raw_playlists_df(directory):
    """Load CSV files from directory and combine them into a single DataFrame with playlist names.
    
    Args:
        directory (str): Path to directory containing CSV files
        
    Returns:
        pd.DataFrame: Combined DataFrame with playlist names from filenames
    """
    directory = Path(directory)
    dfs = []
    
    for csv_file in directory.glob('*.csv'):
        df = pd.read_csv(csv_file)
        df['Playlist'] = csv_file.stem  # Use filename without extension as playlist name
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)

# Define schema for video information columns
video_data_schema = pa.DataFrameSchema({
    'video_id': pa.Column(str, nullable=False),
    'title': pa.Column(str, nullable=False),
    'description': pa.Column(str, nullable=False),
    'channel_id': pa.Column(str, nullable=False),
    'channel_title': pa.Column(str, nullable=False),
    'duration': pa.Column(float, nullable=False, checks=pa.Check.ge(0)),
    'view_count': pa.Column(int, nullable=False, checks=pa.Check.ge(0)),
    'like_count': pa.Column(int, nullable=False, checks=pa.Check.ge(0)),
    'comment_count': pa.Column(int, nullable=False, checks=pa.Check.ge(0))
}, strict=False)  # Allow additional columns from the input DataFrame

def get_playlist_information_df(raw_playlist_df: pd.DataFrame, id_col: str = "Video ID") -> DataFrame[video_data_schema]:
    """Fetch video information from YouTube videos using yt-dlp.
    
    Args:
        raw_playlist_df (pd.DataFrame): DataFrame containing video IDs
        id_col (str): Name of the column containing YouTube video IDs
        
    Returns:
        pd.DataFrame: DataFrame with video information, validated against VideoDataSchema
    """

    
    
    video_ids = raw_playlist_df[id_col].unique().tolist()
    all_video_info = []
    
    # Process videos one by one (yt-dlp handles rate limiting internally)
    all_video_info = [tqdm.contrib.concurrent.process_map(fetch_video_info, video_ids)]
    all_video_info = [info for info in all_video_info if info is not None]

    if not all_video_info:
        return pd.DataFrame()
    
    # Extract relevant information
    video_data = [{
        'video_id': video['id'],
        'title': video['title'],
        'description': video.get('description', ''),
        'channel_id': video['channel_id'],
        'channel_title': video['channel'],
        'duration': video.get('duration', 0),
        'view_count': video.get('view_count', 0),
        'like_count': video.get('like_count', 0),
        'comment_count': video.get('comment_count', 0)
    } for video in all_video_info]
    
    video_df = pd.DataFrame(video_data)
    
    # Merge with original DataFrame to preserve playlist information
    result_df = pd.merge(
        raw_playlist_df,
        video_df,
        left_on=id_col,
        right_on='video_id',
        how='left'
    )
    
    return result_df

def fetch_video_info(video_id: str) -> dict:
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
    }
    """Fetch information for a single video ID"""
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            url = f'https://www.youtube.com/watch?v={video_id}'
            info = ydl.extract_info(url, download=False)
            return info
    except Exception as e:
        print(f"Error fetching video info for {video_id}: {e}")
        return None
