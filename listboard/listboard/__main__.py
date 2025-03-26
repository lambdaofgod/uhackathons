import fire
from pathlib import Path
from .yt_data import load_raw_playlists_df, get_playlist_information_df
import logging


logging.basicConfig(level=logging.INFO)


def extract_yt_takeout_playlist_information(
    takeout_dir: str, output_path: str, include_watch_later=True
):
    """Extract information about YouTube videos from Google Takeout playlist CSVs.

    Args:
        takeout_dir (str): Directory containing playlist CSV files from Google Takeout
        output_path (str): Path where to save the output DataFrame with video information
    """
    # Convert paths to Path objects
    takeout_dir = Path(takeout_dir)
    output_path = Path(output_path)

    # Load all playlist CSVs
    playlists_df = load_raw_playlists_df(takeout_dir)

    if playlists_df.empty:
        print(f"No playlist CSVs found in {takeout_dir}")
        return

    # Fetch video information for all videos
    enriched_df = get_playlist_information_df(
        playlists_df, include_watch_later=include_watch_later
    )

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the result
    enriched_df.to_json(output_path, orient="records", lines=True)
    print(f"Saved enriched playlist data to {output_path}")


def main():
    fire.Fire({"extract": extract_yt_takeout_playlist_information})


if __name__ == "__main__":
    main()
