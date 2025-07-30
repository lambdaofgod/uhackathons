
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pandera as pa
import yt_dlp
from pandera.typing import DataFrame
from pydantic import BaseModel, Field
from tqdm.contrib.concurrent import process_map

VideoDataSchema = pa.DataFrameSchema(
    {
        "video_id": pa.Column(str, nullable=False),
        "title": pa.Column(str, nullable=False),
        "description": pa.Column(str, nullable=False),
        "channel_id": pa.Column(str, nullable=False),
        "channel_title": pa.Column(str, nullable=False),
        "duration": pa.Column(float, nullable=False, checks=pa.Check.ge(0)),
        "view_count": pa.Column(int, nullable=False, checks=pa.Check.ge(0)),
        "like_count": pa.Column(int, nullable=False, checks=pa.Check.ge(0)),
        "comment_count": pa.Column(int, nullable=False, checks=pa.Check.ge(0)),
    },
    strict=False,
    coerce=True,
)


class AuthArgs(BaseModel):
    browser: Optional[str] = Field(
        default=None,
        description="The browser to use for cookie extraction (e.g., 'chrome', 'firefox').",
    )


def _fetch_video_details(video_id: str, browser: Optional[str]) -> Optional[dict]:
    """Helper function to fetch details for a single video."""
    detail_opts = {
        "quiet": True,
        "force_generic_extractor": True,
        "logger": logging.getLogger(__name__),
        "extract_flat": False,
        "skip_download": True,
        "ignore_no_formats_error": True,
    }
    if browser:
        detail_opts["cookiesfrombrowser"] = (browser,)

    try:
        with yt_dlp.YoutubeDL(detail_opts) as ydl:
            video_info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False
            )
            return {
                "video_id": video_info.get("id", ""),
                "title": video_info.get("title", "No Title"),
                "description": video_info.get("description", ""),
                "channel_id": video_info.get("channel_id", ""),
                "channel_title": video_info.get("channel", ""),
                "duration": video_info.get("duration", 0.0),
                "view_count": video_info.get("view_count", 0),
                "like_count": video_info.get("like_count", 0),
                "comment_count": video_info.get("comment_count", 0),
            }
    except yt_dlp.utils.DownloadError as e:
        logging.warning(f"Could not fetch details for video {video_id}. Error: {e}")
        return None


class YTPlaylist(BaseModel):
    playlist_id: str
    playlist_name: str
    auth_args: AuthArgs

    class Config:
        arbitrary_types_allowed = True

    def get_videos(self, n_pages: Optional[int] = None) -> DataFrame[VideoDataSchema]:
        # Stage 1: Reliably get all video IDs using --flat-playlist
        ydl_opts = {
            "quiet": True,
            "extract_flat": True,  # --flat-playlist is the key
            "force_generic_extractor": True,
            "logger": logging.getLogger(__name__),
        }
        if self.auth_args.browser:
            ydl_opts["cookiesfrombrowser"] = (self.auth_args.browser,)

        video_ids = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_url = (
                f"https://www.youtube.com/playlist?list={self.playlist_id}"
            )
            try:
                # The --flat-playlist equivalent gets all entries at once
                playlist_info = ydl.extract_info(playlist_url, download=False)
                if playlist_info and "entries" in playlist_info:
                    video_ids = [
                        entry["id"]
                        for entry in playlist_info["entries"]
                        if entry and "id" in entry
                    ]
            except yt_dlp.utils.DownloadError as e:
                logging.warning(
                    f"Could not fetch playlist {self.playlist_name} ({self.playlist_id}). Error: {e}"
                )
                return DataFrame(columns=[c for c in VideoDataSchema.columns])

        if n_pages is not None:
            # Limit the number of videos based on pages (100 videos per page)
            video_ids = video_ids[: n_pages * 100]

        if not video_ids:
            return DataFrame(columns=[c for c in VideoDataSchema.columns])

        # Stage 2: Fetch details in parallel
        results = process_map(
            _fetch_video_details,
            video_ids,
            [self.auth_args.browser] * len(video_ids),
            desc=f"Fetching details for {len(video_ids)} videos from {self.playlist_name}",
            chunksize=10,
        )

        videos = [res for res in results if res is not None]
        return DataFrame(videos)


def get_current_user_playlists(auth_args: AuthArgs) -> List[YTPlaylist]:
    """
    Fetches all playlists for the currently authenticated user using yt-dlp,
    including special playlists like "Liked videos" and "Watch Later".
    """
    # Add special playlists manually
    playlists = [
        YTPlaylist(playlist_id="LL", playlist_name="Liked Videos", auth_args=auth_args),
        YTPlaylist(playlist_id="WL", playlist_name="Watch Later", auth_args=auth_args),
    ]

    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "force_generic_extractor": True,
        "logger": logging.getLogger(__name__),
    }
    if auth_args.browser:
        ydl_opts["cookiesfrombrowser"] = (auth_args.browser,)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(
                "https://www.youtube.com/feed/playlists", download=False
            )
            if info and "entries" in info:
                for entry in info.get("entries", []):
                    if entry:
                        playlists.append(
                            YTPlaylist(
                                playlist_id=entry["id"],
                                playlist_name=entry["title"],
                                auth_args=auth_args,
                            )
                        )
        except yt_dlp.utils.DownloadError as e:
            logging.warning(
                f"Could not fetch user-created playlists. Maybe you are not logged in? Error: {e}"
            )

    return playlists
