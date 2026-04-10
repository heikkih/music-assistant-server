"""Helpers for the NRK PSAPI provider."""

from __future__ import annotations

import re
from typing import Any

from music_assistant_models.enums import ContentType, ImageType, StreamType
from music_assistant_models.errors import UnplayableMediaError
from music_assistant_models.media_items import (
    MediaItemImage,
    Podcast,
    PodcastEpisode,
    ProviderMapping,
    Radio,
)

RADIO_REPLAY_PREFIX = "replay"
PROGRAM_EPISODE_PREFIX = "program"
PODCAST_EPISODE_PREFIX = "episode"


class NrkItemId:
    """Helper for stable provider item ids."""

    @staticmethod
    def replay_podcast(channel_id: str) -> str:
        """Build the provider id for a replay podcast."""
        return f"{RADIO_REPLAY_PREFIX}:{channel_id}"

    @staticmethod
    def program_episode(channel_id: str, program_id: str) -> str:
        """Build the provider id for a replay/program episode."""
        return f"{PROGRAM_EPISODE_PREFIX}:{channel_id}:{program_id}"

    @staticmethod
    def podcast_episode(podcast_id: str, episode_id: str) -> str:
        """Build the provider id for a podcast episode."""
        return f"{PODCAST_EPISODE_PREFIX}:{podcast_id}:{episode_id}"

    @staticmethod
    def parse_program_episode(item_id: str) -> tuple[str, str] | None:
        """Parse a replay/program episode id."""
        if not item_id.startswith(f"{PROGRAM_EPISODE_PREFIX}:"):
            return None
        _, channel_id, program_id = item_id.split(":", 2)
        return channel_id, program_id

    @staticmethod
    def parse_podcast_episode(item_id: str) -> tuple[str, str] | None:
        """Parse a podcast episode id."""
        if not item_id.startswith(f"{PODCAST_EPISODE_PREFIX}:"):
            return None
        _, podcast_id, episode_id = item_id.split(":", 2)
        return podcast_id, episode_id

    @staticmethod
    def parse_replay_podcast(item_id: str) -> str | None:
        """Parse a replay podcast id."""
        if not item_id.startswith(f"{RADIO_REPLAY_PREFIX}:"):
            return None
        return item_id.split(":", 1)[1]


def provider_mapping(domain: str, instance_id: str, item_id: str) -> ProviderMapping:
    """Build a provider mapping."""
    return ProviderMapping(
        item_id=item_id,
        provider_domain=domain,
        provider_instance=instance_id,
    )


def best_image_url(images: Any) -> str | None:
    """Return the best image URL from an NRK image object or list."""
    image_list: list[Any] = []
    if images is None:
        return None
    if hasattr(images, "web_images"):
        image_list = list(images.web_images or [])
    elif isinstance(images, list):
        image_list = images
    if not image_list:
        return None
    best = sorted(image_list, key=lambda img: int(getattr(img, "width", 0) or 0), reverse=True)
    return getattr(best[0], "url", None)


def add_image(item: Radio | Podcast | PodcastEpisode, images: Any, domain: str) -> None:
    """Attach the best available image to a Music Assistant item."""
    image_url = best_image_url(images)
    if not image_url:
        return
    item.metadata.add_image(
        MediaItemImage(
            type=ImageType.THUMB,
            path=image_url,
            provider=domain,
            remotely_accessible=True,
        )
    )


def parse_podcast_id_from_metadata_href(href: str | None) -> str | None:
    """Extract podcast id from a metadata link href."""
    if not href:
        return None
    match = re.search(r"/radio/catalog/podcast/([^/]+)", href)
    if match:
        return match.group(1)
    return None


def pick_stream_url(manifest: Any) -> tuple[str, Any | None]:
    """Select the best stream URL from an NRK playback manifest."""
    playable = manifest.playable
    if playable is None:
        raise UnplayableMediaError("No playable stream in NRK manifest")
    if playable.resolve:
        return playable.resolve, None

    assets = playable.assets or []
    if not assets:
        raise UnplayableMediaError("No assets in NRK manifest")

    def score(asset: Any) -> int:
        mime = asset.mime_type.lower()
        url = asset.url.lower()
        if "mpegurl" in mime or url.endswith(".m3u8"):
            return 100
        if "aac" in mime:
            return 70
        if "mp3" in mime:
            return 60
        return 10

    chosen = sorted(assets, key=score, reverse=True)[0]
    return chosen.url, chosen


def stream_type_from_url_and_asset(url: str, asset: Any | None) -> StreamType:
    """Infer stream type from URL and asset metadata."""
    if ".m3u8" in url.lower():
        return StreamType.HLS
    if asset and "mpegurl" in asset.mime_type.lower():
        return StreamType.HLS
    return StreamType.HTTP


def content_type_from_url_and_asset(url: str, asset: Any | None) -> ContentType:
    """Infer content type from URL and asset metadata."""
    if asset is not None:
        if asset.mime_type:
            return ContentType.try_parse(asset.mime_type)
        if asset.format:
            return ContentType.try_parse(asset.format)
    return ContentType.try_parse(url)
