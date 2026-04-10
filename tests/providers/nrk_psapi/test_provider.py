"""Tests for the NRK PSAPI provider."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from music_assistant_models.enums import MediaType, StreamType
from music_assistant_models.errors import LoginFailed, ProviderPermissionDenied
from music_assistant_models.media_items import BrowseFolder, Podcast, Radio

from music_assistant.providers.nrk_psapi import SUPPORTED_FEATURES, NrkPsapiProvider
from music_assistant.providers.nrk_psapi.helpers import NrkItemId


def _image(url: str = "https://example.com/image.jpg", width: int = 640) -> SimpleNamespace:
    return SimpleNamespace(url=url, width=width)


def _episode(
    episode_id: str = "ep-1",
    title: str = "Episode 1",
    subtitle: str | None = "Subtitle",
    duration: int = 120,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=episode_id,
        titles=SimpleNamespace(title=title, subtitle=subtitle),
        duration=timedelta(seconds=duration),
        square_image=[_image("https://example.com/episode-square.jpg", 800)],
        image=[_image("https://example.com/episode.jpg", 400)],
    )


def _podcast(item_id: str = "pod-1", title: str = "Podcast 1") -> Podcast:
    return Podcast(
        name=title,
        item_id=item_id,
        provider="nrk_psapi_test",
        provider_mappings=set(),
    )


@pytest.fixture
def provider() -> NrkPsapiProvider:
    """Create provider instance with mocked dependencies."""
    mass = Mock()
    mass.http_session = AsyncMock()
    mass.version = "1.0.0"
    mass.cache = Mock()
    mass.cache.get = AsyncMock(return_value=None)
    mass.cache.set = AsyncMock()

    manifest = Mock()
    manifest.domain = "nrk_psapi"

    config = Mock()
    config.name = "NRK Test"
    config.instance_id = "nrk_psapi_test"
    config.enabled = True
    cast("Any", config.get_value).side_effect = lambda key, default=None: {
        "podcast_limit": 250,
        "episode_limit": 200,
        "replay_items": 8,
        "radio_page": "direkte",
        "log_level": "INFO",
    }.get(key, default)

    provider = NrkPsapiProvider(mass, manifest, config, SUPPORTED_FEATURES)
    provider.podcast_limit = 250
    provider.episode_limit = 200
    provider.replay_items = 8
    provider.radio_page = "direkte"
    provider._api = Mock()
    provider._nrk_not_found_error = Exception
    return provider


def _set_config_values(provider: NrkPsapiProvider, overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "podcast_limit": 250,
        "episode_limit": 200,
        "replay_items": 8,
        "radio_page": "direkte",
        "log_level": "INFO",
        "username": None,
        "password": None,
    }
    values.update(overrides)
    cast("Any", provider.config.get_value).side_effect = lambda key, default=None: values.get(
        key, default
    )


@pytest.mark.asyncio
async def test_browse_root_returns_expected_folders(provider: NrkPsapiProvider) -> None:
    """Root browse returns folders for radio, replay, and podcasts."""
    result = await provider.browse("nrk_psapi_test://")

    assert [item.item_id for item in result] == ["radios", "replay", "podcasts"]
    assert all(isinstance(item, BrowseFolder) for item in result)


@pytest.mark.asyncio
async def test_browse_replay_returns_replay_podcasts(provider: NrkPsapiProvider) -> None:
    """Replay browse wraps live channels as replay podcasts."""
    channel = SimpleNamespace(id="p1", title="NRK P1", image=[_image()], entries=[])
    provider.__dict__["_get_radio_page_channels"] = AsyncMock(
        return_value=[("p1", "NRK P1", [_image()])]
    )
    provider.__dict__["_get_live_channel"] = AsyncMock(return_value=channel)

    result = await provider.browse("nrk_psapi_test://replay")

    assert len(result) == 1
    assert isinstance(result[0], Podcast)
    assert result[0].item_id == "replay:p1"
    assert result[0].name == "NRK P1 replay"


@pytest.mark.asyncio
async def test_search_maps_radio_and_podcast_results(provider: NrkPsapiProvider) -> None:
    """Search maps radio and podcast results into Music Assistant objects."""
    provider._api.search = AsyncMock(
        return_value=SimpleNamespace(
            results=SimpleNamespace(
                channels=SimpleNamespace(
                    results=[SimpleNamespace(id="p1", title="NRK P1", images=[_image()])]
                ),
                series=SimpleNamespace(
                    results=[
                        SimpleNamespace(
                            series_id="pod-1",
                            title="Podcast 1",
                            square_images=[_image("https://example.com/podcast.jpg", 900)],
                            images=[_image()],
                        )
                    ]
                ),
                episodes=SimpleNamespace(
                    results=[
                        SimpleNamespace(
                            episode_id="ep-1",
                            series_id="pod-1",
                            series_title="Podcast 1",
                            title="Episode 1",
                            square_images=[_image("https://example.com/ep.jpg", 700)],
                            images=[_image()],
                        )
                    ]
                ),
            )
        )
    )

    result = await provider.search(
        "nrk",
        [MediaType.RADIO, MediaType.PODCAST, MediaType.PODCAST_EPISODE],
        limit=5,
    )

    assert len(result.radio) == 1
    assert isinstance(result.radio[0], Radio)
    assert len(result.podcasts) == 1
    assert isinstance(result.podcasts[0], Podcast)


@pytest.mark.asyncio
async def test_get_podcast_episode_uses_stable_composite_id(provider: NrkPsapiProvider) -> None:
    """Podcast episode lookup uses encoded provider ids instead of hidden state."""
    provider.__dict__["get_podcast"] = AsyncMock(return_value=_podcast("pod-1", "Podcast 1"))
    provider._api.get_episode = AsyncMock(return_value=_episode())

    result = await provider.get_podcast_episode(NrkItemId.podcast_episode("pod-1", "ep-1"))

    provider._api.get_episode.assert_awaited_once_with("pod-1", "ep-1")
    assert result.item_id == NrkItemId.podcast_episode("pod-1", "ep-1")


@pytest.mark.asyncio
async def test_get_podcast_episode_uses_metadata_fallback_for_legacy_id(
    provider: NrkPsapiProvider,
) -> None:
    """Legacy raw episode ids still resolve through playback metadata."""
    provider.__dict__["get_podcast"] = AsyncMock(return_value=_podcast("pod-1", "Podcast 1"))
    provider._api.get_playback_metadata = AsyncMock(
        return_value=SimpleNamespace(
            podcast=SimpleNamespace(
                _links={"self": SimpleNamespace(href="/radio/catalog/podcast/pod-1")}
            )
        )
    )
    provider._api.get_episode = AsyncMock(return_value=_episode())

    result = await provider.get_podcast_episode("ep-1")

    provider._api.get_playback_metadata.assert_awaited_once_with("ep-1", podcast=True)
    provider._api.get_episode.assert_awaited_once_with("pod-1", "ep-1")
    assert result.item_id == NrkItemId.podcast_episode("pod-1", "ep-1")
    assert result.podcast.item_id == "pod-1"
    assert result.duration == 120


@pytest.mark.asyncio
async def test_get_stream_details_for_live_radio_uses_live_buffer_seek(
    provider: NrkPsapiProvider,
) -> None:
    """Live radio enables seek when the manifest exposes a live buffer."""
    provider._api.get_playback_manifest = AsyncMock(
        return_value=SimpleNamespace(
            playable=SimpleNamespace(
                resolve=None,
                assets=[
                    SimpleNamespace(
                        url="https://example.com/live.m3u8",
                        mime_type="application/vnd.apple.mpegurl",
                        format="hls",
                        encrypted=False,
                    )
                ],
                live_buffer={"duration": 7200},
            )
        )
    )

    result = await provider.get_stream_details("p1", MediaType.RADIO)

    assert result.stream_type == StreamType.HLS
    assert result.can_seek is True
    assert result.allow_seek is True
    provider._api.get_playback_manifest.assert_awaited_once_with("p1", channel=True)


@pytest.mark.asyncio
async def test_get_stream_details_for_replay_episode_is_seekable(
    provider: NrkPsapiProvider,
) -> None:
    """Replay/program episodes are always exposed as seekable podcast episodes."""
    provider._api.get_playback_manifest = AsyncMock(
        return_value=SimpleNamespace(
            playable=SimpleNamespace(
                resolve="https://example.com/replay.mp3",
                assets=None,
                live_buffer=None,
            )
        )
    )

    result = await provider.get_stream_details("program:p1:prog-1", MediaType.PODCAST_EPISODE)

    assert result.stream_type == StreamType.HTTP
    assert result.can_seek is True
    assert result.allow_seek is True
    assert result.path == "https://example.com/replay.mp3"
    provider._api.get_playback_manifest.assert_awaited_once_with("prog-1", program=True)


@pytest.mark.asyncio
async def test_get_stream_details_for_podcast_episode_uses_raw_episode_id(
    provider: NrkPsapiProvider,
) -> None:
    """Composite podcast episode ids are translated back to the raw NRK episode id for manifests."""
    provider._api.get_playback_manifest = AsyncMock(
        return_value=SimpleNamespace(
            playable=SimpleNamespace(
                resolve="https://example.com/podcast.mp3",
                assets=None,
                live_buffer=None,
            )
        )
    )

    result = await provider.get_stream_details(
        NrkItemId.podcast_episode("pod-1", "ep-1"),
        MediaType.PODCAST_EPISODE,
    )

    assert result.path == "https://example.com/podcast.mp3"
    assert result.can_seek is True
    provider._api.get_playback_manifest.assert_awaited_once_with("ep-1", podcast=True)


@pytest.mark.asyncio
async def test_get_stream_details_maps_geoblocked_to_permission_denied(
    provider: NrkPsapiProvider,
) -> None:
    """Geo-blocked NRK content should map to a provider permission error."""
    provider._nrk_geo_blocked_error = RuntimeError
    provider._api.get_playback_manifest = AsyncMock(side_effect=RuntimeError("geo blocked"))

    with pytest.raises(ProviderPermissionDenied):
        await provider.get_stream_details("p1", MediaType.RADIO)


@pytest.mark.asyncio
async def test_probe_geoblocking_fails_setup_on_geo_block(provider: NrkPsapiProvider) -> None:
    """Setup probe should fail fast when NRK reports geoblocking."""
    provider._nrk_geo_blocked_error = RuntimeError
    provider._api.radio_page = AsyncMock(side_effect=RuntimeError("geo blocked"))

    with pytest.raises(ProviderPermissionDenied):
        await provider._probe_geoblocking()


@pytest.mark.asyncio
async def test_configure_auth_sets_login_details_per_instance(provider: NrkPsapiProvider) -> None:
    """Configured NRK credentials are attached to the provider instance auth client only."""
    _set_config_values(
        provider,
        {
            "username": "user@example.com",
            "password": "secret",
        },
    )
    provider._api = SimpleNamespace(
        auth_client=SimpleNamespace(
            login_details=None,
            async_get_access_token_refreshed=AsyncMock(return_value="token"),
        )
    )
    fake_nrk_module = SimpleNamespace(
        NrkUserLoginDetails=lambda email, password: SimpleNamespace(email=email, password=password)
    )

    await provider._configure_auth(fake_nrk_module)

    assert provider._authenticated is True
    assert provider._api.auth_client.login_details.email == "user@example.com"
    assert provider._api.auth_client.login_details.password == "secret"
    provider._api.auth_client.async_get_access_token_refreshed.assert_awaited_once()


@pytest.mark.asyncio
async def test_configure_auth_requires_both_credentials(provider: NrkPsapiProvider) -> None:
    """Half-configured credentials should fail fast instead of sharing ambiguous auth state."""
    _set_config_values(provider, {"username": "user@example.com", "password": None})

    with pytest.raises(LoginFailed):
        await provider._configure_auth(SimpleNamespace(NrkUserLoginDetails=object))
