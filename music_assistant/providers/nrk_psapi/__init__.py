"""NRK Radio/Podcast provider based on nrk-psapi."""

from __future__ import annotations

import importlib
from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import (
    ConfigEntryType,
    MediaType,
    ProviderFeature,
)
from music_assistant_models.errors import (
    LoginFailed,
    MediaNotFoundError,
    ProviderPermissionDenied,
    UnplayableMediaError,
)
from music_assistant_models.media_items import (
    AudioFormat,
    BrowseFolder,
    ItemMapping,
    MediaItemType,
    Podcast,
    PodcastEpisode,
    Radio,
    SearchResults,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.constants import CONF_PASSWORD, CONF_USERNAME
from music_assistant.controllers.cache import use_cache
from music_assistant.models.music_provider import MusicProvider

from .helpers import (
    NrkItemId,
    add_image,
    content_type_from_url_and_asset,
    parse_podcast_id_from_metadata_href,
    pick_stream_url,
    provider_mapping,
    stream_type_from_url_and_asset,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

CONF_PODCAST_LIMIT = "podcast_limit"
CONF_EPISODE_LIMIT = "episode_limit"
CONF_REPLAY_ITEMS = "replay_items"
CONF_RADIO_PAGE = "radio_page"

SUPPORTED_FEATURES = {
    ProviderFeature.BROWSE,
    ProviderFeature.SEARCH,
}


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return NrkPsapiProvider(mass, manifest, config, SUPPORTED_FEATURES)


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return config entries for this provider."""
    # ruff: noqa: ARG001
    values = values or {}
    return (
        ConfigEntry(
            key=CONF_USERNAME,
            type=ConfigEntryType.STRING,
            label="NRK email",
            required=False,
            value=values.get(CONF_USERNAME),
            description="Optional. Set together with password to enable per-user personalization.",
        ),
        ConfigEntry(
            key=CONF_PASSWORD,
            type=ConfigEntryType.SECURE_STRING,
            label="NRK password",
            required=False,
            value=values.get(CONF_PASSWORD),
            description="Optional. Stored per provider instance and used only for NRK personalization.",
        ),
        ConfigEntry(
            key=CONF_PODCAST_LIMIT,
            type=ConfigEntryType.INTEGER,
            label="Max podcasts in browse",
            required=False,
            default_value=250,
            value=values.get(CONF_PODCAST_LIMIT),
            description="Maximum number of podcasts shown in browse.",
            advanced=True,
        ),
        ConfigEntry(
            key=CONF_EPISODE_LIMIT,
            type=ConfigEntryType.INTEGER,
            label="Max episodes per podcast",
            required=False,
            default_value=200,
            value=values.get(CONF_EPISODE_LIMIT),
            description="Maximum number of episodes loaded per podcast. 0 means all.",
            advanced=True,
        ),
        ConfigEntry(
            key=CONF_REPLAY_ITEMS,
            type=ConfigEntryType.INTEGER,
            label="Replay items per radio channel",
            required=False,
            default_value=8,
            value=values.get(CONF_REPLAY_ITEMS),
            description="How many recent programs to expose as replay episodes per channel.",
            advanced=True,
        ),
        ConfigEntry(
            key=CONF_RADIO_PAGE,
            type=ConfigEntryType.STRING,
            label="Radio page id",
            required=False,
            default_value="direkte",
            value=values.get(CONF_RADIO_PAGE),
            description="NRK radio page used to discover live channels.",
            advanced=True,
        ),
    )


class NrkPsapiProvider(MusicProvider):
    """Provider implementation for NRK radio and podcasts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize provider."""
        super().__init__(*args, **kwargs)
        self._api: Any = None
        self._authenticated = False
        self._nrk_not_found_error: type[Exception] = Exception
        self._nrk_geo_blocked_error: type[Exception] | None = None

    @property
    def is_streaming_provider(self) -> bool:
        """Return True for streaming providers."""
        return True

    async def handle_async_init(self) -> None:
        """Set up API client and provider settings."""

        def _config_int(value: object, default: int) -> int:
            if isinstance(value, bool):
                return default
            if isinstance(value, int | float | str):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return default
            return default

        self.podcast_limit = _config_int(self.config.get_value(CONF_PODCAST_LIMIT), 250)
        self.episode_limit = _config_int(self.config.get_value(CONF_EPISODE_LIMIT), 200)
        self.replay_items = _config_int(self.config.get_value(CONF_REPLAY_ITEMS), 8)
        self.radio_page = (
            str(self.config.get_value(CONF_RADIO_PAGE) or "direkte").strip() or "direkte"
        )

        nrk_psapi = importlib.import_module("nrk_psapi")
        nrk_exceptions = importlib.import_module("nrk_psapi.exceptions")
        self._nrk_not_found_error = getattr(nrk_exceptions, "NrkPsApiNotFoundError", Exception)
        geo_blocked_error = getattr(nrk_exceptions, "NrkPsApiGeoBlockedError", None)
        self._nrk_geo_blocked_error = (
            geo_blocked_error if isinstance(geo_blocked_error, type) else None
        )
        self._api = nrk_psapi.NrkPodcastAPI(
            session=self.mass.http_session,
            user_agent=f"Music Assistant/{self.mass.version}",
            disable_credentials_storage=True,
            enable_cache=True,
        )
        await self._configure_auth(nrk_psapi)
        await self._probe_geoblocking()

    def _raise_geo_blocked(self, err: Exception) -> None:
        """Map nrk-psapi geoblocking errors to a provider permission error."""
        if self._nrk_geo_blocked_error and isinstance(err, self._nrk_geo_blocked_error):
            raise ProviderPermissionDenied(
                "NRK content is geo-blocked and only available in supported regions."
            ) from err

    async def _probe_geoblocking(self) -> None:
        """Probe one lightweight endpoint during setup to fail fast on geoblocking only."""
        api = await self._get_api()
        try:
            await api.radio_page(self.radio_page)
        except Exception as err:
            self._raise_geo_blocked(err)
            self.logger.debug(
                "NRK setup probe failed for non-geo reason; deferring error to runtime: %s",
                err,
            )

    async def _configure_auth(self, nrk_psapi: Any) -> None:
        """Configure optional NRK account authentication for personalization."""
        username = self.config.get_value(CONF_USERNAME)
        password = self.config.get_value(CONF_PASSWORD)
        email = username.strip() if isinstance(username, str) else ""
        secret = password if isinstance(password, str) else ""

        if not email and not secret:
            self._authenticated = False
            return
        if not email or not secret:
            msg = "NRK email and password must both be set to enable personalization."
            raise LoginFailed(msg)

        self._api.auth_client.login_details = nrk_psapi.NrkUserLoginDetails(
            email=email,
            password=secret,
        )
        try:
            await self._api.auth_client.async_get_access_token_refreshed()
        except Exception as err:
            msg = "Unable to authenticate with the provided NRK credentials."
            raise LoginFailed(msg) from err
        self._authenticated = True

    async def unload(self, is_removed: bool = False) -> None:
        """Handle provider unload."""
        if self._api is not None:
            await self._api.close()

    def _parse_series_list_item(self, item: Any) -> Podcast | None:
        podcast_id = item.series_id or item.id
        if not podcast_id:
            return None
        if str(item.type) not in ("podcast", "series"):
            return None
        podcast = Podcast(
            name=item.title,
            item_id=podcast_id,
            provider=self.instance_id,
            provider_mappings={provider_mapping(self.domain, self.instance_id, podcast_id)},
        )
        add_image(podcast, item.square_images or item.images, self.domain)
        return podcast

    async def _get_api(self) -> Any:
        if self._api is None:
            raise RuntimeError("Provider is not initialized")
        return self._api

    @use_cache(3600)
    async def _get_radio_page_channels(self) -> list[tuple[str, str, Any]]:
        api = await self._get_api()
        try:
            page = await api.radio_page(self.radio_page)
        except Exception as err:
            self._raise_geo_blocked(err)
            raise
        channels: list[tuple[str, str, Any]] = []
        seen: set[str] = set()
        sections = getattr(page, "sections", []) if page is not None else []
        for section in sections:
            included = getattr(section, "included", None)
            if included is None:
                continue
            plugs = getattr(included, "plugs", [])
            for plug in plugs:
                if str(getattr(plug, "type", "")) != "channel":
                    continue
                channel = getattr(plug, "channel", None)
                if channel is None:
                    continue
                channel_id = getattr(channel, "channel_id", None)
                channel_title = getattr(channel, "channel_title", None)
                if not channel_id or channel_id in seen:
                    continue
                seen.add(channel_id)
                channels.append(
                    (channel_id, channel_title or channel_id, getattr(plug, "image", None))
                )
        return channels

    @use_cache(3600)
    async def _get_live_channel(self, channel_id: str) -> Any:
        api = await self._get_api()
        try:
            return await api.get_live_channel(channel_id)
        except self._nrk_not_found_error as err:
            raise MediaNotFoundError("Radio not found") from err
        except Exception as err:
            self._raise_geo_blocked(err)
            raise

    def _radio_from_channel_tuple(self, channel_id: str, title: str, image_obj: Any) -> Radio:
        radio = Radio(
            name=title,
            item_id=channel_id,
            provider=self.instance_id,
            provider_mappings={provider_mapping(self.domain, self.instance_id, channel_id)},
        )
        add_image(radio, image_obj, self.domain)
        return radio

    def _radio_from_channel(self, channel: Any) -> Radio:
        radio = Radio(
            name=channel.title,
            item_id=channel.id,
            provider=self.instance_id,
            provider_mappings={provider_mapping(self.domain, self.instance_id, channel.id)},
        )
        add_image(radio, channel.image, self.domain)
        return radio

    @use_cache(3600)
    async def _get_all_podcasts(self) -> list[Any]:
        api = await self._get_api()
        try:
            return cast("list[Any]", await api.get_all_podcasts())
        except Exception as err:
            self._raise_geo_blocked(err)
            raise

    @use_cache(3600)
    async def _get_podcast_obj(self, podcast_id: str) -> Any:
        api = await self._get_api()
        try:
            return await api.get_podcast(podcast_id)
        except self._nrk_not_found_error as err:
            raise MediaNotFoundError("Podcast not found") from err
        except Exception as err:
            self._raise_geo_blocked(err)
            raise

    def _podcast_from_catalog(self, podcast_obj: Any, podcast_id: str) -> Podcast:
        series = getattr(podcast_obj, "series", None)
        title = getattr(series, "title", None) or podcast_id
        podcast = Podcast(
            name=title,
            item_id=podcast_id,
            provider=self.instance_id,
            provider_mappings={provider_mapping(self.domain, self.instance_id, podcast_id)},
        )
        image_candidates = []
        if series is not None:
            if getattr(series, "square_image", None):
                image_candidates = series.square_image
            elif getattr(series, "image", None):
                image_candidates = series.image
            elif getattr(series, "poster_image", None):
                image_candidates = series.poster_image
        add_image(podcast, image_candidates, self.domain)

        episodes = getattr(podcast_obj, "episodes", None)
        if isinstance(episodes, list):
            podcast.total_episodes = len(episodes)
        return podcast

    def _podcast_from_search_result(self, result: Any) -> Podcast:
        podcast_id = result.series_id
        podcast = Podcast(
            name=result.title,
            item_id=podcast_id,
            provider=self.instance_id,
            provider_mappings={provider_mapping(self.domain, self.instance_id, podcast_id)},
        )
        add_image(podcast, result.square_images or result.images, self.domain)
        return podcast

    def _replay_podcast_from_channel(self, channel: Any) -> Podcast:
        item_id = NrkItemId.replay_podcast(channel.id)
        podcast = Podcast(
            name=f"{channel.title} replay",
            item_id=item_id,
            provider=self.instance_id,
            provider_mappings={provider_mapping(self.domain, self.instance_id, item_id)},
        )
        podcast.metadata.description = (
            "Recent live broadcasts from this NRK radio channel. "
            "These entries are seekable and suited for manual rewind/skip."
        )
        add_image(podcast, channel.image, self.domain)
        return podcast

    def _episode_from_catalog(
        self, episode: Any, podcast: Podcast, position: int
    ) -> PodcastEpisode:
        item_id = NrkItemId.podcast_episode(podcast.item_id, episode.id)
        duration = int(episode.duration.total_seconds())
        pod_episode = PodcastEpisode(
            name=episode.titles.title,
            item_id=item_id,
            provider=self.instance_id,
            position=position,
            duration=max(duration, 0),
            podcast=ItemMapping(
                item_id=podcast.item_id,
                provider=self.instance_id,
                name=podcast.name,
                media_type=MediaType.PODCAST,
            ),
            provider_mappings={provider_mapping(self.domain, self.instance_id, item_id)},
        )
        if episode.titles.subtitle:
            pod_episode.metadata.description = episode.titles.subtitle
        image_candidates = episode.square_image or episode.image
        add_image(pod_episode, image_candidates, self.domain)
        return pod_episode

    def _replay_episode_from_channel_entry(
        self, channel: Any, program_id: str, title: str, position: int
    ) -> PodcastEpisode:
        item_id = NrkItemId.program_episode(channel.id, program_id)
        podcast_id = NrkItemId.replay_podcast(channel.id)
        episode = PodcastEpisode(
            name=title,
            item_id=item_id,
            provider=self.instance_id,
            position=position,
            duration=0,
            podcast=ItemMapping(
                item_id=podcast_id,
                provider=self.instance_id,
                name=f"{channel.title} replay",
                media_type=MediaType.PODCAST,
            ),
            provider_mappings={provider_mapping(self.domain, self.instance_id, item_id)},
        )
        add_image(episode, channel.image, self.domain)
        return episode

    async def browse(self, path: str) -> Sequence[MediaItemType | ItemMapping | BrowseFolder]:
        """Browse radios and podcasts."""
        subpath = path.split("://", 1)[1] if "://" in path else ""

        if subpath == "radios":
            channels = await self._get_radio_page_channels()
            return [
                self._radio_from_channel_tuple(channel_id, title, image)
                for channel_id, title, image in channels
            ]

        if subpath == "podcasts":
            podcasts = await self._get_all_podcasts()
            items: list[Podcast] = []
            for item in podcasts:
                parsed = self._parse_series_list_item(item)
                if parsed is None:
                    continue
                items.append(parsed)
                if self.podcast_limit > 0 and len(items) >= self.podcast_limit:
                    break
            return items

        if subpath == "replay":
            channels = await self._get_radio_page_channels()
            result: list[Podcast] = []
            for channel_id, _title, _image in channels:
                channel = await self._get_live_channel(channel_id)
                result.append(self._replay_podcast_from_channel(channel))
            return result

        return [
            BrowseFolder(
                item_id="radios",
                provider=self.instance_id,
                path=f"{self.instance_id}://radios",
                name="Radio",
            ),
            BrowseFolder(
                item_id="replay",
                provider=self.instance_id,
                path=f"{self.instance_id}://replay",
                name="Radio replay",
            ),
            BrowseFolder(
                item_id="podcasts",
                provider=self.instance_id,
                path=f"{self.instance_id}://podcasts",
                name="Podcasts",
            ),
        ]

    @use_cache(3600)
    async def search(
        self,
        search_query: str,
        media_types: list[MediaType],
        limit: int = 10,
    ) -> SearchResults:
        """Search across channels and podcasts."""
        api = await self._get_api()
        result = SearchResults()
        try:
            search = await api.search(query=search_query, per_page=max(limit * 2, 20))
        except Exception as err:
            self._raise_geo_blocked(err)
            raise

        if MediaType.RADIO in media_types:
            radios: list[Radio] = []
            for channel in search.results.channels.results:
                radios.append(self._radio_from_search_result(channel))
                if len(radios) >= limit:
                    break
            result.radio = radios

        if MediaType.PODCAST in media_types:
            podcasts: list[Podcast] = []
            for item in search.results.series.results:
                podcasts.append(self._podcast_from_search_result(item))
                if len(podcasts) >= limit:
                    break
            result.podcasts = podcasts

        return result

    def _radio_from_search_result(self, result: Any) -> Radio:
        radio = Radio(
            name=result.title,
            item_id=result.id,
            provider=self.instance_id,
            provider_mappings={provider_mapping(self.domain, self.instance_id, result.id)},
        )
        add_image(radio, result.images, self.domain)
        return radio

    @use_cache(3600)
    async def get_radio(self, prov_radio_id: str) -> Radio:
        """Get full radio details by id."""
        channel = await self._get_live_channel(prov_radio_id)
        return self._radio_from_channel(channel)

    @use_cache(3600)
    async def get_podcast(self, prov_podcast_id: str) -> Podcast:
        """Get full podcast details by id."""
        if channel_id := NrkItemId.parse_replay_podcast(prov_podcast_id):
            channel = await self._get_live_channel(channel_id)
            return self._replay_podcast_from_channel(channel)

        podcast_obj = await self._get_podcast_obj(prov_podcast_id)
        return self._podcast_from_catalog(podcast_obj, prov_podcast_id)

    async def get_podcast_episodes(
        self,
        prov_podcast_id: str,
    ) -> AsyncGenerator[PodcastEpisode, None]:
        """List all episodes for the podcast."""
        api = await self._get_api()

        if channel_id := NrkItemId.parse_replay_podcast(prov_podcast_id):
            channel = await self._get_live_channel(channel_id)
            for idx, entry in enumerate(channel.entries[: self.replay_items]):
                yield self._replay_episode_from_channel_entry(
                    channel=channel,
                    program_id=entry.program_id,
                    title=entry.title,
                    position=idx,
                )
            return

        podcast = await self.get_podcast(prov_podcast_id)
        try:
            episodes = await api.get_podcast_episodes(
                podcast_id=prov_podcast_id,
                page=-1,
            )
        except Exception as err:
            self._raise_geo_blocked(err)
            raise
        for idx, episode in enumerate(episodes):
            if self.episode_limit > 0 and idx >= self.episode_limit:
                break
            yield self._episode_from_catalog(episode, podcast, idx)

    @use_cache(3600)
    async def get_podcast_episode(self, prov_episode_id: str) -> PodcastEpisode:
        """Get one podcast/replay episode by id."""
        api = await self._get_api()

        if parsed_program_episode := NrkItemId.parse_program_episode(prov_episode_id):
            channel_id, program_id = parsed_program_episode
            channel = await self._get_live_channel(channel_id)
            try:
                programme = await api.get_program(program_id)
            except Exception as err:
                self._raise_geo_blocked(err)
                raise
            episode = self._replay_episode_from_channel_entry(
                channel=channel,
                program_id=program_id,
                title=programme.temporal_titles.default_titles.main_title,
                position=0,
            )
            duration_seconds = int(programme.duration.iso8601.total_seconds())
            episode.duration = max(duration_seconds, 0)
            episode.metadata.description = str(programme.program_information)
            add_image(episode, programme.image, self.domain)
            return episode

        podcast_id: str | None = None
        episode_id = prov_episode_id
        if parsed_podcast_episode := NrkItemId.parse_podcast_episode(prov_episode_id):
            podcast_id, episode_id = parsed_podcast_episode
        if podcast_id is None:
            try:
                metadata = await api.get_playback_metadata(episode_id, podcast=True)
            except Exception as err:
                self._raise_geo_blocked(err)
                raise
            podcast_link = None
            if metadata.podcast is not None:
                podcast_link = metadata.podcast._links.get("self")
            podcast_id = parse_podcast_id_from_metadata_href(
                podcast_link.href if podcast_link else None
            )
        if podcast_id is None:
            raise MediaNotFoundError("Episode could not be resolved to a podcast id.")

        try:
            episode = await api.get_episode(podcast_id, episode_id)
        except Exception as err:
            self._raise_geo_blocked(err)
            raise
        podcast = await self.get_podcast(podcast_id)
        return self._episode_from_catalog(episode, podcast, 0)

    async def _stream_details_from_manifest(
        self,
        item_id: str,
        media_type: MediaType,
        manifest: Any,
        can_seek: bool,
    ) -> StreamDetails:
        stream_url, asset = pick_stream_url(manifest)
        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            audio_format=AudioFormat(
                content_type=content_type_from_url_and_asset(stream_url, asset),
            ),
            media_type=media_type,
            stream_type=stream_type_from_url_and_asset(stream_url, asset),
            path=stream_url,
            can_seek=can_seek,
            allow_seek=can_seek,
        )

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Get stream details for radio or podcast episode playback."""
        api = await self._get_api()

        if media_type == MediaType.RADIO:
            try:
                manifest = await api.get_playback_manifest(item_id, channel=True)
            except Exception as err:
                self._raise_geo_blocked(err)
                raise
            has_live_buffer = bool(
                getattr(getattr(manifest, "playable", None), "live_buffer", None)
            )
            return await self._stream_details_from_manifest(
                item_id=item_id,
                media_type=MediaType.RADIO,
                manifest=manifest,
                can_seek=has_live_buffer,
            )

        if media_type == MediaType.PODCAST_EPISODE:
            if parsed_program_episode := NrkItemId.parse_program_episode(item_id):
                _channel_id, program_id = parsed_program_episode
                try:
                    manifest = await api.get_playback_manifest(program_id, program=True)
                except Exception as err:
                    self._raise_geo_blocked(err)
                    raise
                return await self._stream_details_from_manifest(
                    item_id=item_id,
                    media_type=MediaType.PODCAST_EPISODE,
                    manifest=manifest,
                    can_seek=True,
                )

            if parsed_podcast_episode := NrkItemId.parse_podcast_episode(item_id):
                _podcast_id, episode_id = parsed_podcast_episode
            else:
                episode_id = item_id
            try:
                manifest = await api.get_playback_manifest(episode_id, podcast=True)
            except Exception as err:
                self._raise_geo_blocked(err)
                raise
            return await self._stream_details_from_manifest(
                item_id=item_id,
                media_type=MediaType.PODCAST_EPISODE,
                manifest=manifest,
                can_seek=True,
            )

        raise UnplayableMediaError(f"Unsupported media type: {media_type}")

    async def get_resume_position(
        self, item_id: str, media_type: MediaType
    ) -> tuple[bool, int, None]:
        """Return empty resume state until remote progress is implemented."""
        if media_type != MediaType.PODCAST_EPISODE:
            return False, 0, None
        return False, 0, None

    async def on_played(
        self,
        media_type: MediaType,
        prov_item_id: str,
        fully_played: bool,
        position: int,
        media_item: MediaItemType,
        is_playing: bool = False,
    ) -> None:
        """Handle playback callback."""
        # Placeholder for future remote progress support.
        return

    async def resolve_image(self, path: str) -> str | bytes:
        """Forward remote image URLs as-is."""
        return path
