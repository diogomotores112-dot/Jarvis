import json
import sys
import threading
from pathlib import Path

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    _SPOTIPY = True
except ImportError:
    _SPOTIPY = False

_REDIRECT_URI = "http://127.0.0.1:8888/callback"
_SCOPE = (
    "user-modify-playback-state "
    "user-read-playback-state "
    "user-read-currently-playing"
)


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


_BASE        = _base_dir()
_CONFIG_PATH = _BASE / "config" / "api_keys.json"
_CACHE_PATH  = _BASE / "config" / ".spotify_cache"


def _load_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


_client_lock = threading.Lock()
_client: "spotipy.Spotify | None" = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client

        if not _SPOTIPY:
            raise RuntimeError("spotipy not installed. Run: pip install spotipy")

        cfg          = _load_config()
        client_id    = cfg.get("spotify_client_id", "")
        client_secret = cfg.get("spotify_client_secret", "")

        if not client_id or not client_secret:
            raise RuntimeError(
                "Spotify is not configured. Add 'spotify_client_id' and "
                "'spotify_client_secret' to config/api_keys.json (from "
                "developer.spotify.com/dashboard)."
            )

        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=_REDIRECT_URI,
            scope=_SCOPE,
            cache_path=str(_CACHE_PATH),
            open_browser=True,
        )
        _client = spotipy.Spotify(auth_manager=auth_manager)
        return _client


def _active_device_id(sp) -> str | None:
    devices = sp.devices().get("devices", [])
    if not devices:
        return None
    for d in devices:
        if d.get("is_active"):
            return d["id"]
    return devices[0]["id"]


def _search(sp, query: str, item_type: str):
    result = sp.search(q=query, type=item_type, limit=1)
    items = result.get(f"{item_type}s", {}).get("items", [])
    if not items:
        return None
    item = items[0]
    name = item.get("name", query)
    if item_type == "track":
        artists = ", ".join(a["name"] for a in item.get("artists", []))
        label = f"{name} — {artists}" if artists else name
    else:
        label = name
    return item["uri"], label


def spotify_control(
    parameters=None,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params = parameters or {}
    action = params.get("action", "play").lower().strip()
    query  = params.get("query", "").strip()
    item_type = params.get("type", "track").lower().strip()
    if item_type not in ("track", "artist", "album", "playlist"):
        item_type = "track"

    if player:
        player.write_log(f"[Spotify] {action} {query}".strip())

    try:
        sp = _get_client()
    except Exception as e:
        return str(e)

    try:
        if action == "play":
            if not query:
                return "No song, artist, album, or playlist name given."

            found = _search(sp, query, item_type)
            if not found:
                return f"Couldn't find '{query}' on Spotify."
            uri, label = found

            device_id = _active_device_id(sp)
            if not device_id:
                return (
                    "No active Spotify device found. Open Spotify on this "
                    "computer or phone first, then try again."
                )

            if item_type == "track":
                sp.start_playback(device_id=device_id, uris=[uri])
            else:
                sp.start_playback(device_id=device_id, context_uri=uri)

            return f"Playing {label}."

        if action == "queue":
            if not query:
                return "No song name given."
            found = _search(sp, query, "track")
            if not found:
                return f"Couldn't find '{query}' on Spotify."
            uri, label = found
            device_id = _active_device_id(sp)
            sp.add_to_queue(uri, device_id=device_id)
            return f"Added {label} to the queue."

        if action == "pause":
            sp.pause_playback(device_id=_active_device_id(sp))
            return "Paused."

        if action == "resume":
            sp.start_playback(device_id=_active_device_id(sp))
            return "Resumed."

        if action == "next":
            sp.next_track(device_id=_active_device_id(sp))
            return "Skipped to next track."

        if action == "previous":
            sp.previous_track(device_id=_active_device_id(sp))
            return "Went back to the previous track."

        if action == "volume":
            level = int(params.get("level", 50))
            level = max(0, min(100, level))
            sp.volume(level, device_id=_active_device_id(sp))
            return f"Volume set to {level}%."

        if action == "current":
            playing = sp.currently_playing()
            if not playing or not playing.get("item"):
                return "Nothing is currently playing."
            item    = playing["item"]
            artists = ", ".join(a["name"] for a in item.get("artists", []))
            return f"Currently playing: {item['name']} — {artists}"

        return f"Unknown Spotify action: '{action}'"

    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 403:
            return (
                "Spotify refused this action — playback control requires "
                "Spotify Premium."
            )
        if e.http_status == 404:
            return (
                "No active Spotify device found. Open Spotify on this "
                "computer or phone first, then try again."
            )
        return f"Spotify error: {e}"
    except Exception as e:
        return f"Spotify control failed: {e}"
