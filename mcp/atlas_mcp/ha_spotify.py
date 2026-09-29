"""Play one of the Spotify account's own playlists on a Home Assistant
Spotify media player, found by its spoken name.

One rule, restated from `ha.py`: every request this module makes is
preceded by `safety.allow_call`, and `Denied` is never caught here. It
propagates out of the handler with its `reason` intact, and `ha.py` turns it
into a `ToolError` so the reason is spoken as written. Every other failure
in this module is also raised as a `Denied` with a speakable reason, so the
brain never sees a Home Assistant error dict it could retry to the round
cap.

This module imports only from `atlas_mcp.safety` and `atlas_mcp.registry`.
It never imports `atlas_mcp.ha`: that would be circular, and importing it
loads a policy.

Spotify through the Home Assistant `spotify` integration is the only music
source here. The integration is `homeassistant/components/spotify/` in
home-assistant/core (media player, browse, and const modules).
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Sequence

import httpx

from atlas_mcp.registry import RegistryError
from atlas_mcp.safety import Denied, Policy, allow_call

# Feature bits of `MediaPlayerEntityFeature`
# (homeassistant/components/media_player/const.py).
PLAY_MEDIA_FEATURE = 512
SELECT_SOURCE_FEATURE = 2048
BROWSE_MEDIA_FEATURE = 131072

# Browse ids of the account's own playlists. The integration strips the
# `spotify://` prefix from the type (spotify integration, browse module).
PLAYLISTS_CONTENT_TYPE = "spotify://current_user_playlists"
PLAYLISTS_CONTENT_ID = "current_user_playlists"
# `media_content_type` of each playlist child in that browse result.
PLAYLIST_CHILD_TYPE = "spotify://playlist"
# `play_media` accepts `playlist` and uses the id as the context URI
# (spotify integration, media player module).
PLAY_MEDIA_TYPE = "playlist"

_MATCH_THRESHOLD = 0.75
_TIE_MARGIN = 0.1
_SUBSET_SCORE = 0.9
_MAX_NAMED = 3

_NON_WORD_RE = re.compile(r"[^\w ]|_", re.UNICODE)


def _normalize(text: str) -> str:
    """Casefold, turn everything that is not a letter, digit, or space into a
    space, collapse runs of spaces, and strip."""
    return " ".join(_NON_WORD_RE.sub(" ", text.casefold()).split())


def _spoken_list(items: Sequence[str], conjunction: str) -> str:
    """`a`, `a or b`, `a, b or c`."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def _spoken_forms(spoken: str, what: str) -> list[str]:
    """The normalized spoken text, and that text again with a leading `my `
    or `the ` and a trailing ` {what}` removed."""
    base = _normalize(spoken)
    stripped = base
    for lead in ("my ", "the "):
        if stripped.startswith(lead):
            stripped = stripped[len(lead) :]
            break
    tail = f" {what}"
    if stripped.endswith(tail):
        stripped = stripped[: -len(tail)]
    forms = [base]
    if stripped and stripped != base:
        forms.append(stripped)
    return forms


def _score(forms: Sequence[str], candidate: str) -> float:
    best = 0.0
    candidate_words = set(candidate.split())
    for form in forms:
        best = max(best, difflib.SequenceMatcher(None, form, candidate).ratio())
        form_words = set(form.split())
        if form_words and candidate_words and (
            form_words <= candidate_words or candidate_words <= form_words
        ):
            best = max(best, _SUBSET_SCORE)
    return best


def match_spoken_name(spoken: str, candidates: Sequence[str], *, what: str) -> str:
    """The candidate that `spoken` names, or `Denied` with a speakable reason.

    Order of checks: an exact match after normalization (two or more is a
    tie), then a fuzzy or word-subset score. A best score under
    `_MATCH_THRESHOLD` is a miss, and a runner-up within `_TIE_MARGIN` of the
    best is a tie. A miss and a tie both name the closest candidates so the
    person can say which one they meant.
    """
    candidates = list(candidates)
    if not candidates:
        raise Denied(f"i can't see any spotify {what}s right now")
    shown = spoken.strip()
    forms = _spoken_forms(spoken, what)
    normalized = [_normalize(c) for c in candidates]

    exact = [c for c, n in zip(candidates, normalized) if n in forms]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        names = _spoken_list(exact[:_MAX_NAMED], "or")
        raise Denied(f"more than one spotify {what} sounds like {shown} -- {names}?")

    ranked = sorted(
        ((_score(forms, n), c) for c, n in zip(candidates, normalized)),
        key=lambda pair: (-pair[0], pair[1]),
    )
    best_score, best = ranked[0]
    if best_score < _MATCH_THRESHOLD:
        closest = [c for _, c in ranked[:_MAX_NAMED]]
        lead = "the closest is" if len(closest) == 1 else "the closest are"
        raise Denied(
            f"i couldn't find a spotify {what} called {shown} -- {lead} {_spoken_list(closest, 'or')}"
        )
    if len(ranked) > 1 and ranked[1][0] >= best_score - _TIE_MARGIN:
        close = [c for score, c in ranked if score >= best_score - _TIE_MARGIN][:_MAX_NAMED]
        raise Denied(
            f"more than one spotify {what} sounds like {shown} -- {_spoken_list(close, 'or')}?"
        )
    return best


async def handle_play_spotify_playlist(
    policy: Policy,
    client: httpx.AsyncClient,
    base_url: str,
    token: str,
    browser: Any,
    entity_id: str,
    name: str,
    *,
    source: str | None = None,
    default_source: str | None = None,
) -> dict[str, Any]:
    """Play the account's playlist that `name` best matches on a Spotify
    media player.

    `browser` is anything with `async run_command(command)` (a
    `HaRegistryClient`), or `None`.

    Order: read the state, select a speaker if needed, browse the playlists,
    then play. The order follows Home Assistant's feature gates. The spotify
    integration reports only SELECT_SOURCE while nothing is playing (and no
    features at all for an account that is not Premium). `play_media` needs
    PLAY_MEDIA, and the websocket browse command is refused with
    `not_supported` unless BROWSE_MEDIA is set. An idle player therefore
    needs a speaker selected first. `select_source` returns only after the
    state refreshed, so there is no state re-read afterwards (latency).

    `default_source` is the operator's configured speaker. It applies only
    when `source` is blank, and it moves playback there when Spotify plays
    somewhere else. A named `source` always wins. A blank or missing default
    keeps the behavior described above.

    Only the account's first 48 playlists can be matched: that is the most
    the integration requests from Spotify.

    Every failure is a speakable `Denied`. `allow_call` runs for
    `play_media` before any request, and for `select_source` before that
    POST, so a denied entity causes zero HTTP requests and zero websocket
    commands. Each POST is sent once, with no retry.
    """
    if not name.strip():
        raise Denied("which playlist do you want me to play")

    _, _, [checked] = allow_call(policy, "media_player", "play_media", [entity_id])

    if browser is None:
        raise Denied("i can't reach home assistant to read your spotify playlists")

    headers = {"Authorization": f"Bearer {token}"}
    response = await client.get(f"{base_url}/api/states/{checked}", headers=headers)
    if response.status_code == 404:
        raise Denied("i can't find that spotify player in home assistant")
    if not response.is_success:
        raise Denied("home assistant didn't answer when i asked about that spotify player")
    state = response.json()
    attributes = state.get("attributes", {}) if isinstance(state, dict) else {}
    features = attributes.get("supported_features", 0)
    features = features if isinstance(features, int) else 0
    raw_sources = attributes.get("source_list")
    source_list = [s for s in raw_sources if isinstance(s, str)] if isinstance(raw_sources, list) else []
    current_source = attributes.get("source")

    can_play = bool(features & PLAY_MEDIA_FEATURE) and bool(features & BROWSE_MEDIA_FEATURE)
    wanted = (source or "").strip() or (default_source or "").strip()
    wants_other = bool(wanted) and not (
        isinstance(current_source, str) and current_source.casefold() == wanted.casefold()
    )
    chosen: str | None = None
    if not can_play or wants_other:
        if not (features & PLAY_MEDIA_FEATURE) and not (features & SELECT_SOURCE_FEATURE):
            raise Denied("that spotify player can't be controlled from here")
        if wanted:
            chosen = match_spoken_name(wanted, source_list, what="speaker")
        elif len(source_list) == 1:
            chosen = source_list[0]
        elif not source_list:
            raise Denied("spotify can't see any speaker to play on right now -- open spotify on one first")
        else:
            raise Denied(
                "spotify isn't playing anywhere yet -- which speaker should it use: "
                f"{_spoken_list(source_list, 'or')}?"
            )
        allow_call(policy, "media_player", "select_source", [checked])
        select = await client.post(
            f"{base_url}/api/services/media_player/select_source",
            headers=headers,
            json={"entity_id": [checked], "source": chosen},
        )
        if not select.is_success:
            raise Denied(f"home assistant wouldn't move spotify to {chosen}")

    try:
        result = await browser.run_command(
            {
                "type": "media_player/browse_media",
                "entity_id": checked,
                "media_content_type": PLAYLISTS_CONTENT_TYPE,
                "media_content_id": PLAYLISTS_CONTENT_ID,
            }
        )
    except RegistryError as exc:
        if chosen is not None:
            raise Denied(
                f"spotify isn't ready to play on {chosen} yet -- try again in a moment"
            ) from exc
        raise Denied("i couldn't read your spotify playlists right now") from exc

    children = result.get("children") if isinstance(result, dict) else None
    playlists: list[tuple[str, str]] = []
    seen_uris: set[str] = set()
    for child in children or []:
        if not isinstance(child, dict) or child.get("media_content_type") != PLAYLIST_CHILD_TYPE:
            continue
        title, uri = child.get("title"), child.get("media_content_id")
        if not (isinstance(title, str) and title and isinstance(uri, str) and uri):
            continue
        if uri in seen_uris:
            continue
        seen_uris.add(uri)
        playlists.append((title, uri))

    title = match_spoken_name(name, [t for t, _ in playlists], what="playlist")
    uri = next(u for t, u in playlists if t == title)

    played = await client.post(
        f"{base_url}/api/services/media_player/play_media",
        headers=headers,
        json={
            "entity_id": [checked],
            "media_content_id": uri,
            "media_content_type": PLAY_MEDIA_TYPE,
        },
    )
    if not played.is_success:
        raise Denied(f"home assistant wouldn't start {title} -- it answered {played.status_code}")
    return {"playlist": title, "changed": played.json()}
