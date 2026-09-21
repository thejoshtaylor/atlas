"""`ObserverRegistry`/`ObserverPublishingSource` (WEB-06, D-05, D-08).

Task 1 covers the fan-out itself: subscribe/unsubscribe, the non-waiting
drop-oldest publish, and the wrapper's event/barge-in behavior. Task 2 (the
route-level tests through a real `TestClient` WebSocket, driving a real turn
on each path) is added to this same file, below the Task 1 section, per the
plan's own instruction not to split the two.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from spire_voice.session.observers import ObserverPublishingSource, ObserverRegistry
from spire_voice.transports.base import SourceFormat


# --- Task 1: the fan-out -----------------------------------------------


def test_a_subscriber_receives_every_subsequently_published_event():
    registry = ObserverRegistry()
    queue = registry.subscribe()

    registry.publish({"type": "transcript.partial", "text": "a"})
    registry.publish({"type": "transcript.partial", "text": "b"})

    assert queue.get_nowait() == {"type": "transcript.partial", "text": "a"}
    assert queue.get_nowait() == {"type": "transcript.partial", "text": "b"}


def test_unsubscribe_stops_further_delivery():
    registry = ObserverRegistry()
    queue = registry.subscribe()
    registry.unsubscribe(queue)

    registry.publish({"type": "transcript.partial", "text": "a"})

    assert queue.empty()


def test_publishing_with_no_subscribers_does_nothing_and_raises_nothing():
    registry = ObserverRegistry()
    registry.publish({"type": "transcript.partial", "text": "a"})  # must not raise


def test_a_full_queue_drops_its_oldest_message_and_the_publisher_still_returns():
    registry = ObserverRegistry(max_queue_size=2)
    queue = registry.subscribe()

    registry.publish({"type": "e", "n": 1})
    registry.publish({"type": "e", "n": 2})
    # The queue is now full (bound = 2). A third publish must not block --
    # it drops the oldest (n=1) and enqueues the new one (n=3).
    registry.publish({"type": "e", "n": 3})

    assert queue.qsize() == 2
    assert queue.get_nowait() == {"type": "e", "n": 2}
    assert queue.get_nowait() == {"type": "e", "n": 3}


def test_publish_is_synchronous_and_awaits_nothing():
    # A synchronous call, not a coroutine -- calling it from ordinary (not
    # async) code is itself proof `publish` awaits nothing.
    registry = ObserverRegistry()
    registry.subscribe()
    result = registry.publish({"type": "e"})
    assert result is None


def test_two_subscribers_both_receive_the_same_publish():
    registry = ObserverRegistry()
    a = registry.subscribe()
    b = registry.subscribe()

    registry.publish({"type": "e"})

    assert a.get_nowait() == {"type": "e"}
    assert b.get_nowait() == {"type": "e"}


class _RaisingQueue:
    """A double satisfying just enough of `asyncio.Queue`'s put surface to
    drive `ObserverRegistry._put_dropping_oldest`'s outer `except Exception`
    branch without mocking the exception -- `put_nowait` raises a plain
    `RuntimeError`, not `asyncio.QueueFull`, so the drop-oldest path is
    never reached for this queue at all."""

    def put_nowait(self, item: Any) -> None:
        raise RuntimeError("this queue is broken")


def test_a_raising_queue_does_not_propagate_out_of_publish():
    registry = ObserverRegistry()
    good_queue = registry.subscribe()
    registry._queues.add(_RaisingQueue())  # type: ignore[arg-type]

    registry.publish({"type": "e"})  # must not raise

    assert good_queue.get_nowait() == {"type": "e"}


class _EventSource:
    """A minimal `AudioSource`-shaped double: `frames`/`send_audio`/
    `source_format`, plus an optional `send_event`, and a `barge_in`
    attribute settable at construction or afterward."""

    def __init__(self, *, with_send_event: bool = True, barge_in: Any = None) -> None:
        self.sent_audio: list[bytes] = []
        self.received_events: list[dict[str, Any]] = []
        self._has_send_event = with_send_event
        if barge_in is not None:
            self.barge_in = barge_in

    async def frames(self):
        yield b"chunk-1"
        yield b"chunk-2"

    async def send_audio(self, chunk: bytes) -> None:
        self.sent_audio.append(chunk)

    def source_format(self) -> SourceFormat:
        return SourceFormat("pcm", 16000)

    async def send_event(self, event: dict[str, Any]) -> None:
        if not self._has_send_event:
            raise AssertionError("send_event should not exist on this double")
        self.received_events.append(event)


async def test_send_event_publishes_with_the_source_name_added():
    registry = ObserverRegistry()
    queue = registry.subscribe()
    wrapped = _EventSource()
    source = ObserverPublishingSource(wrapped, "camera", registry)

    await source.send_event({"type": "transcript.partial", "text": "hello"})

    published = queue.get_nowait()
    assert published == {"type": "transcript.partial", "text": "hello", "source": "camera"}


async def test_send_event_forwards_the_original_unmodified_event_to_the_wrapped_source():
    registry = ObserverRegistry()
    registry.subscribe()
    wrapped = _EventSource()
    source = ObserverPublishingSource(wrapped, "camera", registry)

    await source.send_event({"type": "transcript.partial", "text": "hello"})

    # The wrapped (real) source's own send_event never sees the "source" key
    # -- that field is an observer-only addition, per the module docstring.
    assert wrapped.received_events == [{"type": "transcript.partial", "text": "hello"}]


async def test_send_event_does_nothing_extra_when_the_wrapped_source_has_no_send_event():
    class _Bare:
        def __init__(self) -> None:
            self.frames_called = False

        async def frames(self):
            yield b"x"

        async def send_audio(self, chunk: bytes) -> None:
            pass

        def source_format(self) -> SourceFormat:
            return SourceFormat("pcm", 16000)

    registry = ObserverRegistry()
    queue = registry.subscribe()
    source = ObserverPublishingSource(_Bare(), "camera", registry)

    await source.send_event({"type": "transcript.partial", "text": "hi"})  # must not raise

    assert queue.get_nowait() == {"type": "transcript.partial", "text": "hi", "source": "camera"}


async def test_frames_send_audio_and_source_format_forward_straight_through():
    registry = ObserverRegistry()
    wrapped = _EventSource()
    source = ObserverPublishingSource(wrapped, "camera", registry)

    chunks = [chunk async for chunk in source.frames()]
    assert chunks == [b"chunk-1", b"chunk-2"]

    await source.send_audio(b"reply-chunk")
    assert wrapped.sent_audio == [b"reply-chunk"]

    assert source.source_format() == SourceFormat("pcm", 16000)


def test_barge_in_set_before_wrapping_reads_through():
    sentinel = object()
    wrapped = _EventSource(barge_in=sentinel)
    source = ObserverPublishingSource(wrapped, "camera", ObserverRegistry())

    assert source.barge_in is sentinel


def test_barge_in_set_after_wrapping_writes_and_reads_through():
    wrapped = _EventSource()
    source = ObserverPublishingSource(wrapped, "camera", ObserverRegistry())
    sentinel = object()

    source.barge_in = sentinel

    assert source.barge_in is sentinel
    assert wrapped.barge_in is sentinel


def test_barge_in_defaults_to_none_when_the_wrapped_source_never_set_one():
    wrapped = _EventSource()
    source = ObserverPublishingSource(wrapped, "camera", ObserverRegistry())

    assert source.barge_in is None


# --- Task 2: one read-only socket beside the turn socket, on both sources --


def _fake_run_turn_config(tmp_path):
    from types import SimpleNamespace

    from spire_voice.config import SessionConfig

    return SimpleNamespace(
        session=SessionConfig(dir=str(tmp_path)),
        brain=SimpleNamespace(max_tool_rounds=3, filler_after_ms=600.0),
        stt=SimpleNamespace(max_utterance_s=15.0),
    )


def _fake_run_turn_app_state(registry: ObserverRegistry):
    """The subset of `app.state` `_make_run_turn_for_source`'s closure
    reads before its (stubbed, in these tests) call to `run_turn` --
    enough to build every argument the real call site builds, without
    booting the real application or reaching a real provider."""
    from types import SimpleNamespace

    async def _list_macros():
        return []

    async def _list_runs(*, statuses):
        return []

    return SimpleNamespace(
        stt=object(),
        brain=object(),
        tts=object(),
        tool_host_lookup=object(),
        tools_schema=[],
        catalog_prompt="",
        tier_brains=[],
        filler_cache={},
        plugin_manager=SimpleNamespace(owners_of_bare_name=lambda name: ()),
        workflow_repo=SimpleNamespace(list_runs=_list_runs),
        speaker_lock=None,
        workflow_tool_host=None,
        macro_repo=SimpleNamespace(list_macros=_list_macros),
        observer_registry=registry,
        provider_slots=None,
    )


async def test_a_camera_turn_publishes_a_labelled_turn_started_event_and_wraps_the_source(
    tmp_path, monkeypatch
):
    """Exercises the real `_make_run_turn_for_source` closure -- the exact
    production wiring this task adds -- against a stubbed `run_turn` so no
    real provider is ever reached. Proves the turn-start boundary message
    and the wrapping, not `run_turn`'s own behavior (already covered by
    `tests/test_turn_controller.py`)."""
    import spire_voice.app as app_module
    from types import SimpleNamespace

    from tests.conftest import FakeAudioSource

    captured: dict = {}

    async def _fake_run_turn(source, *args, **kwargs):
        captured["source"] = source
        # Simulate a real turn's own emission sequence -- a partial, a
        # reply, then the closing timing event -- each of which must reach
        # the observer feed labelled "camera" (D-08).
        await source.send_event({"type": "transcript.partial", "text": "turn the lights on"})
        await source.send_event({"type": "reply.text", "text": "done"})
        await source.send_event({"type": "turn.timing", "turn_outcome": "completed"})

    monkeypatch.setattr(app_module, "run_turn", _fake_run_turn)

    registry = ObserverRegistry()
    app_stub = SimpleNamespace(state=_fake_run_turn_app_state(registry))
    config = _fake_run_turn_config(tmp_path)
    queue = registry.subscribe()

    run_turn_for_source = app_module._make_run_turn_for_source(
        app_stub, config, app_module.CAMERA_SOURCE_NAME
    )
    await run_turn_for_source(FakeAudioSource())

    started = queue.get_nowait()
    assert started["type"] == "turn.started"
    assert started["source"] == "camera"
    assert isinstance(started["turn_id"], str) and started["turn_id"]
    # `session_id` is the real session directory name (a timestamp plus
    # the turn id), not the bare turn id -- `routes/sessions.py`'s detail
    # route is keyed on the directory name, and this is what makes `/live`'s
    # own "View full session ->" link resolvable.
    assert started["session_id"].endswith(started["turn_id"])
    assert started["session_id"] != started["turn_id"]

    for expected_type in ("transcript.partial", "reply.text", "turn.timing"):
        event = queue.get_nowait()
        assert event["type"] == expected_type
        assert event["source"] == "camera"

    assert isinstance(captured["source"], ObserverPublishingSource)


# --- Route-level tests, through a real TestClient WebSocket ---------------


def _boot_authenticated_client(tmp_path, monkeypatch, role: str):
    from fastapi.testclient import TestClient

    from spire_voice.auth.tokens import issue_access_token
    from spire_voice.config import SecurityConfig
    from spire_voice.plugins import manager as plugin_manager_module

    import spire_voice.app as app_module
    from tests import test_startup_smoke as smoke

    monkeypatch.setenv("SPIRE_SECRET_KEY", smoke._TEST_SECRET_KEY)
    # `debug.dir` -- real `SessionConfig`'s own default is `/data/sessions`,
    # unwritable outside a container; `SessionRecorder`'s real constructor
    # (both turn paths still call it unconditionally, session_recorder=None
    # is never wired) needs somewhere it can actually `mkdir`.
    session_dir = tmp_path / "sessions"
    monkeypatch.setattr(
        app_module,
        "CONFIG_PATH",
        str(smoke._write_fake_config(tmp_path, extra={"debug": {"dir": str(session_dir)}})),
    )
    monkeypatch.setattr(plugin_manager_module, "start_plugin_host", smoke._fake_start_plugin_host)
    monkeypatch.setattr(app_module, "precache_all", smoke._fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", smoke._fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", smoke._fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", smoke._fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", smoke._fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", smoke._fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", smoke._fake_build_ffmpeg_supervisor)

    client = TestClient(app_module.app)
    client.__enter__()
    security = SecurityConfig()
    if role is not None:
        # `current_user` reads the role fresh from `account_repo`, never
        # from the token's own claim (`auth/dependencies.py`'s own
        # docstring) -- `_fake_build_repositories` seeds exactly one user,
        # id=1, role="admin". A non-admin role under test needs its own
        # seeded user, at the id `FakeAccountRepository._next_user_id`
        # already reserves.
        if role == "admin":
            user_id = 1
        else:
            from datetime import datetime, timezone

            from spire_voice.db.repository import User

            account_repo = app_module.app.state.account_repo
            user_id = account_repo._next_user_id
            account_repo._next_user_id += 1
            account_repo.users[user_id] = User(
                id=user_id,
                email=f"{role}@example.invalid",
                display_name=role,
                password_hash="not-a-real-hash-never-checked",
                role=role,
                created_at=datetime.now(timezone.utc),
                disabled_at=None,
            )
        token = issue_access_token(user_id=user_id, role=role, security=security)
        client.cookies.set(security.cookie_name, token)
    return client, app_module


def test_connecting_with_no_session_is_refused(tmp_path, monkeypatch):
    client, _ = _boot_authenticated_client(tmp_path, monkeypatch, role=None)
    try:
        try:
            with client.websocket_connect("/ws/sessions/live"):
                raise AssertionError("an unauthenticated connection was accepted")
        except Exception as exc:  # noqa: BLE001 -- asserting on the denial itself
            assert getattr(exc, "status_code", None) == 401
    finally:
        client.__exit__(None, None, None)


def test_connecting_with_a_viewer_role_session_is_refused(tmp_path, monkeypatch):
    client, _ = _boot_authenticated_client(tmp_path, monkeypatch, role="viewer")
    try:
        try:
            with client.websocket_connect("/ws/sessions/live"):
                raise AssertionError("a viewer-role connection was accepted")
        except Exception as exc:  # noqa: BLE001 -- asserting on the denial itself
            assert getattr(exc, "status_code", None) == 403
    finally:
        client.__exit__(None, None, None)


def test_an_operator_receives_the_opening_message_naming_the_wake_phrase_and_sources(
    tmp_path, monkeypatch
):
    client, app_module = _boot_authenticated_client(tmp_path, monkeypatch, role="operator")
    try:
        with client.websocket_connect("/ws/sessions/live") as websocket:
            opening = websocket.receive_json()
        assert opening["type"] == "observer.opened"
        assert opening["wake_phrase"] == app_module.app.state.config.wake.phrase
        assert set(opening["sources"]) == {"camera", "browser_mic", "browser_webrtc"}
    finally:
        client.__exit__(None, None, None)


def test_a_binary_frame_closes_the_connection_and_starts_no_turn(tmp_path, monkeypatch):
    client, app_module = _boot_authenticated_client(tmp_path, monkeypatch, role="operator")

    async def _fake_run_turn(*args, **kwargs):
        raise AssertionError("the observer socket must never start a turn")

    monkeypatch.setattr(app_module, "run_turn", _fake_run_turn)
    try:
        with client.websocket_connect("/ws/sessions/live") as websocket:
            websocket.receive_json()  # the opening message
            websocket.send_bytes(b"\x00\x01")
            # The server closes on the binary frame -- the next receive
            # observes the close rather than another message.
            with pytest.raises(Exception):
                websocket.receive_json()
    finally:
        client.__exit__(None, None, None)


def test_disconnecting_removes_the_observers_queue_from_the_registry(tmp_path, monkeypatch):
    client, app_module = _boot_authenticated_client(tmp_path, monkeypatch, role="operator")
    try:
        registry = app_module.app.state.observer_registry
        before = len(registry._queues)
        with client.websocket_connect("/ws/sessions/live") as websocket:
            websocket.receive_json()  # the opening message
            assert len(registry._queues) == before + 1
        # The `with` block's own exit performs the close handshake; the
        # server-side handler's `finally` has run by the time it returns.
        assert len(registry._queues) == before
    finally:
        client.__exit__(None, None, None)


def test_two_observers_connected_at_once_both_receive_the_same_publish(tmp_path, monkeypatch):
    client, app_module = _boot_authenticated_client(tmp_path, monkeypatch, role="operator")
    try:
        with client.websocket_connect("/ws/sessions/live") as first:
            first.receive_json()  # opening message
            with client.websocket_connect("/ws/sessions/live") as second:
                second.receive_json()  # opening message

                app_module.app.state.observer_registry.publish(
                    {"type": "transcript.partial", "text": "hello", "source": "camera"}
                )

                assert first.receive_json() == {
                    "type": "transcript.partial",
                    "text": "hello",
                    "source": "camera",
                }
                assert second.receive_json() == {
                    "type": "transcript.partial",
                    "text": "hello",
                    "source": "camera",
                }
    finally:
        client.__exit__(None, None, None)


def test_a_turn_started_from_the_websocket_route_is_labelled_browser_mic_on_the_feed(
    tmp_path, monkeypatch
):
    """The `/ws/turn` half of D-08: a turn on that path appears on the
    observer feed labelled `browser_mic`, and the participant socket still
    gets its own events unchanged (the wrapped source's own `send_event`,
    proven generically by `ObserverPublishingSource`'s own unit tests
    above; this test proves the route wires the label and the publish)."""
    client, app_module = _boot_authenticated_client(tmp_path, monkeypatch, role="operator")

    async def _fake_run_turn(source, *args, **kwargs):
        await source.send_event({"type": "reply.text", "text": "done"})

    monkeypatch.setattr(app_module, "run_turn", _fake_run_turn)
    try:
        with client.websocket_connect("/ws/sessions/live") as observer:
            observer.receive_json()  # opening message
            with client.websocket_connect("/ws/turn") as turn_socket:
                started = observer.receive_json()
                assert started == {
                    "type": "turn.started",
                    "source": "browser_mic",
                    "turn_id": started["turn_id"],
                    "session_id": started["session_id"],
                }
                assert started["session_id"].endswith(started["turn_id"])
                labelled_reply = observer.receive_json()
                assert labelled_reply == {"type": "reply.text", "text": "done", "source": "browser_mic"}
                # The participant socket's own event carries no source key
                # -- unchanged from what `/ws/turn` has always sent.
                assert turn_socket.receive_json() == {"type": "reply.text", "text": "done"}
    finally:
        client.__exit__(None, None, None)


def _receive_json_within(websocket, timeout: float = 10.0) -> dict:
    """`receive_json` with a deadline. `TestClient`'s websocket session
    blocks forever on an empty queue, so a route that publishes nothing --
    exactly the WR-03 defect -- would hang the suite instead of failing it.
    A daemon thread is what makes the wait abandonable: it is left blocked
    on the queue and does not hold the interpreter open at exit."""
    import threading

    received: list[dict] = []
    reader = threading.Thread(target=lambda: received.append(websocket.receive_json()), daemon=True)
    reader.start()
    reader.join(timeout)
    assert received, f"nothing arrived on the observer feed within {timeout}s"
    return received[0]


def test_a_turn_started_from_the_webrtc_route_is_labelled_on_the_feed(tmp_path, monkeypatch):
    """The third turn-starting path (WR-03). `POST /webrtc/offer` schedules a
    real `run_turn` with a real `SessionRecorder` -- the turn is recorded and
    shows up in the Sessions list -- and published nothing at all, so `/live`
    sat on the idle state through the whole turn and a session then appeared
    from nowhere."""
    import spire_voice.app as app_module

    client, _ = _boot_authenticated_client(tmp_path, monkeypatch, role="operator")

    class _FakeWebrtcTransport:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

        async def send_event(self, event: dict[str, Any]) -> None:
            return None

    async def _fake_create_offer_answer(transport, offer):
        return {"sdp": "answer-sdp", "type": "answer"}

    async def _fake_run_turn(source, *args, **kwargs):
        await source.send_event({"type": "reply.text", "text": "done"})

    monkeypatch.setattr(app_module, "WebrtcTransport", _FakeWebrtcTransport)
    monkeypatch.setattr(app_module, "create_offer_answer", _fake_create_offer_answer)
    monkeypatch.setattr(app_module, "run_turn", _fake_run_turn)

    try:
        with client.websocket_connect("/ws/sessions/live") as observer:
            observer.receive_json()  # opening message
            answer = client.post("/webrtc/offer", json={"sdp": "offer-sdp", "type": "offer"})
            assert answer.status_code == 200

            started = _receive_json_within(observer)
            assert started["type"] == "turn.started"
            assert started["source"] == "browser_webrtc"
            assert started["session_id"].endswith(started["turn_id"])

            labelled_reply = _receive_json_within(observer)
            assert labelled_reply == {
                "type": "reply.text",
                "text": "done",
                "source": "browser_webrtc",
            }
    finally:
        client.__exit__(None, None, None)


def test_every_run_turn_call_site_in_app_py_is_wrapped_for_the_observer_feed():
    """Structural, deliberately. WR-03 was a whole turn path that simply did
    not publish, and nothing failed -- the two paths that did publish had
    tests, and the third had none to be missing from. This counts the
    `run_turn` call sites against the sources wrapped for the feed and the
    `turn.started` events published, so a fourth path added later cannot go
    missing the same way: it fails here the moment it is written."""
    import ast
    from pathlib import Path

    import spire_voice.app as app_module

    tree = ast.parse(Path(app_module.__file__).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    run_turn_calls = [node for node in calls if node.func.id == "run_turn"]
    wrapped_sources = [node for node in calls if node.func.id == "ObserverPublishingSource"]
    turn_started_events = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == "turn.started"
    ]

    assert len(run_turn_calls) == 3, (
        "a turn-starting path was added or removed; update "
        "app.py's OBSERVED_SOURCE_NAMES block and this count together"
    )
    assert len(wrapped_sources) == len(run_turn_calls)
    assert len(turn_started_events) == len(run_turn_calls)
    # Every one of them runs the turn against the wrapped source, never the
    # raw transport -- the exact substitution WR-03 found missing.
    assert [node.args[0].id for node in run_turn_calls] == ["source"] * len(run_turn_calls)
    assert set(app_module.OBSERVED_SOURCE_NAMES) == {
        "camera",
        "browser_mic",
        "browser_webrtc",
    }


def test_a_text_frame_alongside_a_null_bytes_key_is_discarded_not_closed(tmp_path, monkeypatch):
    """IN-05: the binary check keyed on the presence of `bytes`, not on a
    non-`None` value. An ASGI server that emits both keys with `bytes:
    None` would have closed every observer on its first text frame. Driven
    against the handler's own receive, not through the installed server,
    because the installed server is precisely what hides this."""
    import spire_voice.app as app_module

    client, _ = _boot_authenticated_client(tmp_path, monkeypatch, role="operator")
    try:
        with client.websocket_connect("/ws/sessions/live") as observer:
            observer.receive_json()  # opening message
            # The raw ASGI message, both keys present -- what the
            # installed uvicorn never produces and another server may.
            observer.send(
                {
                    "type": "websocket.receive",
                    "text": "a text frame an observer may send and have ignored",
                    "bytes": None,
                }
            )
            # Still open, and still receiving: a published event arrives.
            app_module.app.state.observer_registry.publish({"type": "reply.text", "text": "still here"})
            assert _receive_json_within(observer) == {"type": "reply.text", "text": "still here"}
    finally:
        client.__exit__(None, None, None)
