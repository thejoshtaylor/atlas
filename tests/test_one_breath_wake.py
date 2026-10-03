"""A wake phrase and a command in one breath, on a provider that gives no
word before it is finalized (debug session wake-command-needs-beep-wait).

On the Pi, "Hey Atlas, what time is it" said without a pause is one VAD
segment, and it is already in progress at the wake hit. `ParakeetStt` gives
no word until it is finalized, so the end-of-speech watch saw no word and no
segment that began inside the drain. The turn waited for new speech or for
`max_utterance_s` (15 s) and then ended `wake_unverified` with no decode:
the chime played, then nothing.

These tests drive `run_turn` with the real `ParakeetStt` over a fake
recognizer and the real `SpeechSignals`. The source marks each frame with
the part of the utterance it carries, so the fake recognizer answers from
the audio it was given, not from call order.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np
import pytest

from atlas.config import PARAKEET_BPE_VOCAB, PARAKEET_MODEL_FILES, SttConfig
from atlas.providers.base import BrainReply
from atlas.providers.stt_faster_whisper import FasterWhisperStt
from atlas.providers.stt_parakeet import ParakeetStt
from atlas.providers.stt_xai import XaiStt
from atlas.transports.base import SourceFormat
from atlas.transports.edge import SpeechSignals

# Sample values the source writes, one per part of the utterance.
_SILENCE = 0
_WAKE = 100
_COMMAND = 200
_MORE = 300


class _MarkedSource:
    """A live mic double. Every 10 ms frame carries the current mark, and
    `frames()` never ends on its own."""

    def __init__(self, signals: SpeechSignals, mark: int = _SILENCE) -> None:
        self.speech_signals = signals
        self.mark = mark
        self.events: list[dict] = []

    async def frames(self) -> AsyncIterator[bytes]:
        while True:
            yield np.full(160, self.mark, dtype="<i2").tobytes()
            await asyncio.sleep(0.01)

    async def send_audio(self, chunk: bytes) -> None:
        pass

    async def send_event(self, event: dict) -> None:
        self.events.append(event)

    def source_format(self) -> SourceFormat:
        return SourceFormat("pcm", 16000)

    def final_text(self) -> str:
        return [e["text"] for e in self.events if e["type"] == "transcript.final"][0]


class _Result:
    def __init__(self, text: str) -> None:
        self.text = text


class _Stream:
    def __init__(self) -> None:
        self.audio: "np.ndarray | None" = None
        self.result = _Result("")

    def accept_waveform(self, sample_rate, audio) -> None:
        self.audio = audio


class _MarkReadingRecognizer:
    """Answers with the text for the highest mark in the audio it decodes.
    `decode_s` makes each decode take that long, like the real model."""

    def __init__(self, texts: "dict[int, str]", *, decode_s: float = 0.0) -> None:
        self._texts = texts
        self._decode_s = decode_s
        self.decoded: list[str] = []

    def create_stream(self) -> _Stream:
        return _Stream()

    def decode_stream(self, stream: _Stream) -> None:
        if self._decode_s:
            time.sleep(self._decode_s)
        audio = stream.audio if stream.audio is not None else np.zeros(0, dtype=np.float32)
        top = round(float(np.max(audio)) * 32768) if audio.size else _SILENCE
        text = self._texts.get(top, "")
        self.decoded.append(text)
        stream.result = _Result(text)


def _parakeet(tmp_path, recognizer: _MarkReadingRecognizer) -> ParakeetStt:
    model_dir = tmp_path / "parakeet"
    model_dir.mkdir()
    for name in PARAKEET_MODEL_FILES:
        (model_dir / name).write_bytes(b"x")
    (model_dir / PARAKEET_BPE_VOCAB).write_text("a\t0\n", encoding="utf-8")
    return ParakeetStt(SttConfig(parakeet_model_dir=str(model_dir)), load_recognizer=lambda spec: recognizer)


def _segment_in_progress() -> SpeechSignals:
    """The wake segment is still in progress at the wake hit."""
    signals = SpeechSignals(hangover_s=0.0)
    signals.publish({"type": "vad.start", "seq": 1})
    return signals


async def _run_wake_turn(source, stt, brain, tts, timings, *, timeout_s: float = 3.0, **kwargs):
    from atlas.turn.controller import run_turn

    await asyncio.wait_for(
        run_turn(
            source,
            stt,
            brain,
            tts,
            None,
            tools_schema=[],
            system_prompt="you control a home",
            max_tool_rounds=3,
            timings=timings,
            wake_phrase="hey atlas",
            verify_wake=True,
            **kwargs,
        ),
        timeout=timeout_s,
    )


async def test_a_wake_phrase_and_command_in_one_breath_finalizes_at_the_segment_end(
    tmp_path, fake_brain, fake_tts
):
    """The reported bug: one segment holds the wake phrase and the command,
    and no other speech follows. The turn must reach the brain at that
    segment's end, not wait for new speech or for `max_utterance_s`."""
    from atlas.timing import TurnTimings

    signals = _segment_in_progress()
    source = _MarkedSource(signals, mark=_COMMAND)
    recognizer = _MarkReadingRecognizer({_COMMAND: "Hey Atlas, what time is it?"})
    brain = fake_brain(replies=[BrainReply(text="noon")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    async def _end_of_speech() -> None:
        await asyncio.sleep(0.15)
        signals.publish({"type": "vad.end", "seq": 1})

    asyncio.ensure_future(_end_of_speech())
    await _run_wake_turn(source, _parakeet(tmp_path, recognizer), brain, tts, timings, timeout_s=2)

    assert source.final_text() == "what time is it?"
    assert brain.call_count == 1
    assert tts.received_text == ["noon"]
    assert timings.vad_end_at is not None
    assert {"type": "wake.confirmed"} in source.events


@pytest.mark.parametrize("wake_text", ["Hey Atlas.", "Yeah.", ""])
async def test_waiting_for_the_chime_still_transcribes_the_whole_turn(tmp_path, fake_brain, fake_tts, wake_text):
    """Wait mode: the wake segment ends, the operator waits for the chime,
    then speaks the command in a new segment. Parakeet hears "atlas" best
    with the whole turn as context (a wake segment alone decoded "Yeah." or
    "" on real recordings), so the command's end must still decode from the
    turn's first frame, whatever the wake segment alone decodes to."""
    from atlas.timing import TurnTimings

    signals = _segment_in_progress()
    source = _MarkedSource(signals, mark=_WAKE)
    recognizer = _MarkReadingRecognizer({_WAKE: wake_text, _COMMAND: "Hey Atlas, what time is it?"})
    brain = fake_brain(replies=[BrainReply(text="noon")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    async def _segments() -> None:
        await asyncio.sleep(0.1)
        signals.publish({"type": "vad.end", "seq": 1})
        source.mark = _SILENCE
        await asyncio.sleep(0.3)  # the chime, and the operator's pause
        source.mark = _COMMAND
        signals.publish({"type": "vad.start", "seq": 2})
        await asyncio.sleep(0.15)
        signals.publish({"type": "vad.end", "seq": 2})

    asyncio.ensure_future(_segments())
    await _run_wake_turn(source, _parakeet(tmp_path, recognizer), brain, tts, timings)

    assert source.final_text() == "what time is it?"
    assert brain.call_count == 1
    # The early transcription ran on the wake segment and was not used; the
    # final decode covers the whole turn.
    assert recognizer.decoded == [wake_text, "Hey Atlas, what time is it?"]


async def test_an_unfinished_command_in_the_wake_segment_waits_for_the_rest(tmp_path, fake_brain, fake_tts):
    """"Hey Atlas, turn off." and a pause: the rest of the command comes in
    the next segment, so the early transcription must not end the turn."""
    from atlas.timing import TurnTimings

    signals = _segment_in_progress()
    source = _MarkedSource(signals, mark=_COMMAND)
    recognizer = _MarkReadingRecognizer(
        {_COMMAND: "Hey Atlas, turn off.", _MORE: "Hey Atlas, turn off the radio."}
    )
    brain = fake_brain(replies=[BrainReply(text="done")])
    timings = TurnTimings()

    async def _segments() -> None:
        await asyncio.sleep(0.1)
        signals.publish({"type": "vad.end", "seq": 1})
        await asyncio.sleep(0.3)
        source.mark = _MORE
        signals.publish({"type": "vad.start", "seq": 2})
        await asyncio.sleep(0.15)
        signals.publish({"type": "vad.end", "seq": 2})

    asyncio.ensure_future(_segments())
    await _run_wake_turn(source, _parakeet(tmp_path, recognizer), brain, fake_tts(chunks=[b"\x01\x02"]), timings)

    assert source.final_text() == "turn off the radio."
    assert recognizer.decoded == ["Hey Atlas, turn off.", "Hey Atlas, turn off the radio."]


async def test_new_speech_during_the_early_transcription_wins(tmp_path, fake_brain, fake_tts):
    """The operator speaks again while the drain transcribes the wake segment.
    A complete early text must not cut off the new segment."""
    from atlas.timing import TurnTimings

    signals = _segment_in_progress()
    source = _MarkedSource(signals, mark=_COMMAND)
    recognizer = _MarkReadingRecognizer(
        {_COMMAND: "Hey Atlas, what time is it?", _MORE: "Hey Atlas, what time is it in Paris?"},
        decode_s=0.15,
    )
    brain = fake_brain(replies=[BrainReply(text="noon")])
    timings = TurnTimings()

    async def _segments() -> None:
        await asyncio.sleep(0.1)
        signals.publish({"type": "vad.end", "seq": 1})
        await asyncio.sleep(0.03)  # inside the early transcription
        source.mark = _MORE
        signals.publish({"type": "vad.start", "seq": 2})
        await asyncio.sleep(0.3)
        signals.publish({"type": "vad.end", "seq": 2})

    asyncio.ensure_future(_segments())
    await _run_wake_turn(source, _parakeet(tmp_path, recognizer), brain, fake_tts(chunks=[b"\x01\x02"]), timings)

    assert source.final_text() == "what time is it in Paris?"


async def test_an_unverified_early_transcription_does_not_end_the_turn(tmp_path, fake_brain, fake_tts):
    """A television false fire: the wake segment's text does not open with
    the wake phrase. The early transcription changes nothing; the turn ends
    the way it did before, and the brain is never called."""
    from atlas.timing import TurnTimings

    signals = _segment_in_progress()
    source = _MarkedSource(signals, mark=_COMMAND)
    recognizer = _MarkReadingRecognizer({_COMMAND: "It is very nice in the spring."})
    brain = fake_brain(replies=[])
    timings = TurnTimings()

    async def _end_of_speech() -> None:
        await asyncio.sleep(0.1)
        signals.publish({"type": "vad.end", "seq": 1})

    asyncio.ensure_future(_end_of_speech())
    started = time.monotonic()
    await _run_wake_turn(
        source,
        _parakeet(tmp_path, recognizer),
        brain,
        fake_tts(chunks=[]),
        timings,
        max_utterance_s=0.8,
    )

    assert timings.turn_outcome == "wake_unverified"
    assert brain.call_count == 0
    assert time.monotonic() - started >= 0.8


def test_the_whole_utterance_providers_say_they_give_no_word_before_finalize():
    assert ParakeetStt.words_before_finalize is False
    assert FasterWhisperStt.words_before_finalize is False
    assert getattr(XaiStt, "words_before_finalize", True) is True
