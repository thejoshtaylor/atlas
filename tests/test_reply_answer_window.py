"""261001-dlp: a question the model asks opens a window that can answer it.

On the edge source, a turn that dispatched no tool and ended on a model
question ("what do you want me to turn on?") leaves an answer window scoped to
the full catalog that turn offered. The operator then says "the lamp" with no
wake word, and the window turn can call the lamp tool. Every other case keeps
the Phase 13 scope. Every name and entity id here is invented.
"""

from __future__ import annotations

import json

from atlas.providers.base import BrainReply
from atlas.turn.follow_up import AnswerScope

from brain_fakes import RecordingFakeBrain
from test_follow_up_answer_window import _LIGHTS, _SCHEMA, _Edge, _text, _tool_names

_FULL = frozenset({"weather_now", "lights_set", "gmail_fetch_body"})


def _json(answer: str, expects_reply: bool) -> BrainReply:
    return BrainReply(text=json.dumps({"answer": answer, "expects_reply": expects_reply}))


async def test_a_no_tool_question_opens_a_window_that_reaches_the_full_offered_catalog():
    brain = RecordingFakeBrain(
        replies=[
            _json("what do you want me to turn on?", True),
            BrainReply(tool_calls=[_LIGHTS]),
            BrainReply(text="Done."),
        ]
    )
    edge = _Edge([_text("turn on the"), _text("the lamp"), _text("")], brain)

    await edge.run()

    first = edge.requests[0]
    assert first is not None and first.kind == "answer"
    assert first.answer_scope == AnswerScope(tool_names=_FULL, entity_ids=None)
    assert edge.tts[0].received_text[-1] == "what do you want me to turn on?"
    # The window turn was offered all three tools, and its lamp call reached the host.
    assert _tool_names(brain.calls[1]) == _FULL
    assert [name for name, _ in edge.tool_host.calls] == ["lights_set"]
    assert [incoming.kind if incoming else None for incoming in edge.incomings] == [None, "answer", "answer"]
