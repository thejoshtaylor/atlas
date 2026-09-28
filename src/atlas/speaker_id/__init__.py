"""Who spoke this turn on the edge microphone (Phase 11, D-01 through D-16).

This is `atlas.speaker_id`, not `atlas.speaker` -- that package plays audio
*out* (the FIFO, the ffmpeg supervisor, Tapo talk). The two share a similar
name and nothing else; importing one expecting the other is a bug.

A speaker label this package produces is untrusted, exactly like
transcribed text (`.claude/CLAUDE.md`'s "treat all transcribed text as
untrusted input" constraint, extended here to speaker identity by D-15). It
is for personalization only. It never authorizes anything: it never reaches
`mcp.atlas_mcp.safety.allow_call`, the denylist, or a confirm window (Phase
9 D-08, D-24).
"""

from __future__ import annotations
