"""Five enrollment phrases read into the real edge microphone (D-02).

Claude's Discretion (11-CONTEXT.md): plain sentences, about 2 to 3 seconds
each read aloud, varied sounds, no wake phrase, and no house data. The
Pi-side enrollment corpus tool (plan 11-05) restates this same list; a
contract test pins the two together so they never drift apart.
"""

from __future__ import annotations

ENROLLMENT_PHRASES: "tuple[str, ...]" = (
    "The morning light comes through the kitchen window.",
    "A quiet street holds more sound than it first seems.",
    "Every good recipe starts with a clean counter.",
    "The garden needs water at least twice this week.",
    "Reading a little before bed makes the house feel calm.",
)
