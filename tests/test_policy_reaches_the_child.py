"""The policy the operator writes must reach the process that enforces it.

`mcp/atlas_mcp/ha.py` is the only process that calls Home Assistant, so its
`Policy` is the one that decides. It once hardcoded `Policy.from_config(None)`
and never saw the `safety:` block at all: `config.py` parsed the block into
`Config.policy`, and nothing read that field. A denylist written by an
operator was therefore silently inert -- the config said one thing and the
enforcing process did another.

No test caught it, because every other safety test constructs a `Policy`
directly and hands it to the function under test. That proves `allow_call`
works; it proves nothing about where the running child gets its policy. These
tests assert on the wiring instead.

CONTEXT.md D-13 locked config-loading as Phase 1's job; only the
database-backed policy was deferred to Phase 3.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _child_policy_probe(env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    """Import the child module with a given env and print its resolved policy."""
    code = textwrap.dedent(
        """
        import json
        import atlas_mcp.ha as ha
        p = ha._policy
        print(json.dumps({
            "mode": p.mode,
            "deny_entities": sorted(p.deny_entities),
            "deny_patterns": sorted(p.deny_patterns),
        }))
        """
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": f"{REPO}/src:{REPO}/mcp",
        "HA_URL": "http://ha.invalid:8123",
        "HA_TOKEN": "not-a-real-token",
        **env_extra,
    }
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60
    )


def test_child_applies_the_safety_block_it_is_given():
    """A denylist in the config reaches the child and is actually in its policy."""
    block = {
        "mode": "allow_all_except_denylist",
        "deny_entities": ["switch.example_server_socket"],
        "deny_patterns": ["switch.example_camera_*"],
    }
    proc = _child_policy_probe({"ATLAS_SAFETY": json.dumps(block)})
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "switch.example_server_socket" in got["deny_entities"], (
        "the operator's denylist did not reach the process that enforces it"
    )
    assert "switch.example_camera_*" in got["deny_patterns"]


def test_child_without_a_block_falls_back_to_compiled_defaults():
    """No `safety:` block is a real choice: defaults, and no entity rules."""
    proc = _child_policy_probe({})
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["deny_entities"] == []
    assert got["deny_patterns"] == []
    assert got["mode"] == "allow_all_except_denylist"


def test_child_refuses_to_start_on_a_malformed_block():
    """Malformed must fail loudly, never degrade into "no entity rules".

    That degradation is the silent failure this whole module exists to close:
    a denylist that looks configured and enforces nothing.
    """
    proc = _child_policy_probe({"ATLAS_SAFETY": "{not json"})
    assert proc.returncode != 0, "a malformed policy must not start"
    assert "ATLAS_SAFETY" in proc.stderr

    proc = _child_policy_probe({"ATLAS_SAFETY": json.dumps(["not", "a", "mapping"])})
    assert proc.returncode != 0, "a non-mapping policy must not start"
    assert "ATLAS_SAFETY" in proc.stderr


def test_app_forwards_the_raw_block_rather_than_reserializing_policy():
    """One parser, `safety.py`, on both sides of the process boundary.

    Phase 3 (plan 03-02) moves the source: `lifespan` no longer forwards the
    unparsed block straight off `Config` (that field and the `safety:`
    config key are both retired, D-11) -- it now derives the block from the
    database, through `safety_block_from_policy(await
    policy_repo.load_policy())`. The intent this test protects is
    unchanged: `lifespan` builds the JSON-shaped block `Policy.from_config`
    expects, from whatever the repository returns, and never constructs a
    `Policy` object here and serializes that -- one parser, `safety.py`, on
    both sides of the process boundary, same as before this plan.
    """
    import inspect

    import atlas.app as app_mod

    source = inspect.getsource(app_mod.lifespan)
    assert "safety_block_from_policy(await policy_repo.load_policy())" in source, (
        "the block handed to the tool host must be derived from the policy repository, "
        "through safety_block_from_policy(await policy_repo.load_policy())"
    )
