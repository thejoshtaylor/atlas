"""The locked rule this whole phase stands on (T-09-49): ATLAS never sends
email. `test_nothing_can_send` is what fails the suite the moment that
stops being true -- no file under `src/atlas/` or `mcp/atlas_mcp/` may
spell either Gmail send endpoint, even in a comment or a docstring;
`REQUIRED_SCOPES` may carry no send-only scope; and a real draft-creation
call, run against `FakeGoogle`, records no request that ends in a send
segment.

Both forbidden endpoint patterns are built from string pieces at run time
(a collection name, a slash, the method name) -- this file's own source
text never spells either one, so a naive grep for the literal string
could not even find it here, and each pattern carries a negative
lookahead for a following letter so the send-as settings path
`list_send_as` uses (`.../settings/sendAs`) never matches either.
"""

from __future__ import annotations

import re
from pathlib import Path

from google_fakes import FakeGoogle

from atlas_mcp.google import handle_gmail_create_draft
from atlas_mcp.google_boundary import AccountGrant
from atlas_mcp.google_tools import REQUIRED_SCOPES

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCAN_ROOTS = (_REPO_ROOT / "src" / "atlas", _REPO_ROOT / "mcp" / "atlas_mcp")

# Built from pieces, on purpose -- see the module docstring above.
_SLASH = "/"
_SEND_METHOD = "".join(("s", "e", "n", "d"))
_MESSAGES_COLLECTION = "messages"
_DRAFTS_COLLECTION = "drafts"
_FORBIDDEN_PATTERNS = tuple(
    re.compile(re.escape(collection + _SLASH + _SEND_METHOD) + r"(?![A-Za-z])")
    for collection in (_MESSAGES_COLLECTION, _DRAFTS_COLLECTION)
)

# A send-capable scope, also spelled from pieces so it cannot be mistaken
# for the endpoint-path check above, kept as a set of one scope string
# this project must never request.
_SEND_SCOPE = "https://www.googleapis.com/auth/gmail." + _SEND_METHOD


def _iter_py_files():
    for root in _SCAN_ROOTS:
        yield from root.rglob("*.py")


async def test_nothing_can_send():
    # 1. No source file spells either forbidden endpoint path, anywhere --
    #    comments and docstrings included, since a human reading the code
    #    is exactly who this guards against a future well-meaning "just
    #    add send while we're in here" edit.
    violations = []
    for path in _iter_py_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern.search(text):
                violations.append(f"{path.relative_to(_REPO_ROOT)}: {pattern.pattern}")
    assert not violations, f"a Gmail send endpoint is spelled in source: {violations}"

    # 2. The one OAuth consent this whole phase requests carries no
    #    send-capable scope.
    assert _SEND_SCOPE not in REQUIRED_SCOPES

    # 3. A real draft-creation call, against a fake transport that records
    #    every request it ever answers, never reaches a send-shaped URL.
    fake = FakeGoogle()
    fake.add_gmail_metadata(
        "at-1",
        "m1",
        headers={"From": "Dana <dana@example.com>", "Subject": "Hi", "Message-Id": "<o1@example.com>"},
    )
    accounts = (
        AccountGrant(
            label="work", email="work@example.com", is_default=True,
            access_token="at-1", unreachable_reason=None, calendars=(),
        ),
    )

    await handle_gmail_create_draft(
        accounts, fake.client, account="work", message_id="m1", body_text="a reply, never sent."
    )

    assert fake.drafts, "the draft-creation call above did not actually reach the fake"
    for request in fake.requests:
        url = str(request.url)
        for pattern in _FORBIDDEN_PATTERNS:
            assert not pattern.search(url), f"a request reached a forbidden endpoint: {url}"
