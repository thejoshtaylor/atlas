"""The route enforces a role, not the user interface (WEB-04).

CONTEXT.md is explicit that a hidden button is not access control: a
`require_role(...)` FastAPI dependency must sit on each route that needs
one, checked server-side, every request. The failure mode this file exists
to catch is a policy or account route that is reachable by any authenticated
user regardless of role -- a viewer editing the denylist, say -- because the
webapp merely hid the control rather than the backend refusing the call.
The second test guards a narrower, easier mistake: a role read from
somewhere the caller controls (a request body field, a header) rather than
from the signed cookie the server itself issued, which would let any
authenticated request claim whatever role it wants.
"""

from __future__ import annotations


def test_a_viewer_cannot_reach_an_operator_route():
    """A `viewer`-role account calling a route gated
    `require_role(Role.OPERATOR)` must get a 403, not a hidden-but-reachable
    success -- plan 03-05 fills this in."""
    raise AssertionError(
        "plan 03-05 fills this in (WEB-04: the route enforces the role, not the UI)"
    )


def test_a_role_is_read_from_the_cookie_not_the_request_body():
    """A request body or header claiming a higher role than the caller's
    actual signed cookie carries must not elevate what `require_role`
    grants -- plan 03-05 fills this in."""
    raise AssertionError(
        "plan 03-05 fills this in (WEB-04: role comes from the signed cookie only)"
    )
