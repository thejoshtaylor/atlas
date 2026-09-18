"""There is no default password at any point (WEB-01, D-08).

While the `users` table is empty, only the create-admin route may answer --
every other route must return 503 naming "setup incomplete". Once an admin
exists, the create-admin route must be permanently closed. The failure mode
this file exists to catch is not "the wizard looks broken"; it is "a
half-installed spire-voice is a reachable spire-voice" -- an admin panel
that quietly serves its ordinary routes before an operator has ever set a
password would let a television-shaped sentence, or a stranger on the same
network before setup finishes, reach a route this project's whole safety
posture assumes only an authenticated operator can reach.
"""

from __future__ import annotations


def test_create_admin_answers_only_while_no_user_exists():
    """`POST /auth/create-admin` must succeed while `users` is empty, and
    must refuse (the route "permanently closed", D-08) once an admin
    exists -- plan 03-05 fills this in."""
    raise AssertionError(
        "plan 03-05 fills this in (WEB-01, D-08: create-admin closes forever after the first admin)"
    )


def test_every_other_route_reports_setup_incomplete_until_an_admin_exists():
    """Every route other than create-admin (and a health check) must answer
    503 naming "setup incomplete" while `users` is empty -- plan 03-05
    fills this in."""
    raise AssertionError(
        "plan 03-05 fills this in (WEB-01, D-08: a half-installed system is never a reachable one)"
    )
