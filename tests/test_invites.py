"""An admin invites another person at a role, and only at that role
(WEB-05).

The `invites` table holds a token, an expiry, and the role being granted.
Three failure modes this file exists to catch, each a different way an
invite could grant more access than the admin who created it intended: a
role other than the one the invite named (a `viewer` invite that somehow
creates an `operator` account), an expired token still working (a stale
invite link, found in an old chat or email, silently still valid), and a
token usable more than once (one invite link creating two accounts, one of
them nobody the admin meant to invite).
"""

from __future__ import annotations


def test_an_invite_grants_exactly_the_role_it_names():
    """Accepting an invite created with a given role must create an account
    at exactly that role, never a different one -- plan 03-05 fills this
    in."""
    raise AssertionError(
        "plan 03-05 fills this in (WEB-05: an invite grants exactly the role it names)"
    )


def test_an_expired_invite_is_refused():
    """An invite past its expiry must be refused, not silently honored --
    plan 03-05 fills this in."""
    raise AssertionError("plan 03-05 fills this in (WEB-05: an expired invite is refused)")


def test_an_invite_cannot_be_accepted_twice():
    """A second attempt to accept an already-accepted invite token must be
    refused -- plan 03-05 fills this in."""
    raise AssertionError(
        "plan 03-05 fills this in (WEB-05: an invite is single-use)"
    )
