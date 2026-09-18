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

Every test here boots the real application (reusing
`tests/test_auth_setup.py`'s `_boot_with_empty_accounts` helper) and drives
the actual `/api/auth/create-admin` -> `/api/invites` -> `/api/invites/
{token}/accept` flow over real HTTP -- the failure mode this file guards
against is specifically about the wired-together routes, not a handler
called in isolation.

One more test below is a named Task 3 acceptance criterion, not a
pre-written scaffold: an invite token must appear in exactly one response
body, ever -- the create response, and never again from any later route
that reads the same invite back.
"""

from __future__ import annotations

from test_auth_setup import _boot_with_empty_accounts


def _create_admin(client) -> None:
    response = client.post(
        "/api/auth/create-admin",
        json={
            "email": "admin@example.invalid",
            "display_name": "The Admin",
            "password": "a-plainly-fictional-test-password",
        },
    )
    assert response.status_code == 201, response.text


def test_an_invite_grants_exactly_the_role_it_names(tmp_path, monkeypatch):
    """Accepting an invite created with a given role must create an
    account at exactly that role, never a different one -- even when the
    acceptance body includes an unsolicited `role` field asking for a
    different one (`AcceptInviteRequest` has no `role` field at all, so a
    client-supplied one is silently ignored, never read)."""
    with _boot_with_empty_accounts(tmp_path, monkeypatch) as client:
        _create_admin(client)

        create_response = client.post(
            "/api/invites",
            json={"role": "operator", "email": "invitee@example.invalid"},
        )
        assert create_response.status_code == 201, create_response.text
        invite_body = create_response.json()
        token = invite_body["token"]
        assert invite_body["role"] == "operator"

        accept_response = client.post(
            f"/api/invites/{token}/accept",
            json={
                "display_name": "The Invitee",
                "password": "another-plainly-fictional-password",
                "role": "admin",  # an unsolicited, unread field
            },
        )
        assert accept_response.status_code == 201, accept_response.text
        accepted_body = accept_response.json()
        assert accepted_body["role"] == "operator", (
            "the invite named 'operator', but the created account's role is "
            f"{accepted_body['role']!r} -- a request-body role claim must never win"
        )
        assert accepted_body["email"] == "invitee@example.invalid"


def test_an_expired_invite_is_refused(tmp_path, monkeypatch):
    """An invite past its expiry must be refused, not silently honored."""
    with _boot_with_empty_accounts(tmp_path, monkeypatch) as client:
        _create_admin(client)

        create_response = client.post(
            "/api/invites",
            # Negative TTL: expires_at is already in the past the instant
            # this invite is created -- no timing race to get wrong.
            json={"role": "viewer", "email": "late@example.invalid", "ttl_s": -1},
        )
        assert create_response.status_code == 201, create_response.text
        token = create_response.json()["token"]

        accept_response = client.post(
            f"/api/invites/{token}/accept",
            json={"display_name": "Too Late", "password": "yet-another-fictional-password"},
        )
        assert accept_response.status_code == 400, accept_response.text
        assert "invalid" in accept_response.json()["detail"].lower()


def test_an_invite_cannot_be_accepted_twice(tmp_path, monkeypatch):
    """A second attempt to accept an already-accepted invite token must be
    refused."""
    with _boot_with_empty_accounts(tmp_path, monkeypatch) as client:
        _create_admin(client)

        create_response = client.post(
            "/api/invites",
            json={"role": "viewer", "email": "twice@example.invalid"},
        )
        assert create_response.status_code == 201, create_response.text
        token = create_response.json()["token"]

        first_accept = client.post(
            f"/api/invites/{token}/accept",
            json={"display_name": "First Accept", "password": "a-fictional-password-one"},
        )
        assert first_accept.status_code == 201, first_accept.text

        second_accept = client.post(
            f"/api/invites/{token}/accept",
            json={"display_name": "Second Accept", "password": "a-fictional-password-two"},
        )
        assert second_accept.status_code == 400, second_accept.text
        assert "invalid" in second_accept.json()["detail"].lower()


def test_an_invite_token_appears_in_exactly_one_response_body_ever(tmp_path, monkeypatch):
    """An invite's plaintext token is returned exactly once, in the create
    response -- listing invites must never return the token or its hash,
    asserted against the serialized body, not merely against the response
    model's declared fields."""
    with _boot_with_empty_accounts(tmp_path, monkeypatch) as client:
        _create_admin(client)

        create_response = client.post(
            "/api/invites",
            json={"role": "operator", "email": "token-once@example.invalid"},
        )
        assert create_response.status_code == 201, create_response.text
        token = create_response.json()["token"]
        assert len(token) > 16, "the plaintext token must be a real, non-trivial bearer credential"

        list_response = client.get("/api/invites")
        assert list_response.status_code == 200, list_response.text
        assert token not in list_response.text, (
            "the plaintext invite token must never reappear in the invite-listing response"
        )
        for invite in list_response.json():
            assert "token" not in invite, f"invite listing entry carries a token field: {invite!r}"
            assert "token_hash" not in invite, f"invite listing entry carries a token_hash field: {invite!r}"
