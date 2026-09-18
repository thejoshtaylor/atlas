"""Provider credentials are encrypted at rest and write-only from the
browser (PROV-04).

CONTEXT.md is explicit: after a credential is saved, the webapp shows a
mask and never receives the plaintext back. A route that returns a stored
credential's value -- even to an authenticated admin, even for a
"show/hide" convenience toggle -- defeats the reason this data is encrypted
at all: the whole point of write-only is that a compromised admin session,
or a bug in a response serializer that includes one field too many, cannot
leak the plaintext back out over the same channel that would leak the leak.
The second test guards the storage layer directly: a row read straight out
of the database, with no `SPIRE_SECRET_KEY`, must not be interpretable as
the original credential -- proving the encryption is real, not a
base64-shaped no-op.
"""

from __future__ import annotations


def test_a_saved_credential_never_comes_back_from_any_route():
    """No route -- not even a GET on the credential's own resource -- may
    ever return the plaintext value after it has been saved once -- plan
    03-07 fills this in."""
    raise AssertionError(
        "plan 03-07 fills this in (PROV-04: credentials are write-only from the browser)"
    )


def test_a_stored_credential_is_unreadable_without_the_key():
    """A credential's stored ciphertext, read directly from the database
    with no `SPIRE_SECRET_KEY` available, must not decode to the original
    plaintext -- plan 03-07 fills this in."""
    raise AssertionError(
        "plan 03-07 fills this in (PROV-04: a stored credential is encrypted, not merely encoded)"
    )
