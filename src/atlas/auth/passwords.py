"""Password hashing: a thin wrapper over `pwdlib`'s recommended Argon2
hasher, and nothing more (D-05).

The wrapper exists so the rest of the codebase imports one name
(`hash_password`/`verify_password`) and a future parameter change (a
different hasher, a cost-parameter bump) lands in this one file -- not so
the library is abstracted away. There is no home-grown hashing scheme here
and there is not meant to be one.
"""

from __future__ import annotations

from pwdlib import PasswordHash
from pwdlib.exceptions import PwdlibError

# `PasswordHash.recommended()` is pwdlib's own currently-recommended
# hasher (Argon2, per its own docstring) -- not a parameter set this module
# chooses by hand. One instance, module-level: hasher construction reads no
# environment and has no per-call state to keep separate.
_password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """Hash `password` for storage -- `UserRow.password_hash`'s only writer."""
    return _password_hash.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """True when `password` matches `password_hash`.

    `pwdlib`'s own `verify` returns `False` for a well-formed hash and a
    wrong password, but raises `PwdlibError` for a hash it cannot even
    identify (corrupt data, or a hash from a scheme this process never
    wrote) -- caught here and treated the same as a wrong password, so a
    corrupt stored hash is a refused sign-in, never an unhandled 500 that
    tells an unauthenticated caller something about the account's internal
    state.
    """
    try:
        return _password_hash.verify(password, password_hash)
    except PwdlibError:
        return False
