"""JWT access tokens, opaque refresh tokens, and the session cookie pair
(D-05, T-03-26, T-03-27, T-03-28).

**The session model (Task 1's checkpoint, confirmed by the operator --
recorded verbatim in 03-05-SUMMARY.md):** an HttpOnly, `SameSite=Lax`
cookie holds a short-lived JWT (the access token); a second, sibling
HttpOnly/`SameSite=Lax` cookie holds an opaque refresh token whose hash is
stored server-side in `refresh_tokens` and rotated on every use. The
alternative considered -- a single opaque session cookie looked up
server-side on every request -- was rejected because it puts a database
read on every request, including the turn surfaces this project's latency
budget is already over on. A separate environment variable for the JWT
signing key, instead of the HKDF derivation below, was also considered and
rejected: see the derivation note just below for why.

**HKDF derivation, not a second environment variable (CD-3):** the JWT
signing key here and the credential-encryption key plan 03-07 uses are two
independent values derived from the one `ATLAS_SECRET_KEY` the operator
supplies, via HKDF-SHA256 under two distinct, fixed `info` labels. The
deciding reason: a deployment with only one of two required secrets set is
a deployment that half-works, and it half-works *silently* -- this
project's whole deployment story (DEP-03) is a stranger, following a
README, on a machine they have not used before, reaching a running
assistant with no file editing. One secret to generate and not lose is one
failure mode to get right, not two. HKDF's own `info` parameter is exactly
what the standard provides for deriving multiple independent keys from one
input key material without the two ever colliding or reusing randomness.

**The labels are a wire format now.** `_JWT_SIGNING_INFO` and
`_CREDENTIAL_ENCRYPTION_INFO` below are read by every future run of this
process against every already-issued token and already-encrypted
credential. Changing either label changes the derived key: every
previously-issued JWT becomes unverifiable (every signed-in operator is
signed out) and every previously-encrypted credential becomes permanently
undecryptable (PROV-04's "losing this key makes every stored credential
permanently unreadable" applies just the same to changing the label as to
losing the key itself). Do not edit either constant after this ships.

**`validate_secret_key_strength` refuses to boot under a missing, weak, or
placeholder key** -- the same "refusal beats starting half-configured"
posture `app.py` already applies to an unmigrated schema and an
uncalibrated correlation gate (03-CONTEXT.md). A process that hashes no
passwords of its own (that is `auth/passwords.py`'s job) but signs every
session token and will encrypt every provider credential under this key
does not belong behind a warning nobody reads.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from cryptography.hazmat.primitives import hashes as _crypto_hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from fastapi import Response

from atlas.config import ConfigError, SecurityConfig, read_secret_key
from atlas.db.repository import AccountRepository

_JWT_ALGORITHM = "HS256"

# Wire format -- see the module docstring above. Do not change either value
# once any token has been issued or any credential encrypted under the key
# it labels.
_JWT_SIGNING_INFO = b"spire-voice/jwt-signing/v1"
_CREDENTIAL_ENCRYPTION_INFO = b"spire-voice/credential-encryption/v1"

_DERIVED_KEY_LENGTH = 32

# The raw byte length `cryptography.fernet.Fernet.generate_key()` encodes
# (32 raw bytes -> 44 urlsafe-base64 characters with padding) -- the shape
# `config.py::read_secret_key`'s own docstring already tells the operator to
# generate. `validate_secret_key_strength` checks the operator's value
# against this same shape, not an arbitrary length this module invents.
_EXPECTED_RAW_KEY_LENGTH = 32
_MIN_DISTINCT_BYTES = 4


def _derive_key(secret: str, info: bytes, *, length: int = _DERIVED_KEY_LENGTH) -> bytes:
    """One HKDF-SHA256 derivation, `info`-labeled -- the shared primitive
    both `_jwt_signing_key` here and plan 03-07's credential-encryption key
    call, so the two derived values can never accidentally collide."""
    hkdf = HKDF(algorithm=_crypto_hashes.SHA256(), length=length, salt=None, info=info)
    return hkdf.derive(secret.encode("utf-8"))


def _jwt_signing_key(security: SecurityConfig) -> bytes:
    return _derive_key(read_secret_key(security), _JWT_SIGNING_INFO)


def validate_secret_key_strength(security: SecurityConfig) -> None:
    """Raise `ConfigError`, naming `security.secret_key_env`, when the
    operator's secret is missing, malformed, or an obvious placeholder --
    called once, at startup, before any token is issued or any credential
    encrypted under a key derived from it.

    `read_secret_key` already refuses a missing/blank value (raised from
    inside this call, unchanged). This adds the structural checks
    `Fernet.generate_key()`'s own documented shape gives for free: the
    value must decode as `_EXPECTED_RAW_KEY_LENGTH` urlsafe-base64-encoded
    bytes with real byte variety -- not a short, truncated, or
    all-one-byte value a human might paste in by hand while testing, and
    not literal placeholder text copied from a README or an example file.
    """
    value = read_secret_key(security)
    try:
        raw = base64.urlsafe_b64decode(value.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise ConfigError(
            f"{security.secret_key_env} is not a valid urlsafe-base64 key -- generate "
            'one with `.venv/bin/python -c "from cryptography.fernet import Fernet; '
            'print(Fernet.generate_key().decode())"` and set it in the environment '
            "before startup"
        ) from exc
    if len(raw) != _EXPECTED_RAW_KEY_LENGTH:
        raise ConfigError(
            f"{security.secret_key_env} decodes to {len(raw)} byte(s), not the "
            f"{_EXPECTED_RAW_KEY_LENGTH} a generated key produces -- this looks like a "
            "placeholder or a truncated value, not a real generated secret. Generate "
            'one with `.venv/bin/python -c "from cryptography.fernet import Fernet; '
            'print(Fernet.generate_key().decode())"`'
        )
    if len(set(raw)) < _MIN_DISTINCT_BYTES:
        raise ConfigError(
            f"{security.secret_key_env} has almost no byte variety "
            f"({len(set(raw))} distinct byte value(s) across {len(raw)} bytes) -- "
            "this looks like a placeholder (e.g. all zeros), not a real generated "
            'secret. Generate one with `.venv/bin/python -c "from cryptography.fernet '
            'import Fernet; print(Fernet.generate_key().decode())"`'
        )


def issue_access_token(*, user_id: int, role: str, security: SecurityConfig) -> str:
    """Issue a short-lived JWT naming `user_id` and `role`, signed with the
    HKDF-derived signing key -- `security.access_token_ttl_s` after now."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "role": role,
        "iat": now,
        "exp": now + timedelta(seconds=security.access_token_ttl_s),
    }
    return jwt.encode(payload, _jwt_signing_key(security), algorithm=_JWT_ALGORITHM)


class InvalidAccessToken(Exception):
    """Raised instead of returned, matching `atlas_mcp.safety.Denied`'s and
    `atlas.config.ConfigError`'s own doctrine: an expired, malformed,
    or wrong-signature access token is a refusal, not a value a caller
    could silently ignore."""


def verify_access_token(token: str, security: SecurityConfig) -> dict[str, Any]:
    """Verify `token` and return its payload, or raise `InvalidAccessToken`.

    `algorithms=[_JWT_ALGORITHM]` is passed explicitly as a one-element
    list -- never `None`, and never read off the token's own `alg` header
    -- because trusting a token to name its own verification algorithm is
    exactly the class CVE-2026-48526 sits in (a public JWK accepted as an
    HMAC secret because the caller let the token's header pick the
    algorithm).
    """
    try:
        return jwt.decode(token, _jwt_signing_key(security), algorithms=[_JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise InvalidAccessToken(str(exc)) from exc


def issue_refresh_token() -> str:
    """Generate one opaque, unguessable refresh token -- never JWT-shaped,
    never carrying any claim: the whole point of "opaque" is that nothing
    about it is decodable without the database row its hash matches."""
    return secrets.token_urlsafe(32)


def hash_refresh_token(token: str) -> str:
    """The storage form of a refresh token -- SHA-256 over the opaque
    value. `refresh_tokens.token_hash` is the only place this ever lands;
    the plaintext travels in the cookie and nowhere else."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def rotate_refresh_token(
    repo: AccountRepository, presented_token: str, *, security: SecurityConfig
) -> tuple[str, int] | None:
    """Present `presented_token`, rotate it through `repo`, and return
    `(new_token, user_id)` -- or `None` when the presented token is unknown
    or was already revoked (a replay: `AccountRepository.rotate_refresh_token`
    revokes the whole chain in that case, per its own docstring, and the
    caller here must treat a `None` result as a full sign-out, never a
    retryable error). `user_id` is returned alongside the token because the
    route calling this (`POST /api/auth/refresh`) needs it to mint a new
    access token, and this is the one call that already has the row.
    """
    old_hash = hash_refresh_token(presented_token)
    new_token = issue_refresh_token()
    new_hash = hash_refresh_token(new_token)
    now = datetime.now(timezone.utc)
    result = await repo.rotate_refresh_token(
        old_hash,
        new_token_hash=new_hash,
        issued_at=now,
        expires_at=now + timedelta(seconds=security.refresh_token_ttl_s),
    )
    return (new_token, result.user_id) if result is not None else None


_REFRESH_COOKIE_SUFFIX = "_refresh"


def _refresh_cookie_name(security: SecurityConfig) -> str:
    return security.cookie_name + _REFRESH_COOKIE_SUFFIX


def set_session_cookie(
    response: Response, security: SecurityConfig, *, access_token: str, refresh_token: str
) -> None:
    """Set both session cookies -- the access-token cookie
    (`security.cookie_name`) and the refresh-token cookie (the same name
    plus `_refresh`) -- both HttpOnly, `SameSite=Lax`, `path="/"`, and
    `secure` from `SecurityConfig.cookie_secure` (T-03-28). Neither token
    is ever returned in a response body; this is the only place either
    leaves this process, and it leaves only as a cookie."""
    response.set_cookie(
        key=security.cookie_name,
        value=access_token,
        max_age=security.access_token_ttl_s,
        path="/",
        httponly=True,
        samesite="lax",
        secure=security.cookie_secure,
    )
    response.set_cookie(
        key=_refresh_cookie_name(security),
        value=refresh_token,
        max_age=security.refresh_token_ttl_s,
        path="/",
        httponly=True,
        samesite="lax",
        secure=security.cookie_secure,
    )


def clear_session_cookie(response: Response, security: SecurityConfig) -> None:
    """Clear both session cookies -- sign-out's counterpart to
    `set_session_cookie`."""
    response.delete_cookie(key=security.cookie_name, path="/")
    response.delete_cookie(key=_refresh_cookie_name(security), path="/")


def read_refresh_cookie(request_cookies: dict[str, str], security: SecurityConfig) -> str | None:
    """The refresh-token cookie's value, or `None` when absent -- the one
    place a route reads the refresh cookie's name, so `_REFRESH_COOKIE_SUFFIX`
    is never duplicated at a route call site."""
    return request_cookies.get(_refresh_cookie_name(security))
