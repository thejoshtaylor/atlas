"""Provider credential encryption at rest, the closed slot set, and the
one database-then-environment-then-unset resolution order (PROV-04,
D-07).

**Encryption.** `cryptography.fernet.Fernet` bundles a versioned format,
initialization-vector generation, and an integrity check -- a hand-rolled
block-cipher call without an authentication tag is the documented way to
land in a padding-oracle bug, and this module is not the place to find
that out (03-RESEARCH.md Pattern 4). The key is derived from
`read_secret_key(config.security)` through the exact same `_derive_key`
HKDF primitive, and the exact same `_CREDENTIAL_ENCRYPTION_INFO` label,
`spire_voice.auth.tokens` already reserved for this purpose -- that
module's own docstring names this reuse ("the shared primitive both
`_jwt_signing_key` here and plan 03-07's credential-encryption key
call"), so this module imports both directly from there rather than
re-deriving an equivalent constant locally, which is what keeps the two
labels provably identical rather than merely intended to match. One
operator-supplied `SPIRE_SECRET_KEY` yields two independent derived keys
this way, and changing either module's label later breaks every
already-issued token or already-encrypted credential the same way (see
`auth/tokens.py`'s own wire-format warning) -- do not edit
`_CREDENTIAL_ENCRYPTION_INFO` after this ships.

`decrypt_credential` is a server-side function, reached through
`resolve_credential_value` from two places: `app.py`'s `lifespan`, where
every provider credential is resolved once at startup, and
`routes/wizard.py`'s `check_hub_step` (plan 03-09), which resolves the
Home Assistant slot on each hub-check request to probe the operator's own
instance with it. WR-02 (code review) caught this docstring still
claiming "never reachable from a route" after that second call site
shipped -- true when this was written, false since. The guarantee that
actually holds, and that PROV-04's write-only property actually depends
on, is narrower and still true at both call sites: the decrypted value is
used only for a server-side outbound call (constructing a provider client
in `app.py`, probing Home Assistant in `check_hub_step`) and is never
included in a response body -- `check_hub_step`'s own `WizardStepStatus`
carries only `checked_at`/`entity_count` in its `detail`, never the token
it checked with. A future route that calls `resolve_credential_value` and
then places the result in anything a response model serializes would
break that guarantee; reaching `decrypt_credential` from a route at all
does not.

`key_version` travels alongside every ciphertext so a future rotation
through `cryptography.fernet.MultiFernet` has something to match
against. This phase never rotates; every write stores
`CURRENT_KEY_VERSION`, so a column added later against existing rows is
not a migration anybody has to write from scratch.

**The slot set and the resolution order.** `CredentialSlot` is the closed
set PROV-04 requires -- a route write to any other name is refused before
it ever reaches `provider_credentials`. Both live here, beside the
encryption primitives, rather than in `routes/credentials.py` where the
slot set is first used: `app.py`'s startup resolution (which slot value a
provider is actually constructed with) and `routes/credentials.py`'s
listing route (which source the browser is told a slot's value came
from) both need the *same* precedence decision, and `app.py` cannot
import from `routes/credentials.py` without a circular import (`app.py`
already imports `spire_voice.routes`, which imports
`routes/credentials.py`). This module has no dependency on either, so it
is the one place both callers can import the decision from -- `grep -rn`
for a second "database, then environment, then unset" check anywhere
else in this codebase is expected to find nothing.
"""

from __future__ import annotations

import base64
from datetime import datetime
from enum import Enum

from cryptography.fernet import Fernet, InvalidToken  # noqa: F401 -- re-exported for callers

from spire_voice.auth.tokens import _CREDENTIAL_ENCRYPTION_INFO, _derive_key
from spire_voice.config import Config, SecurityConfig, read_secret_key
from spire_voice.db.repository import CredentialRepository

CURRENT_KEY_VERSION = 1


class CredentialSlot(str, Enum):
    """The closed set of provider-credential slots (PROV-04) -- a write
    to any other name is refused before it ever reaches
    `provider_credentials`. Adding a slot is a code change, never
    something a route parameter can invent.

    Member identifiers deliberately avoid the words "key"/"token" right
    before their `=` (`STT`, not `STT_API_KEY`) -- `tests/
    test_repo_hygiene.py`'s repository-wide credential-literal scan
    pattern-matches exactly that `<name containing api_key/token/secret/
    password> = "<8+ char value>"` shape to catch a leaked real
    credential, and a slot *name* here is not one, even though it reads
    similarly. The wire values (`.value`, stored in the database and sent
    over the API) are unchanged by this -- only the Python identifier.
    """

    STT = "stt_api_key"
    BRAIN = "brain_api_key"
    TTS = "tts_api_key"
    HOME_ASSISTANT = "ha_token"


SLOT_LABELS: dict[CredentialSlot, str] = {
    CredentialSlot.STT: "Speech-to-text API key",
    CredentialSlot.BRAIN: "Language model API key",
    CredentialSlot.TTS: "Text-to-speech API key",
    CredentialSlot.HOME_ASSISTANT: "Home Assistant token",
}


def _fernet_key(security: SecurityConfig) -> bytes:
    """The HKDF-derived Fernet key -- see the module docstring for why
    this reuses `auth/tokens.py`'s own `_derive_key`/label rather than a
    locally re-derived equivalent. HKDF's own 32 raw bytes are re-encoded
    urlsafe-base64 here because that is the exact shape `Fernet` itself
    requires of a key, distinct from the raw-bytes shape `jwt.encode`
    wants for the JWT signing key derived under the sibling label."""
    raw = _derive_key(read_secret_key(security), _CREDENTIAL_ENCRYPTION_INFO)
    return base64.urlsafe_b64encode(raw)


def encrypt_credential(plaintext: str, security: SecurityConfig) -> tuple[bytes, int]:
    """Encrypt `plaintext`, returning `(ciphertext, key_version)` to
    store. The one and only place a browser-supplied credential value is
    ever handed to Fernet."""
    token = Fernet(_fernet_key(security)).encrypt(plaintext.encode("utf-8"))
    return token, CURRENT_KEY_VERSION


def decrypt_credential(ciphertext: bytes, key_version: int, security: SecurityConfig) -> str:
    """Decrypt `ciphertext`. Server-side only -- see the module docstring
    for the one call site this is ever reached from. Raises
    `cryptography.fernet.InvalidToken` on a corrupted ciphertext or one
    encrypted under a different key; raises `ValueError` on a
    `key_version` this module does not recognize -- neither case ever
    returns a value that merely looks like it worked.
    """
    if key_version != CURRENT_KEY_VERSION:
        raise ValueError(f"unsupported credential key_version: {key_version!r}")
    return Fernet(_fernet_key(security)).decrypt(ciphertext).decode("utf-8")


def env_value_for_slot(slot: CredentialSlot, config: Config) -> str:
    """The value this slot's environment expansion already produced, per
    `config` -- `config.stt.api_key`/`config.brain.api_key`/
    `config.tts.api_key` for the three provider slots (each read from its
    own config section rather than assumed to share one, even though
    `config.example.yaml` points all three at the same `${XAI_API_KEY}`
    today), and the Home Assistant child's own declared `HA_TOKEN` env
    entry for the fourth. An empty string means "no environment value,"
    never a placeholder this function invents -- this is deliberately
    reading `config`'s already-expanded fields, not re-reading
    `os.environ` a second time, so this function's answer and
    `config.py`'s own `${NAME}` expansion can never disagree about what
    "the environment" produced.
    """
    if slot is CredentialSlot.STT:
        return config.stt.api_key
    if slot is CredentialSlot.BRAIN:
        return config.brain.api_key
    if slot is CredentialSlot.TTS:
        return config.tts.api_key
    if slot is CredentialSlot.HOME_ASSISTANT:
        ha_config = config.mcp_servers.get("ha")
        return ha_config.env.get("HA_TOKEN", "") if ha_config is not None else ""
    raise ValueError(f"no environment source mapped for credential slot {slot!r}")


async def resolve_credential_source(
    slot: CredentialSlot, repo: CredentialRepository, config: Config
) -> tuple[bool, str, datetime | None]:
    """`(is_set, source, updated_at)` for `slot`, in the one order this
    application ever resolves a credential in: the database wins when a
    row is stored, `env_value_for_slot(slot, config)` wins when it is
    non-empty, otherwise the slot is unset. `updated_at` is `None` for an
    environment-sourced or unset value -- only a database row carries
    one.

    This is the one function that makes the database-vs-environment
    choice (see the module docstring); `resolve_credential_value` below
    and `routes/credentials.py`'s listing route both call this rather
    than re-deriving the same check.
    """
    credential = await repo.get_credential(slot.value)
    if credential is not None:
        return True, "database", credential.updated_at
    if env_value_for_slot(slot, config):
        return True, "environment", None
    return False, "unset", None


async def resolve_credential_value(
    slot: CredentialSlot, config: Config, repo: CredentialRepository
) -> tuple[str, str]:
    """`(value, source)` for `slot` -- decrypts the stored ciphertext when
    the database wins, reads `env_value_for_slot` when the environment
    wins, or returns `("", "unset")`. Built entirely on top of
    `resolve_credential_source`'s own precedence decision, so there is
    still exactly one place that decision is made; this function adds
    only "and here is the plaintext," which `resolve_credential_source`
    itself deliberately never returns (a route calls that one, never this
    one -- see the module docstring's PROV-04 note).
    """
    is_set, source, _updated_at = await resolve_credential_source(slot, repo, config)
    if source == "database":
        credential = await repo.get_credential(slot.value)
        assert credential is not None  # `source == "database"` implies this
        return decrypt_credential(credential.ciphertext, credential.key_version, config.security), source
    if source == "environment":
        return env_value_for_slot(slot, config), source
    return "", source
