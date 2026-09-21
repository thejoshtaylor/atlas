"""`register_routers(app)` -- the one call `app.py` makes to mount every
route this plan's account system adds, so `app.py` gains one call rather
than a hundred lines of route bodies inline in the lifespan file.

The setup gate (`require_setup_complete`, WEB-01/D-08) used to be an
`app`-level dependency, which meant `app.frontend()`'s own SPA routes
inherited it too and a fresh deployment's first page load 503'd before the
browser ever loaded the JavaScript that could clear the gate
(deferred-items.md #1, Option B). It now lives here instead, on the one
parent router every backend router is included into -- see
`register_routers`'s own docstring for why that is a structural guarantee,
not a per-router convention.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI

from spire_voice.auth.dependencies import require_setup_complete
from spire_voice.routes.accounts import router as accounts_router
from spire_voice.routes.auth import router as auth_router
from spire_voice.routes.auth import setup_router
from spire_voice.routes.credentials import router as credentials_router
from spire_voice.routes.macros import router as macros_router
from spire_voice.routes.plugins import router as plugins_router
from spire_voice.routes.policy import router as policy_router
from spire_voice.routes.providers import router as providers_router
from spire_voice.routes.sessions import router as sessions_router
from spire_voice.routes.wake import router as wake_router
from spire_voice.routes.wizard import router as wizard_router
from spire_voice.routes.workflows import router as workflows_router


def register_routers(app: FastAPI) -> None:
    """Mount every backend route behind the setup gate, structurally.

    `gated` carries `require_setup_complete` as its own constructor-level
    `dependencies`, never repeated per included router below.
    `APIRouter.add_api_route`/`add_api_websocket_route` -- what every
    `include_router` call resolves to underneath, including a nested one
    like `gated.include_router(...)` here -- always prepend the *including*
    router's own constructor dependencies onto each route being added
    (verified directly against the installed `fastapi==0.141.1`'s own
    source this session). So a router `include_router`'d onto `gated`
    inherits the gate with no opt-in of its own, and this function is the
    only legal place to mount a backend router: one added here later is
    gated by construction, and one mounted anywhere else (directly on
    `app`, bypassing this function) is not -- which is exactly why
    `app.py`'s own top-level routes carry the identical dependency
    explicitly on their own `backend_router` instead of skipping it.
    `setup_router` and the wizard/auth create-admin routes stay listed here
    too: `require_setup_complete` itself still exempts their exact paths
    (`SETUP_GATE_EXEMPT_PATHS`), so running the dependency on them is a
    no-op, not a second gate.
    """
    gated = APIRouter(dependencies=[Depends(require_setup_complete)])
    gated.include_router(auth_router)
    gated.include_router(setup_router)
    gated.include_router(accounts_router)
    gated.include_router(policy_router)
    gated.include_router(macros_router)
    gated.include_router(credentials_router)
    gated.include_router(wizard_router)
    gated.include_router(workflows_router)
    gated.include_router(plugins_router)
    gated.include_router(providers_router)
    gated.include_router(sessions_router)
    gated.include_router(wake_router)
    app.include_router(gated)
