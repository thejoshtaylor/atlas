"""`register_routers(app)` -- the one call `app.py` makes to mount every
route this plan's account system adds, so `app.py` gains one call rather
than a hundred lines of route bodies inline in the lifespan file.
"""

from __future__ import annotations

from fastapi import FastAPI

from spire_voice.routes.accounts import router as accounts_router
from spire_voice.routes.auth import router as auth_router
from spire_voice.routes.auth import setup_router


def register_routers(app: FastAPI) -> None:
    app.include_router(auth_router)
    app.include_router(setup_router)
    app.include_router(accounts_router)
