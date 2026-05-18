"""
main.py — Application entry point.

Bootstraps the FastAPI app, registers routers, and starts the Uvicorn server.
All configuration is read from app/core/config.py (env-driven).

Android integration notes:
  - Android's OkHttp does NOT send an Origin header for native app requests,
    so CORS restrictions are effectively bypassed for native clients.
  - For development, ALLOWED_ORIGINS is set to ["*"] when DEBUG=True so that
    browser-based tools (Swagger UI, Postman web) can also call the API freely.
  - In production, restrict ALLOWED_ORIGINS in your .env file to your actual
    frontend domains; the Android native app will work regardless.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "ScamShield — AI-powered scam detection API.\n\n"
            "**Primary endpoint:** `POST /api/v1/analyze`\n\n"
            "Send any text message; get back a scam probability score, "
            "a human-readable label, and the suspicious keywords detected."
        ),
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )

    # CORS — allow all origins in DEBUG so Swagger UI and Postman work out of the box.
    # Android native apps don't send Origin headers, so CORS doesn't block them either way.
    allowed_origins = ["*"] if settings.DEBUG else settings.ALLOWED_ORIGINS

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,  # credentials=True is incompatible with allow_origins=["*"]
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    # Register API routers (imported here to avoid circular imports)
    from app.api.v1 import router as api_v1_router
    app.include_router(api_v1_router, prefix="/api/v1")

    return app


app = create_app()


# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
