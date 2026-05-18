"""
api/v1/routes/health.py — GET /api/v1/health

Liveness and readiness probe used by:
  • Render / Railway / Fly.io — health-check URL to decide whether to route traffic
  • Android app — optional pre-flight check before sending analyse requests
  • CI pipelines — smoke test after deploying a new model artefact

── Example curl ──────────────────────────────────────────────────────────────

  curl -s https://<your-host>/api/v1/health | python -m json.tool

  # Expected response (model trained):
  {
    "status": "ok",
    "model_loaded": true,
    "version": "0.1.0"
  }

  # Expected response (model not yet trained — API still works):
  {
    "status": "ok",
    "model_loaded": false,
    "version": "0.1.0"
  }
"""

from fastapi import APIRouter
from loguru import logger

from app.core.config import settings
from app.models.ml.scam_classifier import MODEL_PATH

router = APIRouter()


@router.get(
    "/health",
    status_code=200,
    summary="Liveness and readiness probe",
    description=(
        "Returns `status: 'ok'` as long as the process is alive.\n\n"
        "`model_loaded: true` indicates the trained `.joblib` artefact is present "
        "and the classifier is ready.  `false` means the API is running but will "
        "return `label='unknown'` until the model is trained and the server restarted."
    ),
    tags=["System"],
)
def health() -> dict:
    """
    Lightweight health probe — no ML inference, no database calls.

    Configure this URL as the health-check path in Render's service settings:
      Path : /api/v1/health
      Method: GET
      Expected status: 200
    """
    model_loaded = MODEL_PATH.exists()

    logger.debug("GET /health | model_loaded={}", model_loaded)

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "version": settings.VERSION,
    }
