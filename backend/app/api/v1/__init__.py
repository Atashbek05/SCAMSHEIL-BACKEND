"""
api/v1/__init__.py — Aggregates all v1 route modules into a single router.

Route registry
--------------
  POST /api/v1/analyze   — scam detection (primary endpoint)
  GET  /api/v1/health    — liveness/readiness probe

Add new route modules below as features are built out.
"""

from fastapi import APIRouter

from app.api.v1.routes import analyze, health

router = APIRouter()

# Primary scam-detection endpoint
router.include_router(analyze.router, tags=["Detection"])

# Health / readiness probe (used by Render, CI, and Android pre-flight checks)
router.include_router(health.router, tags=["System"])

# Future route modules go here, e.g.:
# from app.api.v1.routes import reports, users
# router.include_router(reports.router, prefix="/reports", tags=["Reports"])
