"""
api/v1/__init__.py — Aggregates all v1 route modules into a single router.

Add new route modules here as features are built out.
"""

from fastapi import APIRouter

from app.api.v1.routes import analyze

router = APIRouter()

# Detection — POST /api/v1/analyze
router.include_router(analyze.router, tags=["Detection"])

# Future route modules go here, e.g.:
# from app.api.v1.routes import reports, users
# router.include_router(reports.router, prefix="/reports", tags=["Reports"])
# router.include_router(users.router,   prefix="/users",   tags=["Users"])
