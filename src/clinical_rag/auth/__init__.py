"""JWT / demo-token auth for secure FHIR and summary APIs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from clinical_rag.config import get_settings

bearer = HTTPBearer(auto_error=False)


def create_access_token(subject: str, roles: list[str] | None = None) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "roles": roles or ["clinician"],
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from exc


def require_auth(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer)],
) -> dict[str, Any]:
    settings = get_settings()
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization bearer token")

    token = credentials.credentials
    if token == settings.demo_api_token:
        return {"sub": "demo-user", "roles": ["clinician", "coder", "admin"]}

    return decode_token(token)


AuthUser = Annotated[dict[str, Any], Depends(require_auth)]
