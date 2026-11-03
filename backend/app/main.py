"""
SMARTLEITO API
Gestão inteligente de leitos hospitalares
"""
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import leitos, previsao

app = FastAPI(
    title="SMARTLEITO API",
    description=(
        "API para controle de leitos, previsão de alta e previsão de demanda. "
        "Stack: Python + FastAPI + Random Forest / heurísticas."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

_ORIGENS_PADRAO = (
    "http://localhost:8501,"
    "http://localhost:3000,"
    "http://localhost:5173"
)
ALLOWED_ORIGINS = [
    origem.strip()
    for origem in os.getenv(
        "ALLOWED_ORIGINS",
        _ORIGENS_PADRAO
    ).split(",")
    if origem.strip()
]

_ALLOW_CREDENTIALS = "*" not in ALLOWED_ORIGINS

if not _ALLOW_CREDENTIALS:
    logging.getLogger("uvicorn.error").warning(
        "ALLOWED_ORIGINS contém '*': allow_credentials foi desativado."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(leitos.router, prefix="/api/v1")
app.include_router(previsao.router, prefix="/api/v1")


@app.get("/", tags=["Health"])
def root():
    return {
        "projeto": "SMARTLEITO",
        "status": "online",
        "docs": "/docs",
        "versao": "1.0.0",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
