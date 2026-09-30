"""
SMARTLEITO API
Gestão inteligente de leitos hospitalares
"""

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
