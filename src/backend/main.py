from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.backend.backend_api import router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:6379"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)