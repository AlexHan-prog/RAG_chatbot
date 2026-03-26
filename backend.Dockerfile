FROM python:3.13

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir .

COPY . .

EXPOSE 8000
# starts FastAPI with auto-reload (remove this for production)
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]