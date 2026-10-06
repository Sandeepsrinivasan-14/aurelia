FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY mock_ehr ./mock_ehr
COPY eval ./eval
COPY ui ./ui
RUN pip install --no-cache-dir . && useradd -m aurelia && mkdir -p data && chown -R aurelia data
USER aurelia
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "aurelia.api.app:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
