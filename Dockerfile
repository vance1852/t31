FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NANO_DISPERSION_PORT=6817 \
    NANO_DISPERSION_DB_PATH=/app/data/nano_dispersion.db \
    NANO_DISPERSION_DATA_DIR=/app/data/batches \
    NANO_DISPERSION_RESULT_DIR=/app/results \
    NANO_DISPERSION_HOST=0.0.0.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]" && \
    apt-get purge -y gcc g++ && \
    apt-get autoremove -y

COPY nano_dispersion/ ./nano_dispersion/
RUN mkdir -p /app/data/batches /app/results

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 6817

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "nano_dispersion.api.main:app", "--host", "0.0.0.0", "--port", "6817"]
