#!/bin/bash
set -e

echo "Starting Nano Dispersion Analysis Service..."

if [ ! -d "/app/data/batches" ]; then
    mkdir -p /app/data/batches
fi

if [ ! -d "/app/results" ]; then
    mkdir -p /app/results
fi

if [ "${NANO_DISPERSION_GENERATE_SAMPLES:-false}" = "true" ]; then
    echo "Generating sample experiment data..."
    python -m nano_dispersion.cli.main init-samples --output-dir /app/data/batches --num-batches 4 || true
    echo "Sample data generation complete."
fi

echo "Starting API server on port ${NANO_DISPERSION_PORT:-6817}..."
exec "$@"
