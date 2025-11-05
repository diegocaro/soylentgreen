# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /app

# Set work directory
WORKDIR /app

# Install the application dependencies.
RUN uv sync --frozen --no-cache --no-dev --no-default-groups

# Expose port
EXPOSE 8000

# Run the application.
CMD ["uv", "run", "--no-sync", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
