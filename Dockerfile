# 1. BASE IMAGE
FROM python:3.11-slim

# 2. OPTIMIZATION
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# 3. SYSTEM DEPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 curl \
    && rm -rf /var/lib/apt/lists/*

# 4. DEPENDENCIES
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. SECURITY: Create user and directories early
RUN useradd -m blossom_user && \
    mkdir -p /app/chroma_db /app/data

# 6. APP CODE: Copy files and SET OWNERSHIP in one layer
# This prevents 'root' from owning your source code and data folders
COPY --chown=blossom_user:blossom_user . .

# Final safety check: ensure the DB folder is fully writable
RUN chmod -R 775 /app/chroma_db /app/data

# 8. HEALTHCHECK
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER blossom_user

# 9. START COMMAND
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "65"]

USER blossom_user