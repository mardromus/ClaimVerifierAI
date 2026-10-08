# ClaimVerifier AI - API + web UI in one image.
#   docker build -t claimverifier .                                                    # full transformer stack (CPU torch)
#   docker build -t claimverifier-lite --build-arg REQUIREMENTS=requirements-lite.txt .   # offline lite stack

# ---- 1. build the React UI ------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build -- --outDir /web/dist

# ---- 2. Python runtime -------------------------------------------------------------------
FROM python:3.11-slim
ARG REQUIREMENTS=requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.cache/huggingface

WORKDIR /app
COPY requirements.txt requirements-lite.txt ./
# CPU-only PyTorch wheels keep the image small; drop the extra index for a CUDA base image.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r ${REQUIREMENTS}

COPY . .
COPY --from=web /web/dist ./claimverifier/web/dist
RUN pip install --no-deps -e .

EXPOSE 8000
# data/, artifacts/, models/ and reports/ are best mounted as volumes (see docker-compose.yml).
CMD ["sh", "-c", "python -m claimverifier serve --config ${CLAIMVERIFIER_CONFIG:-configs/default.yaml} --host 0.0.0.0 --port 8000"]
