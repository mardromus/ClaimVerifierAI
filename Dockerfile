# ClaimVerifier AI - API + web UI image.
#   docker build -t claimverifier .                                         # full transformer stack (CPU torch)
#   docker build -t claimverifier-lite --build-arg REQUIREMENTS=requirements-lite.txt .   # offline lite stack
FROM python:3.11-slim

ARG REQUIREMENTS=requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.cache/huggingface

WORKDIR /app
COPY requirements.txt requirements-lite.txt ./
# CPU-only PyTorch wheels keep the image small; drop the extra index for a CUDA base image.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r ${REQUIREMENTS}

COPY . .
RUN pip install --no-deps -e .

EXPOSE 8000 8501
# data/, artifacts/ and models/ are best mounted as volumes (see docker-compose.yml).
CMD ["python", "-m", "claimverifier", "serve", "--config", "configs/default.yaml", "--host", "0.0.0.0", "--port", "8000"]
