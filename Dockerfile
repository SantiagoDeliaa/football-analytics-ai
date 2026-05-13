FROM node:22-alpine AS frontend-builder

WORKDIR /build/front-tip

COPY front-tip/package*.json ./
RUN npm ci

COPY front-tip/ ./
RUN npm run build

FROM python:3.11-slim

RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user api/ ./api/
COPY --chown=user src/ ./src/
COPY --chown=user app.py ./
COPY --chown=user front-tip/ ./front-tip/
COPY --from=frontend-builder --chown=user /build/front-tip/dist ./front-tip/dist
RUN install -d -o user -g user $HOME/app/outputs/api $HOME/app/data

USER user

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/api/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
