# ---------------------------------------------------------------------------
# Backend FastAPI + CLIP multilingüe, listo para Hugging Face Spaces (Docker).
#
#   docker build -t image-search .
#   docker run --rm -p 7860:7860 --env-file .env image-search
#
# El modelo CLIP se descarga en tiempo de build para que el contenedor
# arranque en segundos en lugar de bajar ~1 GB en cada reinicio.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=7860

# Usuario sin privilegios (requisito de Hugging Face Spaces: uid 1000).
RUN useradd -m -u 1000 appuser
WORKDIR /app

# 1) PyTorch CPU primero, desde el índice oficial (evita las ruedas CUDA de varios GB).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# 2) Resto de dependencias.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Pre-descarga del modelo CLIP multilingüe en la caché de Hugging Face.
ARG MODEL_NAME=xlm-roberta-base-ViT-B-32
ARG PRETRAINED=laion5b_s13b_b90k
RUN mkdir -p "$HF_HOME" && python -c "import open_clip; \
        open_clip.create_model_and_transforms('${MODEL_NAME}', pretrained='${PRETRAINED}'); \
        open_clip.get_tokenizer('${MODEL_NAME}')" \
    && chown -R appuser:appuser /app

# A partir de aquí todo está en caché: sin llamadas a huggingface.co en el arranque.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 4) Código.
COPY --chown=appuser:appuser backend ./backend
COPY --chown=appuser:appuser scripts ./scripts

USER appuser
RUN mkdir -p /app/data/images /app/data/embeddings

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/health')" || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
