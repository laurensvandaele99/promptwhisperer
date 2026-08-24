FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    NLTK_DATA=/opt/nltk_data

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY model/requirements_model.txt /app/model/requirements_model.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt -r /app/model/requirements_model.txt

# Cache the exact text resources needed by the notebook-derived feature pipeline.
RUN python -m nltk.downloader -d /opt/nltk_data vader_lexicon
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . /app

ENV MODEL_PATH=/app/model/model_bundle.joblib
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
